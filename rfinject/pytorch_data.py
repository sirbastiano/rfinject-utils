from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import zarr

from .utils import (
    DEFAULT_HF_BUCKET_ID,
    access_array_data,
    download_hf_bucket_path,
    open_hf_bucket_zarr,
)

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset

    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in lightweight environments
    torch = F = None
    Dataset = None
    _TORCH_IMPORT_ERROR = exc


def _require_torch() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "PyTorch is required to use rfinject.pytorch_data. "
            "Install torch in the active environment."
        ) from _TORCH_IMPORT_ERROR


DatasetBase = Dataset if Dataset is not None else object


def burst_sort_key(burst_name: str):
    suffix = burst_name.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else burst_name


def estimate_burst_payload_bytes(metadata_group: zarr.Group, burst_name: str) -> int:
    burst_group = metadata_group[burst_name]
    total_bytes = 0
    for item_name in burst_group.keys():
        item = burst_group[item_name]
        if hasattr(item, "nbytes"):
            total_bytes += int(item.nbytes)
    return total_bytes


def select_bursts_by_fraction(
    metadata_group: zarr.Group,
    burst_names: list[str],
    sample_fraction: float,
):
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in the interval (0, 1].")

    burst_sizes = {
        burst_name: estimate_burst_payload_bytes(metadata_group, burst_name)
        for burst_name in burst_names
    }
    total_product_bytes = sum(burst_sizes.values())
    target_bytes = total_product_bytes * sample_fraction

    selected_bursts = []
    selected_bytes = 0
    for burst_name in burst_names:
        selected_bursts.append(burst_name)
        selected_bytes += burst_sizes[burst_name]
        if selected_bytes >= target_bytes:
            break

    actual_fraction = selected_bytes / total_product_bytes if total_product_bytes else 0.0
    return burst_sizes, total_product_bytes, target_bytes, selected_bursts, selected_bytes, actual_fraction


def complex_or_real_to_tensor(array: np.ndarray) -> torch.Tensor:
    _require_torch()
    array = np.asarray(array)
    if np.iscomplexobj(array):
        stacked = np.stack([array.real, array.imag], axis=0)
    else:
        stacked = np.expand_dims(array, axis=0)
    return torch.from_numpy(stacked.astype(np.float32, copy=False))


def pad_spatial_tensors(tensors: Iterable[torch.Tensor], pad_value: float = 0.0):
    _require_torch()
    tensors = list(tensors)
    if not tensors:
        raise ValueError("tensors must contain at least one element.")

    max_height = max(tensor.shape[-2] for tensor in tensors)
    max_width = max(tensor.shape[-1] for tensor in tensors)

    padded = []
    masks = []
    shapes = []
    for tensor in tensors:
        height, width = tensor.shape[-2:]
        shapes.append((height, width))
        padded.append(F.pad(tensor, (0, max_width - width, 0, max_height - height), value=pad_value))

        mask = torch.zeros((max_height, max_width), dtype=torch.bool)
        mask[:height, :width] = True
        masks.append(mask)

    return torch.stack(padded), torch.stack(masks), torch.tensor(shapes, dtype=torch.int64)


def pad_burst_batch(batch):
    echo, echo_mask, echo_shape = pad_spatial_tensors(sample["echo"] for sample in batch)
    rfi, rfi_mask, rfi_shape = pad_spatial_tensors(sample["rfi"] for sample in batch)
    return {
        "split": [sample["split"] for sample in batch],
        "scene_path": [sample["scene_path"] for sample in batch],
        "geo_section": [sample["geo_section"] for sample in batch],
        "parent_name": [sample["parent_name"] for sample in batch],
        "burst_name": [sample["burst_name"] for sample in batch],
        "echo": echo,
        "rfi": rfi,
        "echo_mask": echo_mask,
        "rfi_mask": rfi_mask,
        "echo_shape": echo_shape,
        "rfi_shape": rfi_shape,
    }


def tensor_magnitude(tensor: torch.Tensor) -> np.ndarray:
    _require_torch()
    tensor = tensor.detach().cpu()
    if tensor.shape[0] == 1:
        return tensor[0].numpy()
    return torch.sqrt((tensor**2).sum(dim=0)).numpy()


def describe_access_mode(
    *,
    download_full_scene: bool,
    prefetch_selected_bursts: bool,
    allow_remote_fetch: bool,
) -> str:
    if download_full_scene:
        return "full scenes cached up front"
    if prefetch_selected_bursts:
        return "selected bursts cached up front"
    if allow_remote_fetch:
        return "metadata first, burst payload on demand"
    return "local cache only"


class RFInjectSplitBurstDataset(DatasetBase):
    """Burst-level dataset backed by a geographic RFInject split."""

    def __init__(
        self,
        scene_entries: list[dict[str, str]],
        *,
        split_name: str,
        bucket: str = DEFAULT_HF_BUCKET_ID,
        cache_dir: str | Path | None = None,
        download_full_scene: bool = False,
        sample_fraction: float = 1.0,
        rfi_channel: int = 0,
        prefetch_selected_bursts: bool = True,
        allow_remote_fetch: bool = False,
    ):
        _require_torch()
        super().__init__()

        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be in the interval (0, 1].")
        if not scene_entries:
            raise ValueError("scene_entries must contain at least one bucket-available scene.")

        self.split_name = split_name
        self.bucket = bucket
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else Path("data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download_full_scene = download_full_scene
        self.sample_fraction = sample_fraction
        self.rfi_channel = rfi_channel
        self.prefetch_selected_bursts = prefetch_selected_bursts
        self.allow_remote_fetch = allow_remote_fetch
        self.scene_entries = [dict(entry) for entry in scene_entries]
        self.scene_lookup = {
            entry["scene_path"]: dict(entry)
            for entry in self.scene_entries
        }
        self.scene_summaries = []
        self.records = []
        self.total_available_bytes = 0
        self.total_selected_bytes = 0
        self._downloaded_bursts = set()
        self._downloaded_scenes = set()

        for scene_path in sorted(self.scene_lookup):
            scene_entry = self.scene_lookup[scene_path]
            metadata_group = open_hf_bucket_zarr(
                self.bucket,
                scene_path,
                local_dir=self.cache_dir,
                metadata_only=True,
            )
            burst_names = sorted(
                (name for name in metadata_group.keys() if name.startswith("burst_")),
                key=burst_sort_key,
            )
            if not burst_names:
                raise RuntimeError(f"No bursts were found in {scene_path}.")

            (
                burst_sizes,
                total_product_bytes,
                target_bytes,
                selected_bursts,
                selected_bytes,
                selected_fraction,
            ) = select_bursts_by_fraction(metadata_group, burst_names, self.sample_fraction)

            self.total_available_bytes += total_product_bytes
            self.total_selected_bytes += selected_bytes
            self.scene_summaries.append(
                {
                    "scene_path": scene_path,
                    "geo_section": scene_entry["geo_section"],
                    "parent_name": scene_entry["parent_name"],
                    "burst_count": len(burst_names),
                    "selected_burst_count": len(selected_bursts),
                    "total_product_bytes": total_product_bytes,
                    "selected_bytes": selected_bytes,
                    "target_bytes": target_bytes,
                    "selected_fraction": selected_fraction,
                    "burst_sizes": burst_sizes,
                }
            )

            for burst_name in selected_bursts:
                self.records.append(
                    {
                        "scene_path": scene_path,
                        "geo_section": scene_entry["geo_section"],
                        "parent_name": scene_entry["parent_name"],
                        "burst_name": burst_name,
                    }
                )

            if self.download_full_scene:
                download_hf_bucket_path(self.bucket, scene_path, local_dir=self.cache_dir)
                self._downloaded_scenes.add(scene_path)
            elif self.prefetch_selected_bursts:
                for burst_name in selected_bursts:
                    self._download_burst_payload(scene_path, burst_name)

        if not self.records:
            raise RuntimeError(f"No bursts were selected for split {self.split_name}.")

    def __len__(self) -> int:
        return len(self.records)

    def _scene_local_path(self, scene_path: str) -> Path:
        return self.cache_dir / scene_path

    def _open_scene(self, scene_path: str) -> zarr.Group:
        return zarr.open_group(str(self._scene_local_path(scene_path)), mode="r")

    def _download_burst_payload(self, scene_path: str, burst_name: str) -> None:
        download_key = (scene_path, burst_name)
        if scene_path in self._downloaded_scenes or download_key in self._downloaded_bursts:
            return

        download_hf_bucket_path(
            self.bucket,
            f"{scene_path}/{burst_name}",
            local_dir=self.cache_dir,
        )
        self._downloaded_bursts.add(download_key)

    def _burst_payload_present(self, scene_path: str, burst_name: str) -> bool:
        burst_root = self._scene_local_path(scene_path) / burst_name
        if not burst_root.exists():
            return False
        return any(path.is_file() and path.name != "zarr.json" for path in burst_root.rglob("*"))

    def _ensure_burst_payload(self, scene_path: str, burst_name: str) -> None:
        download_key = (scene_path, burst_name)
        if scene_path in self._downloaded_scenes or download_key in self._downloaded_bursts:
            return

        if self._burst_payload_present(scene_path, burst_name):
            self._downloaded_bursts.add(download_key)
            return

        if self.allow_remote_fetch:
            self._download_burst_payload(scene_path, burst_name)
            return

        raise FileNotFoundError(
            f"Local payload for {scene_path}/{burst_name} is missing under {self.cache_dir}. "
            "Enable prefetch_selected_bursts or allow_remote_fetch, or re-run the prefetch step."
        )

    def __getitem__(self, index: int):
        record = self.records[index]
        scene_path = record["scene_path"]
        burst_name = record["burst_name"]
        self._ensure_burst_payload(scene_path, burst_name)

        scene = self._open_scene(scene_path)
        echo_array = np.asarray(access_array_data(scene, burst_name, "echo"))
        rfi_array = access_array_data(scene, burst_name, "rfi")

        if self.rfi_channel >= rfi_array.shape[0]:
            raise IndexError(
                f"rfi_channel={self.rfi_channel} is out of bounds for burst {burst_name} "
                f"with {rfi_array.shape[0]} channels."
            )

        rfi_slice = np.asarray(rfi_array[self.rfi_channel])

        return {
            "split": self.split_name,
            "scene_path": scene_path,
            "geo_section": record["geo_section"],
            "parent_name": record["parent_name"],
            "burst_name": burst_name,
            "echo": complex_or_real_to_tensor(echo_array),
            "rfi": complex_or_real_to_tensor(rfi_slice),
        }


__all__ = [
    "RFInjectSplitBurstDataset",
    "burst_sort_key",
    "complex_or_real_to_tensor",
    "describe_access_mode",
    "estimate_burst_payload_bytes",
    "pad_burst_batch",
    "pad_spatial_tensors",
    "select_bursts_by_fraction",
    "tensor_magnitude",
]
