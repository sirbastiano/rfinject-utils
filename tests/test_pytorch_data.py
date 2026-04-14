from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import zarr

torch = pytest.importorskip("torch")

from rfinject.pytorch_data import RFInjectSplitBurstDataset
import rfinject.pytorch_data as pytorch_data
from rfinject.utils import download_hf_bucket_path, open_hf_bucket_zarr


@dataclass
class FakeBucketEntry:
    path: str
    type: str = "file"
    size: int | None = None


class FakeBucketApi:
    def __init__(self, bucket_root: Path):
        self.bucket_root = bucket_root
        self.bucket_id = "org/demo-bucket"
        self.file_map = {
            str(path.relative_to(bucket_root)): path
            for path in bucket_root.rglob("*")
            if path.is_file()
        }
        self.root_entries = self._build_root_entries()

    def _build_root_entries(self) -> list[FakeBucketEntry]:
        entries = []
        seen_folders = set()
        for relative_path, absolute_path in sorted(self.file_map.items()):
            top_level = relative_path.split("/", 1)[0]
            if "/" in relative_path:
                if top_level not in seen_folders:
                    entries.append(FakeBucketEntry(path=top_level, type="folder"))
                    seen_folders.add(top_level)
            else:
                entries.append(
                    FakeBucketEntry(path=top_level, type="file", size=absolute_path.stat().st_size)
                )
        return entries

    def list_bucket_tree(
        self,
        bucket_id: str,
        prefix: str | None = None,
        recursive: bool | None = None,
        token=None,
    ):
        del token
        assert bucket_id == self.bucket_id

        if recursive is False:
            if prefix is not None:
                raise AssertionError("Prefix-based non-recursive listing is not expected in these tests.")
            yield from self.root_entries
            return

        for relative_path, absolute_path in sorted(self.file_map.items()):
            if prefix is not None and not relative_path.startswith(prefix):
                continue
            yield FakeBucketEntry(path=relative_path, type="file", size=absolute_path.stat().st_size)

    def get_bucket_paths_info(self, bucket_id: str, paths: list[str], token=None):
        del token
        assert bucket_id == self.bucket_id
        for path in paths:
            if path in self.file_map:
                yield FakeBucketEntry(path=path, type="file", size=self.file_map[path].stat().st_size)

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[FakeBucketEntry | str, Path]],
        raise_on_missing_files: bool = True,
        token=None,
    ):
        del raise_on_missing_files, token
        assert bucket_id == self.bucket_id
        for source, destination in files:
            remote_path = source.path if isinstance(source, FakeBucketEntry) else source
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.file_map[remote_path].read_bytes())


def _build_fake_bucket(bucket_root: Path, *, burst_count: int = 3) -> tuple[FakeBucketApi, str]:
    scene_name = "scene.zarr"
    scene_root = bucket_root / scene_name
    group = zarr.open_group(scene_root, mode="w")
    group.attrs["l0_name"] = "synthetic"

    for burst_idx in range(burst_count):
        burst = group.create_group(f"burst_{burst_idx}")
        echo = np.full((2, 3), burst_idx + 1, dtype=np.float32)
        rfi = np.full((2, 2, 3), (burst_idx + 1) * 10, dtype=np.float32)
        burst.create_array("echo", data=echo, chunks=(1, 3))
        burst.create_array("rfi", data=rfi, chunks=(1, 2, 3))

    return FakeBucketApi(bucket_root), scene_name


def _patch_bucket_access(monkeypatch: pytest.MonkeyPatch, api: FakeBucketApi) -> None:
    monkeypatch.setattr(
        pytorch_data,
        "open_hf_bucket_zarr",
        lambda bucket, zarr_path, local_dir, metadata_only=True: open_hf_bucket_zarr(
            bucket=bucket,
            zarr_path=zarr_path,
            local_dir=local_dir,
            metadata_only=metadata_only,
            api=api,
        ),
    )
    monkeypatch.setattr(
        pytorch_data,
        "download_hf_bucket_path",
        lambda bucket, remote_path, local_dir: download_hf_bucket_path(
            bucket=bucket,
            remote_path=remote_path,
            local_dir=local_dir,
            api=api,
        ),
    )


def _payload_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.name != "zarr.json"
    )


@pytest.fixture
def fake_bucket(tmp_path: Path) -> tuple[FakeBucketApi, str]:
    return _build_fake_bucket(tmp_path / "bucket")


def test_prefetch_selected_bursts_downloads_only_the_requested_subset(
    fake_bucket: tuple[FakeBucketApi, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    api, scene_name = fake_bucket
    _patch_bucket_access(monkeypatch, api)

    dataset = RFInjectSplitBurstDataset(
        [{"scene_path": scene_name, "geo_section": "north", "parent_name": "scene"}],
        split_name="train",
        bucket=api.bucket_id,
        cache_dir=tmp_path / "data",
        sample_fraction=0.3,
        prefetch_selected_bursts=True,
        allow_remote_fetch=False,
    )

    assert [record["burst_name"] for record in dataset.records] == ["burst_0"]
    assert _payload_files(tmp_path / "data" / scene_name / "burst_0")
    assert not _payload_files(tmp_path / "data" / scene_name / "burst_1")

    monkeypatch.setattr(
        pytorch_data,
        "download_hf_bucket_path",
        lambda *args, **kwargs: pytest.fail("dataset access should be local-only after prefetch"),
    )

    sample = dataset[0]

    assert sample["scene_path"] == scene_name
    assert sample["burst_name"] == "burst_0"
    assert tuple(sample["echo"].shape) == (1, 2, 3)
    assert tuple(sample["rfi"].shape) == (1, 2, 3)
    assert sample["echo"].dtype == torch.float32


def test_local_only_mode_raises_when_selected_payload_is_not_prefetched(
    fake_bucket: tuple[FakeBucketApi, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    api, scene_name = fake_bucket
    _patch_bucket_access(monkeypatch, api)

    dataset = RFInjectSplitBurstDataset(
        [{"scene_path": scene_name, "geo_section": "north", "parent_name": "scene"}],
        split_name="train",
        bucket=api.bucket_id,
        cache_dir=tmp_path / "data",
        sample_fraction=0.3,
        prefetch_selected_bursts=False,
        allow_remote_fetch=False,
    )

    assert [record["burst_name"] for record in dataset.records] == ["burst_0"]
    assert not _payload_files(tmp_path / "data" / scene_name / "burst_0")

    with pytest.raises(FileNotFoundError, match="Local payload"):
        dataset[0]


def test_lazy_remote_fetch_mode_still_downloads_selected_bursts_on_first_access(
    fake_bucket: tuple[FakeBucketApi, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    api, scene_name = fake_bucket
    _patch_bucket_access(monkeypatch, api)

    dataset = RFInjectSplitBurstDataset(
        [{"scene_path": scene_name, "geo_section": "north", "parent_name": "scene"}],
        split_name="train",
        bucket=api.bucket_id,
        cache_dir=tmp_path / "data",
        sample_fraction=0.3,
        prefetch_selected_bursts=False,
        allow_remote_fetch=True,
    )

    assert not _payload_files(tmp_path / "data" / scene_name / "burst_0")

    sample = dataset[0]

    assert _payload_files(tmp_path / "data" / scene_name / "burst_0")
    assert sample["burst_name"] == "burst_0"
