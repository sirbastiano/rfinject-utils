from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rfinject.trainer as trainer
from rfinject.utils import (
    access_array_data,
    access_attributes,
    download_hf_bucket_path,
    explore_all_attributes,
    explore_zarr_structure,
    get_array_slice,
    get_burst_info,
    get_hf_bucket_info,
    list_hf_bucket_files,
    list_hf_bucket_zarrs,
    open_hf_bucket_zarr,
    parse_hf_bucket_reference,
    sync_hf_bucket_zarr,
)
from rfinject.viz import plot_complex_array, plot_magnitude


@dataclass
class FakeBucketEntry:
    path: str
    type: str = "file"
    size: int | None = None


@dataclass
class FakeBucketInfo:
    id: str
    total_files: int
    size: int


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

    def bucket_info(self, bucket_id: str, token=None):
        del token
        assert bucket_id == self.bucket_id
        size = sum(path.stat().st_size for path in self.file_map.values())
        return FakeBucketInfo(id=bucket_id, total_files=len(self.file_map), size=size)

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
                raise AssertionError("Prefix-based non-recursive listing is not used in tests.")
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
            data = self.file_map[remote_path].read_bytes()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)


def _build_fake_bucket(bucket_root: Path) -> tuple[FakeBucketApi, str]:
    readme_path = bucket_root / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text("bucket readme")

    scene_name = "scene.zarr"
    scene_root = bucket_root / scene_name
    group = zarr.open_group(scene_root, mode="w")
    group.attrs["l0_name"] = "synthetic"
    group.attrs["ephemeris"] = [{"time_stamp": 1, "x": 0.0, "y": 1.0, "z": 2.0}]

    burst = group.create_group("burst_0")
    burst.attrs["metadata"] = [{"burst_id": 0, "polarization": "HH"}]
    burst.attrs["rfi_params"] = [{"snr_db": 12.5, "kind": "tone"}]

    echo = burst.create_array(
        "echo",
        data=np.arange(6, dtype=np.float32).reshape(2, 3),
        chunks=(1, 3),
    )
    echo.attrs["units"] = "arbitrary"

    rfi = burst.create_array(
        "rfi",
        data=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        chunks=(1, 2, 3),
    )
    rfi.attrs["channels"] = 2

    return FakeBucketApi(bucket_root), scene_name


@pytest.fixture
def local_zarr(tmp_path: Path) -> zarr.Group:
    api, scene_name = _build_fake_bucket(tmp_path / "bucket")
    del api
    return zarr.open_group(tmp_path / "bucket" / scene_name, mode="r")


@pytest.fixture
def fake_bucket(tmp_path: Path) -> tuple[FakeBucketApi, str]:
    return _build_fake_bucket(tmp_path / "bucket")


def test_parse_hf_bucket_reference_accepts_urls_and_ids():
    assert parse_hf_bucket_reference("ESA-philab/RFInject-v1-L0") == "ESA-philab/RFInject-v1-L0"
    assert parse_hf_bucket_reference("buckets/ESA-philab/RFInject-v1-L0") == "ESA-philab/RFInject-v1-L0"
    assert (
        parse_hf_bucket_reference("https://huggingface.co/buckets/ESA-philab/RFInject-v1-L0")
        == "ESA-philab/RFInject-v1-L0"
    )


def test_parse_hf_bucket_reference_rejects_invalid_values():
    with pytest.raises(ValueError):
        parse_hf_bucket_reference("")

    with pytest.raises(ValueError):
        parse_hf_bucket_reference("ESA-philab")


def test_local_zarr_helpers_cover_structure_and_attributes(local_zarr: zarr.Group, capsys):
    explore_zarr_structure(local_zarr)
    printed = capsys.readouterr().out
    assert "📁 burst_0/" in printed
    assert "📄 echo:" in printed

    echo_array = access_array_data(local_zarr, "burst_0", "echo")
    rfi_array = access_array_data(local_zarr, "burst_0", "rfi")

    assert echo_array.shape == (2, 3)
    assert rfi_array.shape == (2, 2, 3)
    assert np.array_equal(get_array_slice(echo_array), np.arange(6, dtype=np.float32).reshape(2, 3))
    assert np.array_equal(get_array_slice(rfi_array), np.arange(6, dtype=np.float32).reshape(2, 3))

    root_attrs = access_attributes(local_zarr)
    burst_attrs = access_attributes(local_zarr, "burst_0")
    echo_attrs = access_attributes(local_zarr, "burst_0/echo")
    all_attrs = explore_all_attributes(local_zarr)
    burst_info = get_burst_info(local_zarr)

    assert root_attrs["l0_name"] == "synthetic"
    assert burst_attrs["metadata"][0]["burst_id"] == 0
    assert echo_attrs["units"] == "arbitrary"
    assert {"root", "burst_0", "burst_0/echo", "burst_0/rfi"} <= set(all_attrs)
    assert burst_info["burst_0"]["arrays"]["echo"]["shape"] == (2, 3)
    assert burst_info["burst_0"]["arrays"]["rfi"]["shape"] == (2, 2, 3)


def test_access_attributes_raises_on_invalid_path(local_zarr: zarr.Group):
    with pytest.raises(KeyError):
        access_attributes(local_zarr, "burst_0/missing")


def test_bucket_helpers_cover_listing_info_and_downloads(fake_bucket: tuple[FakeBucketApi, str], tmp_path: Path):
    api, scene_name = fake_bucket

    info = get_hf_bucket_info(api.bucket_id, api=api)
    root_entries = list_hf_bucket_files(api.bucket_id, api=api)
    metadata_entries = list_hf_bucket_files(
        api.bucket_id,
        prefix=scene_name,
        recursive=True,
        suffix="zarr.json",
        api=api,
    )
    zarr_paths = list_hf_bucket_zarrs(api.bucket_id, api=api)
    downloaded_readme = download_hf_bucket_path(
        api.bucket_id,
        "README.md",
        local_dir=tmp_path / "mirror_readme",
        api=api,
    )
    downloaded_echo = download_hf_bucket_path(
        api.bucket_id,
        f"{scene_name}/burst_0/echo",
        local_dir=tmp_path / "mirror_echo",
        api=api,
    )

    assert info.id == api.bucket_id
    assert info.total_files == len(api.file_map)
    assert any(entry.path == scene_name for entry in root_entries)
    assert metadata_entries
    assert all(entry.path.endswith("zarr.json") for entry in metadata_entries)
    assert zarr_paths == [scene_name]
    assert downloaded_readme[0].read_text() == "bucket readme"
    assert any(str(path).endswith("zarr.json") for path in downloaded_echo)
    assert any("/c/0/0" in str(path) for path in downloaded_echo)


def test_sync_and_open_hf_bucket_zarr_metadata_only(fake_bucket: tuple[FakeBucketApi, str], tmp_path: Path):
    api, scene_name = fake_bucket
    local_dir = tmp_path / "mirror"

    local_path = sync_hf_bucket_zarr(
        api.bucket_id,
        scene_name,
        local_dir=local_dir,
        metadata_only=True,
        api=api,
    )
    group = open_hf_bucket_zarr(
        api.bucket_id,
        scene_name,
        local_dir=local_dir,
        metadata_only=True,
        api=api,
    )

    downloaded_files = {
        str(path.relative_to(local_dir))
        for path in local_dir.rglob("*")
        if path.is_file()
    }

    assert local_path == local_dir / scene_name
    assert "burst_0" in group
    assert set(group["burst_0"].keys()) == {"echo", "rfi"}
    assert downloaded_files
    assert all(path.endswith("zarr.json") for path in downloaded_files)


def test_plot_helpers_render_and_save(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    complex_array = np.array(
        [[1 + 1j, 2 + 0j, 3 - 1j], [4 + 2j, 5 + 0j, 6 - 2j]],
        dtype=np.complex64,
    )
    output_path = tmp_path / "magnitude.png"

    plot_complex_array(complex_array, title="Complex Test")
    plot_magnitude(complex_array, title="Magnitude Test", savefig=output_path)

    assert output_path.exists()


def test_trainer_module_is_import_safe_without_optional_dependencies():
    assert trainer._TRAINING_IMPORT_ERROR is not None

    with pytest.raises(ModuleNotFoundError, match="Optional training dependencies"):
        trainer.train_model()
