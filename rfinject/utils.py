# Copyright (c) Roberto Del Prete. All rights reserved.

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import zarr
from typing import Any, Dict, Optional, Tuple


DEFAULT_HF_BUCKET_ID = "ESA-philab/RFInject-v1-L0"
DEFAULT_CACHE_ENV_VAR = "RFINJECT_CACHE_DIR"


def _build_hf_api(api: Any = None, token: Optional[str] = None) -> Any:
    if api is not None:
        return api

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Bucket support requires huggingface-hub>=1.5.0. "
            "Install the project dependencies again."
        ) from exc

    return HfApi(token=token)


def parse_hf_bucket_reference(bucket: str) -> str:
    """Normalize a bucket reference to ``owner/name`` format.

    Supports bucket IDs directly (``owner/name``), prefixed paths
    (``buckets/owner/name`` or ``hf://buckets/owner/name``), and public
    bucket URLs such as
    ``https://huggingface.co/buckets/ESA-philab/RFInject-v1-L0``.
    """
    if not bucket or not bucket.strip():
        raise ValueError("Bucket reference cannot be empty.")

    bucket = bucket.strip()
    if bucket.startswith("hf://"):
        bucket = bucket[5:]

    if bucket.startswith(("https://", "http://")):
        parts = [part for part in urlparse(bucket).path.split("/") if part]
        if len(parts) < 3 or parts[0] != "buckets":
            raise ValueError(
                "Bucket URLs must look like https://huggingface.co/buckets/<owner>/<name>."
            )
        return "/".join(parts[1:3])

    if bucket.startswith("buckets/"):
        bucket = bucket[len("buckets/") :]

    parts = [part for part in bucket.split("/") if part]
    if len(parts) < 2:
        raise ValueError("Bucket references must look like '<owner>/<name>'.")

    return "/".join(parts[:2])


def _normalize_hf_remote_path(remote_path: str) -> str:
    if not remote_path or not remote_path.strip():
        raise ValueError("Remote path cannot be empty.")

    remote_path = remote_path.strip()
    if remote_path.startswith("hf://"):
        remote_path = remote_path[5:]

    if remote_path.startswith(("https://", "http://")):
        parts = [part for part in urlparse(remote_path).path.split("/") if part]
        if len(parts) < 4 or parts[0] != "buckets":
            raise ValueError(
                "Bucket object URLs must include a path after /buckets/<owner>/<name>/."
            )
        return "/".join(parts[3:])

    if remote_path.startswith("buckets/"):
        parts = [part for part in remote_path.split("/") if part]
        if len(parts) < 4:
            raise ValueError(
                "Bucket object references must include a path after buckets/<owner>/<name>/."
            )
        return "/".join(parts[3:])

    return remote_path.lstrip("/")


def _resolve_bucket_local_root(bucket_id: str, local_dir: Optional[os.PathLike | str]) -> Path:
    if local_dir is not None:
        root = Path(local_dir).expanduser()
    else:
        root = Path(
            os.environ.get(
                DEFAULT_CACHE_ENV_VAR,
                str(Path.home() / ".cache" / "rfinject" / "buckets"),
            )
        )
        root = root / bucket_id

    root.mkdir(parents=True, exist_ok=True)
    return root


def _download_bucket_entries(
    bucket_id: str,
    entries: list[Any],
    destination_root: Path,
    *,
    api: Any,
    token: Optional[str] = None,
    raise_on_missing_files: bool = True,
) -> list[Path]:
    local_paths: list[Path] = []
    file_pairs = []

    for entry in entries:
        remote_path = getattr(entry, "path", str(entry))
        local_path = destination_root / remote_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        file_pairs.append((entry, local_path))
        local_paths.append(local_path)

    api.download_bucket_files(
        bucket_id,
        file_pairs,
        raise_on_missing_files=raise_on_missing_files,
        token=token,
    )
    return local_paths


def get_hf_bucket_info(bucket: str = DEFAULT_HF_BUCKET_ID, *, token: Optional[str] = None, api: Any = None) -> Any:
    """Return metadata about a Hugging Face bucket."""
    bucket_id = parse_hf_bucket_reference(bucket)
    api = _build_hf_api(api=api, token=token)
    return api.bucket_info(bucket_id, token=token)


def list_hf_bucket_files(
    bucket: str = DEFAULT_HF_BUCKET_ID,
    *,
    prefix: Optional[str] = None,
    recursive: bool = False,
    suffix: Optional[str] = None,
    limit: Optional[int] = None,
    token: Optional[str] = None,
    api: Any = None,
) -> list[Any]:
    """List bucket entries, optionally filtering by prefix/suffix."""
    bucket_id = parse_hf_bucket_reference(bucket)
    api = _build_hf_api(api=api, token=token)
    normalized_prefix = _normalize_hf_remote_path(prefix) if prefix else None

    entries = []
    for entry in api.list_bucket_tree(
        bucket_id,
        prefix=normalized_prefix,
        recursive=recursive,
        token=token,
    ):
        if suffix is not None and not entry.path.endswith(suffix):
            continue

        entries.append(entry)
        if limit is not None and len(entries) >= limit:
            break

    return entries


def list_hf_bucket_zarrs(
    bucket: str = DEFAULT_HF_BUCKET_ID,
    *,
    limit: Optional[int] = None,
    token: Optional[str] = None,
    api: Any = None,
) -> list[str]:
    """List top-level Zarr products available in a bucket."""
    entries = list_hf_bucket_files(
        bucket,
        recursive=False,
        token=token,
        api=api,
    )

    zarr_paths = [entry.path.rstrip("/") for entry in entries if entry.path.endswith(".zarr")]
    if limit is not None:
        return zarr_paths[:limit]
    return zarr_paths


def download_hf_bucket_path(
    bucket: str = DEFAULT_HF_BUCKET_ID,
    remote_path: str = "",
    local_dir: Optional[os.PathLike | str] = None,
    *,
    recursive: bool = True,
    token: Optional[str] = None,
    api: Any = None,
) -> list[Path]:
    """Download a file or a directory-like prefix from a bucket.

    Files are mirrored under ``local_dir`` preserving their relative paths in the
    bucket. If ``local_dir`` is omitted, a cache directory under
    ``~/.cache/rfinject/buckets/<owner>/<bucket>`` is used.
    """
    bucket_id = parse_hf_bucket_reference(bucket)
    api = _build_hf_api(api=api, token=token)
    remote_path = _normalize_hf_remote_path(remote_path)
    destination_root = _resolve_bucket_local_root(bucket_id, local_dir)

    exact_file_matches = list(api.get_bucket_paths_info(bucket_id, [remote_path], token=token))
    if exact_file_matches:
        return _download_bucket_entries(
            bucket_id,
            exact_file_matches,
            destination_root,
            api=api,
            token=token,
        )

    prefix = remote_path.rstrip("/") + "/"
    entries = list_hf_bucket_files(
        bucket_id,
        prefix=prefix,
        recursive=recursive,
        token=token,
        api=api,
    )
    if not entries:
        raise FileNotFoundError(
            f"'{remote_path}' was not found in bucket '{bucket_id}'."
        )

    return _download_bucket_entries(
        bucket_id,
        entries,
        destination_root,
        api=api,
        token=token,
    )


def sync_hf_bucket_zarr(
    bucket: str = DEFAULT_HF_BUCKET_ID,
    zarr_path: str = "",
    local_dir: Optional[os.PathLike | str] = None,
    *,
    metadata_only: bool = False,
    token: Optional[str] = None,
    api: Any = None,
) -> Path:
    """Mirror a bucket-hosted Zarr store locally and return its local path.

    When ``metadata_only=True``, only ``zarr.json`` files are downloaded. This is
    enough to inspect hierarchy, attributes, shapes, and chunk metadata for the
    RFInject Zarr v3 products hosted in the bucket.
    """
    bucket_id = parse_hf_bucket_reference(bucket)
    api = _build_hf_api(api=api, token=token)
    normalized_zarr_path = _normalize_hf_remote_path(zarr_path).rstrip("/")
    prefix = normalized_zarr_path + "/"

    entries = list_hf_bucket_files(
        bucket_id,
        prefix=prefix,
        recursive=True,
        token=token,
        api=api,
    )
    if metadata_only:
        entries = [entry for entry in entries if entry.path.endswith("zarr.json")]

    if not entries:
        raise FileNotFoundError(
            f"Zarr path '{normalized_zarr_path}' was not found in bucket '{bucket_id}'."
        )

    destination_root = _resolve_bucket_local_root(bucket_id, local_dir)
    _download_bucket_entries(
        bucket_id,
        entries,
        destination_root,
        api=api,
        token=token,
    )
    return destination_root / normalized_zarr_path


def open_hf_bucket_zarr(
    bucket: str = DEFAULT_HF_BUCKET_ID,
    zarr_path: str = "",
    local_dir: Optional[os.PathLike | str] = None,
    *,
    metadata_only: bool = True,
    token: Optional[str] = None,
    api: Any = None,
    mode: str = "r",
) -> zarr.Group:
    """Open a bucket-hosted Zarr store through a local mirror.

    The bucket is treated as the source of truth while the local filesystem is a
    cache/mirror used by Zarr. This avoids relying on repository-style
    ``hf://`` filesystem semantics, which do not apply to buckets today.
    """
    local_path = sync_hf_bucket_zarr(
        bucket=bucket,
        zarr_path=zarr_path,
        local_dir=local_dir,
        metadata_only=metadata_only,
        token=token,
        api=api,
    )
    return zarr.open_group(str(local_path), mode=mode)



def explore_zarr_structure(zarr_group: zarr.Group, max_depth: int = 3) -> None:
    """Explore and print the structure of a Zarr group recursively.
    
    Args:
        zarr_group (zarr.Group): The Zarr group to explore.
        max_depth (int): Maximum depth to explore. Defaults to 3.
    """
    assert isinstance(zarr_group, zarr.Group), 'Input must be a Zarr group'
    
    def _print_structure(group: zarr.Group, indent: str = '', depth: int = 0) -> None:
        if depth > max_depth:
            return
            
        for key in group.keys():
            item = group[key]
            if isinstance(item, zarr.Group):
                print(f'{indent}📁 {key}/')
                _print_structure(item, indent + '  ', depth + 1)
            else:
                shape_str = f'{item.shape}' if hasattr(item, 'shape') else 'unknown'
                dtype_str = f'{item.dtype}' if hasattr(item, 'dtype') else 'unknown'
                print(f'{indent}📄 {key}: {shape_str} {dtype_str}')
                
                # Show attributes if any
                if hasattr(item, 'attrs') and item.attrs:
                    for attr_key, attr_val in item.attrs.items():
                        print(f'{indent}    📋 {attr_key}: {attr_val}')
    
    print('Zarr Structure:')
    _print_structure(zarr_group)


def access_array_data(zarr_group: zarr.Group, burst_name: str, array_name: str) -> zarr.Array:
    """Access a specific array from a burst.
    
    Args:
        zarr_group (zarr.Group): The Zarr group containing the data.
        burst_name (str): Name of the burst (e.g., 'burst_0').
        array_name (str): Name of the array (e.g., 'echo', 'rfi', 'echo_w_rfi').
        
    Returns:
        zarr.Array: The requested array.
    """
    assert burst_name in zarr_group.keys(), f'Burst {burst_name} not found'
    assert array_name in zarr_group[burst_name].keys(), f'Array {array_name} not found in {burst_name}'
    
    return zarr_group[burst_name][array_name]


def get_array_slice(array: zarr.Array, slice_params: Optional[Tuple] = None) -> np.ndarray:
    """Get a slice of data from a Zarr array.
    
    Args:
        array (zarr.Array): The Zarr array to slice.
        slice_params (Optional[Tuple]): Slice parameters. If None, returns first 10x10 slice.
        
    Returns:
        np.ndarray: The sliced data as a NumPy array.
    """
    if slice_params is None:
        # Default: get a small slice for inspection
        if len(array.shape) == 2:
            slice_params = (slice(0, 10), slice(0, 10))
        elif len(array.shape) == 3:
            slice_params = (0, slice(0, 10), slice(0, 10))
        else:
            slice_params = tuple(slice(0, 10) for _ in array.shape)
    
    return array[slice_params]


def get_burst_info(zarr_group: zarr.Group) -> Dict[str, Dict[str, Any]]:
    """Extract information about all bursts in the Zarr group.
    
    Args:
        zarr_group (zarr.Group): The Zarr group containing burst data.
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with burst information.
    """
    assert isinstance(zarr_group, zarr.Group), 'Input must be a Zarr group'
    
    burst_info = {}
    
    for key in zarr_group.keys():
        if key.startswith('burst_'):
            burst = zarr_group[key]
            
            # Get basic info about each array in the burst
            arrays_info = {}
            for array_name in burst.keys():
                array = burst[array_name]
                arrays_info[array_name] = {
                    'shape': array.shape,
                    'dtype': array.dtype,
                    'size_mb': array.nbytes / (1024 * 1024),
                    'chunks': array.chunks if hasattr(array, 'chunks') else None
                }
            
            burst_info[key] = {
                'arrays': arrays_info,
                'total_size_mb': sum(info['size_mb'] for info in arrays_info.values())
            }
    
    return burst_info


def access_attributes(zarr_item: zarr.Group, path: Optional[str] = None) -> Dict[str, Any]:
    """Access attributes from a Zarr array or group, optionally at a specific path.
    
    Args:
        zarr_item (zarr.Group): The Zarr group or array to explore.
        path (Optional[str]): Optional path to navigate to (e.g., 'burst_0', 'burst_0/echo').
                             If None, returns attributes of the root item.
        
    Returns:
        Dict[str, Any]: Dictionary of attributes.
        
    Raises:
        KeyError: If the specified path does not exist.
    """
    target_item = zarr_item
    
    if path is not None:
        # Navigate to the specified path
        path_parts = path.split('/')
        for part in path_parts:
            if part in target_item:
                target_item = target_item[part]
            else:
                raise KeyError(f'Path "{path}" not found. Part "{part}" does not exist.')
    
    if hasattr(target_item, 'attrs'):
        return dict(target_item.attrs)
    return {}


def explore_all_attributes(zarr_group: zarr.Group) -> Dict[str, Dict[str, Any]]:
    """Explore all attributes in the Zarr group hierarchy.
    
    Args:
        zarr_group (zarr.Group): The Zarr group to explore.
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of all attributes found.
    """
    all_attrs = {}
    
    # Root level attributes
    root_attrs = access_attributes(zarr_group)
    if root_attrs:
        all_attrs['root'] = root_attrs
    
    # Burst level attributes
    for burst_name in zarr_group.keys():
        if burst_name.startswith('burst_'):
            burst = zarr_group[burst_name]
            burst_attrs = access_attributes(burst)
            if burst_attrs:
                all_attrs[burst_name] = burst_attrs
            
            # Array level attributes
            for array_name in burst.keys():
                array = burst[array_name]
                array_attrs = access_attributes(array)
                if array_attrs:
                    all_attrs[f'{burst_name}/{array_name}'] = array_attrs
    
    return all_attrs


__all__ = [
    "DEFAULT_HF_BUCKET_ID",
    "access_array_data",
    "access_attributes",
    "download_hf_bucket_path",
    "explore_all_attributes",
    "explore_zarr_structure",
    "get_array_slice",
    "get_burst_info",
    "get_hf_bucket_info",
    "list_hf_bucket_files",
    "list_hf_bucket_zarrs",
    "open_hf_bucket_zarr",
    "parse_hf_bucket_reference",
    "sync_hf_bucket_zarr",
]

