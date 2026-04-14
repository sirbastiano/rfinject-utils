"""Deterministic geographic partitioning for catalog parquet files."""

from __future__ import annotations

import argparse
from ast import literal_eval
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SPLIT_NAMES = ("train", "validation", "test")
CDSE_ODATA_BASE_URL = "https://download.dataspace.copernicus.eu/odata/v1"
ONLINE_CHILD_NODE_RE = re.compile(
    r"^(?P<stem>s1[abc]-iw-raw-s-(?:hh|hv|vh|vv)-\d{8}t\d{6}-\d{8}t\d{6}-\d{6}-[0-9a-f]{6})"
    r"(?P<suffix>-(?:annot|index))?\.dat$"
)


class CatalogSplitError(ValueError):
    """Raised when a catalog cannot satisfy the geographic split gates."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic geographic partitioning."""

    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    lat_sections: int = 4
    lon_sections: int = 4
    seed: int = 42
    min_rows: int = 3
    min_unique_sections: int = 2

    @property
    def ratios(self) -> tuple[float, float, float]:
        return (self.train_ratio, self.validation_ratio, self.test_ratio)

    def validate(self) -> None:
        if self.lat_sections < 1 or self.lon_sections < 1:
            raise ValueError("lat_sections and lon_sections must be positive integers.")
        if self.min_rows < len(SPLIT_NAMES):
            raise ValueError("min_rows must be at least the number of splits.")
        if self.min_unique_sections < 1:
            raise ValueError("min_unique_sections must be at least 1.")
        ratios = self.ratios
        if any(r <= 0 for r in ratios):
            raise ValueError("train/validation/test ratios must all be strictly positive.")
        if not math.isclose(sum(ratios), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("train/validation/test ratios must sum to 1.0.")


@dataclass
class CatalogInspection:
    """In-memory representation of a catalog plus geographic diagnostics."""

    input_path: Path
    table: pa.Table
    frame: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass
class LinkedCatalogFrames:
    """Parent/child catalog frames produced from the RFInject CSV plus bucket children."""

    parents: pd.DataFrame
    linked_parents: pd.DataFrame
    unmatched_parents: pd.DataFrame
    children: pd.DataFrame
    diagnostics: dict[str, Any]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _bbox_to_record(index: int, item_id: Any, bbox: Any) -> dict[str, Any]:
    bbox = bbox if isinstance(bbox, dict) else {}
    xmin = _coerce_float(bbox.get("xmin"))
    xmax = _coerce_float(bbox.get("xmax"))
    ymin = _coerce_float(bbox.get("ymin"))
    ymax = _coerce_float(bbox.get("ymax"))
    bbox_valid = None not in (xmin, xmax, ymin, ymax)
    centroid_lon = (xmin + xmax) / 2.0 if bbox_valid else None
    centroid_lat = (ymin + ymax) / 2.0 if bbox_valid else None
    bbox_all_zero = bbox_valid and xmin == xmax == ymin == ymax == 0.0
    return {
        "row_index": index,
        "item_id": item_id,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "bbox_valid": bbox_valid,
        "bbox_all_zero": bbox_all_zero,
    }


def inspect_catalog(input_path: str | Path) -> CatalogInspection:
    """Load a parquet catalog and extract geography diagnostics from `bbox`."""

    path = Path(input_path)
    table = pq.read_table(path)
    if "bbox" not in table.column_names:
        raise CatalogSplitError(
            "Catalog parquet does not contain a `bbox` column required for geographic partitioning.",
            diagnostics={"input_path": str(path.resolve()), "rows": table.num_rows, "columns": table.column_names},
        )

    item_ids = table["id"].to_pylist() if "id" in table.column_names else [None] * table.num_rows
    records = [
        _bbox_to_record(index=i, item_id=item_id, bbox=bbox)
        for i, (item_id, bbox) in enumerate(zip(item_ids, table["bbox"].to_pylist(), strict=True))
    ]
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "row_index",
                "item_id",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "centroid_lon",
                "centroid_lat",
                "bbox_valid",
                "bbox_all_zero",
            ]
        )

    valid_frame = frame.loc[frame["bbox_valid"]].copy()
    unique_centroids = (
        valid_frame.loc[:, ["centroid_lon", "centroid_lat"]].drop_duplicates().shape[0] if not valid_frame.empty else 0
    )
    diagnostics = {
        "input_path": str(path.resolve()),
        "rows": int(table.num_rows),
        "columns": table.column_names,
        "bbox_valid_rows": int(frame["bbox_valid"].sum()) if not frame.empty else 0,
        "bbox_all_zero_rows": int(frame["bbox_all_zero"].sum()) if not frame.empty else 0,
        "unique_centroids": int(unique_centroids),
        "sample_item_ids": frame["item_id"].head(5).tolist(),
    }
    return CatalogInspection(input_path=path, table=table, frame=frame, diagnostics=diagnostics)


def normalize_parent_product_key(name: str) -> str:
    """Normalize a SAFE product name into the bucket child match key."""

    normalized = Path(name).name
    if normalized.endswith(".SAFE"):
        normalized = normalized[: -len(".SAFE")]
    tokens = normalized.lower().replace("_", "-").split("-")
    if len(tokens) < 5:
        raise CatalogSplitError(f"Cannot normalize parent product name '{name}'.")
    return "-".join(tokens[-5:-1])


def normalize_child_product_key(child_product: str) -> str:
    """Normalize a bucket child product path into the parent match key."""

    normalized = Path(child_product).name
    if normalized.endswith(".zarr"):
        normalized = normalized[: -len(".zarr")]
    tokens = normalized.lower().split("-")
    if len(tokens) < 4:
        raise CatalogSplitError(f"Cannot normalize child product path '{child_product}'.")
    return "-".join(tokens[-4:])


def _collect_coordinate_pairs(node: Any, pairs: list[tuple[float, float]]) -> None:
    if isinstance(node, (list, tuple)):
        if len(node) == 2 and all(isinstance(value, (int, float)) for value in node):
            pairs.append((float(node[0]), float(node[1])))
            return
        for item in node:
            _collect_coordinate_pairs(item, pairs)


def parse_geofootprint_bounds(geofootprint: str | dict[str, Any]) -> dict[str, float]:
    """Extract a bounding box and centroid from a GeoFootprint payload."""

    payload = literal_eval(geofootprint) if isinstance(geofootprint, str) else geofootprint
    if not isinstance(payload, dict) or "coordinates" not in payload:
        raise CatalogSplitError("GeoFootprint payload is missing coordinates.")

    pairs: list[tuple[float, float]] = []
    _collect_coordinate_pairs(payload["coordinates"], pairs)
    if not pairs:
        raise CatalogSplitError("GeoFootprint payload does not contain any coordinate pairs.")

    longitudes = [lon for lon, _ in pairs]
    latitudes = [lat for _, lat in pairs]
    xmin, xmax = min(longitudes), max(longitudes)
    ymin, ymax = min(latitudes), max(latitudes)
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "centroid_lon": (xmin + xmax) / 2.0,
        "centroid_lat": (ymin + ymax) / 2.0,
    }


def _frame_with_bounds(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    bounds = working["GeoFootprint"].map(parse_geofootprint_bounds).apply(pd.Series)
    for column in ["xmin", "xmax", "ymin", "ymax", "centroid_lon", "centroid_lat"]:
        working[column] = bounds[column].astype(float)
    working["bbox_valid"] = True
    working["bbox_all_zero"] = False
    return working


def load_rfinject_csv_catalog(input_path: str | Path) -> pd.DataFrame:
    """Load the RFInject parent-product CSV catalog."""

    frame = pd.read_csv(input_path)
    required_columns = {"Id", "Name", "GeoFootprint"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise CatalogSplitError(
            "RFInject CSV is missing required columns.",
            diagnostics={"input_path": str(Path(input_path).resolve()), "missing_columns": missing},
        )
    return frame


def link_rfinject_catalog_to_children(
    frame: pd.DataFrame,
    child_products: Iterable[str],
    *,
    input_path: str | Path | None = None,
) -> LinkedCatalogFrames:
    """Link RFInject parent products to bucket child `.zarr` products."""

    bucket_children_by_key: dict[str, list[str]] = defaultdict(list)
    for child_product in child_products:
        bucket_children_by_key[normalize_child_product_key(child_product)].append(child_product)

    parents = _frame_with_bounds(frame)
    parents["match_key"] = parents["Name"].map(normalize_parent_product_key)
    parents["child_products"] = parents["match_key"].map(
        lambda key: sorted(bucket_children_by_key.get(key, []))
    )
    parents["child_count"] = parents["child_products"].map(len).astype(int)

    linked_parents = parents.loc[parents["child_count"] > 0].copy()
    unmatched_parents = parents.loc[parents["child_count"] == 0].copy()

    children = linked_parents.loc[
        :,
        [
            "Id",
            "Name",
            "match_key",
            "centroid_lon",
            "centroid_lat",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "child_products",
        ],
    ].explode("child_products", ignore_index=True)
    children = children.rename(
        columns={
            "Id": "parent_id",
            "Name": "parent_name",
            "child_products": "child_product",
        }
    )
    if not children.empty:
        children["child_key"] = children["child_product"].map(normalize_child_product_key)
        children["child_name"] = children["child_product"].map(lambda value: Path(value).name)
    else:
        children["child_key"] = pd.Series(dtype="string")
        children["child_name"] = pd.Series(dtype="string")

    diagnostics = {
        "input_path": str(Path(input_path).resolve()) if input_path is not None else None,
        "rows": int(len(parents)),
        "linked_parents": int(len(linked_parents)),
        "unmatched_parents": int(len(unmatched_parents)),
        "linked_children": int(len(children)),
        "unique_centroids": int(
            linked_parents.loc[:, ["centroid_lon", "centroid_lat"]].drop_duplicates().shape[0]
        )
        if not linked_parents.empty
        else 0,
        "bbox_valid_rows": int(linked_parents["bbox_valid"].sum()) if not linked_parents.empty else 0,
        "bbox_all_zero_rows": int(linked_parents["bbox_all_zero"].sum()) if not linked_parents.empty else 0,
        "sample_linked_names": linked_parents["Name"].head(5).tolist(),
        "sample_unmatched_names": unmatched_parents["Name"].head(5).tolist(),
    }
    return LinkedCatalogFrames(
        parents=parents,
        linked_parents=linked_parents,
        unmatched_parents=unmatched_parents,
        children=children,
        diagnostics=diagnostics,
    )


def fetch_cdse_product_nodes(
    product_id: str,
    product_name: str,
    *,
    base_url: str = CDSE_ODATA_BASE_URL,
    timeout_seconds: int = 30,
) -> list[dict[str, Any]]:
    """Fetch the direct child nodes for a CDSE product SAFE root via OData."""

    encoded_name = quote(product_name, safe="._-")
    url = f"{base_url}/Products({product_id})/Nodes({encoded_name})/Nodes"
    request = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:  # pragma: no cover - exercised in live runs
        raise CatalogSplitError(
            f"CDSE node listing failed for product '{product_name}' with HTTP {exc.code}.",
            diagnostics={"product_id": product_id, "product_name": product_name, "status_code": exc.code},
        ) from exc
    except URLError as exc:  # pragma: no cover - exercised in live runs
        raise CatalogSplitError(
            f"CDSE node listing failed for product '{product_name}': {exc.reason}.",
            diagnostics={"product_id": product_id, "product_name": product_name},
        ) from exc

    result = payload.get("result")
    if not isinstance(result, list):
        raise CatalogSplitError(
            f"CDSE node listing returned an unexpected payload for product '{product_name}'.",
            diagnostics={"product_id": product_id, "product_name": product_name},
        )
    return result


def extract_online_child_records(node_entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse CDSE node entries into child-product records."""

    grouped: dict[str, dict[str, Any]] = {}

    for entry in node_entries:
        node_name = entry.get("Name")
        if not isinstance(node_name, str):
            continue
        match = ONLINE_CHILD_NODE_RE.match(node_name)
        if match is None:
            continue

        child_stem = match.group("stem")
        suffix = match.group("suffix")
        variant = "data" if suffix is None else suffix.lstrip("-")

        record = grouped.setdefault(
            child_stem,
            {
                "child_product": child_stem,
                "child_name": child_stem,
                "has_data": False,
                "has_annot": False,
                "has_index": False,
                "node_names": [],
            },
        )
        record[f"has_{variant}"] = True
        record["node_names"].append(node_name)

    child_records = []
    for child_stem, record in grouped.items():
        node_names = sorted(record["node_names"])
        child_records.append(
            {
                "child_product": child_stem,
                "child_name": record["child_name"],
                "has_data": bool(record["has_data"]),
                "has_annot": bool(record["has_annot"]),
                "has_index": bool(record["has_index"]),
                "node_names": node_names,
                "node_names_json": json.dumps(node_names),
            }
        )
    return sorted(child_records, key=lambda item: item["child_product"])


def link_rfinject_catalog_to_cdse_nodes(
    frame: pd.DataFrame,
    *,
    input_path: str | Path | None = None,
    max_workers: int = 8,
    fetcher: Any = None,
) -> LinkedCatalogFrames:
    """Link RFInject parent products to their live CDSE child nodes."""

    fetcher = fetcher or fetch_cdse_product_nodes
    parents = _frame_with_bounds(frame)
    parents["match_key"] = parents["Name"].map(normalize_parent_product_key)

    node_payloads: dict[int, list[dict[str, Any]]] = {}
    query_errors: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(fetcher, row["Id"], row["Name"]): index
            for index, row in parents.loc[:, ["Id", "Name"]].iterrows()
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            row = parents.loc[index]
            try:
                node_payloads[index] = future.result()
            except CatalogSplitError as exc:
                query_errors.append(
                    {
                        "row_index": int(index),
                        "product_id": row["Id"],
                        "product_name": row["Name"],
                        "error": str(exc),
                    }
                )
                node_payloads[index] = []

    parents["child_records"] = parents.index.map(
        lambda index: extract_online_child_records(node_payloads.get(index, []))
    )
    parents["child_products"] = parents["child_records"].map(
        lambda records: [record["child_product"] for record in records]
    )
    parents["child_count"] = parents["child_products"].map(len).astype(int)
    parents["online_node_count"] = parents.index.map(lambda index: len(node_payloads.get(index, []))).astype(int)

    linked_parents = parents.loc[parents["child_count"] > 0].copy()
    unmatched_parents = parents.loc[parents["child_count"] == 0].copy()

    child_rows: list[dict[str, Any]] = []
    for _, row in linked_parents.iterrows():
        for child_record in row["child_records"]:
            child_rows.append(
                {
                    "parent_id": row["Id"],
                    "parent_name": row["Name"],
                    "match_key": normalize_parent_product_key(row["Name"]),
                    "centroid_lon": row["centroid_lon"],
                    "centroid_lat": row["centroid_lat"],
                    "xmin": row["xmin"],
                    "xmax": row["xmax"],
                    "ymin": row["ymin"],
                    "ymax": row["ymax"],
                    **child_record,
                }
            )
    children = pd.DataFrame.from_records(child_rows)
    if children.empty:
        children = pd.DataFrame(
            columns=[
                "parent_id",
                "parent_name",
                "match_key",
                "centroid_lon",
                "centroid_lat",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "child_product",
                "child_name",
                "has_data",
                "has_annot",
                "has_index",
                "node_names",
                "node_names_json",
            ]
        )

    diagnostics = {
        "input_path": str(Path(input_path).resolve()) if input_path is not None else None,
        "rows": int(len(parents)),
        "linked_parents": int(len(linked_parents)),
        "unmatched_parents": int(len(unmatched_parents)),
        "linked_children": int(len(children)),
        "unique_centroids": int(
            linked_parents.loc[:, ["centroid_lon", "centroid_lat"]].drop_duplicates().shape[0]
        )
        if not linked_parents.empty
        else 0,
        "bbox_valid_rows": int(linked_parents["bbox_valid"].sum()) if not linked_parents.empty else 0,
        "bbox_all_zero_rows": int(linked_parents["bbox_all_zero"].sum()) if not linked_parents.empty else 0,
        "query_errors": query_errors,
        "sample_linked_names": linked_parents["Name"].head(5).tolist(),
        "sample_unmatched_names": unmatched_parents["Name"].head(5).tolist(),
    }
    return LinkedCatalogFrames(
        parents=parents,
        linked_parents=linked_parents,
        unmatched_parents=unmatched_parents,
        children=children,
        diagnostics=diagnostics,
    )


def _quantile_section_codes(series: pd.Series, max_bins: int) -> pd.Series:
    valid = series.dropna()
    codes = pd.Series(pd.NA, index=series.index, dtype="Int64")
    unique_values = valid.nunique()
    if unique_values == 0:
        return codes
    if unique_values == 1:
        codes.loc[valid.index] = 0
        return codes

    bin_count = min(max_bins, int(unique_values))
    qcut = pd.qcut(valid, q=bin_count, labels=False, duplicates="drop")
    codes.loc[valid.index] = pd.Series(qcut, index=valid.index, dtype="Int64")
    return codes


def _assign_geo_sections(frame: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    working = frame.copy()
    working["lat_section"] = _quantile_section_codes(working["centroid_lat"], config.lat_sections)
    working["lon_section"] = _quantile_section_codes(working["centroid_lon"], config.lon_sections)
    working["geo_section"] = (
        "lat"
        + working["lat_section"].astype("string")
        + "_lon"
        + working["lon_section"].astype("string")
    )
    return working


def _validate_split_inputs(frame: pd.DataFrame, diagnostics: dict[str, Any], config: SplitConfig) -> None:
    errors: list[str] = []
    row_count = int(len(frame))
    unique_sections = int(frame["geo_section"].nunique(dropna=True)) if "geo_section" in frame else 0
    diagnostics["unique_sections"] = unique_sections

    if row_count < config.min_rows:
        errors.append(
            f"Need at least {config.min_rows} rows to populate train/validation/test; found {row_count}."
        )
    if diagnostics.get("bbox_valid_rows", 0) != row_count:
        errors.append(
            "Every row needs a finite bounding box for geographic sectioning."
        )
    if diagnostics.get("bbox_all_zero_rows", 0) == row_count and row_count > 0:
        errors.append("All bounding boxes are zero-valued, so the catalog does not expose usable geography.")
    if diagnostics.get("unique_centroids", 0) < config.min_unique_sections:
        errors.append(
            f"Need at least {config.min_unique_sections} distinct geographic centroids; "
            f"found {diagnostics.get('unique_centroids', 0)}."
        )
    if unique_sections < config.min_unique_sections:
        errors.append(
            f"Need at least {config.min_unique_sections} geographic sections after binning; found {unique_sections}."
        )

    if errors:
        diagnostics["validation_errors"] = errors
        raise CatalogSplitError(" ".join(errors), diagnostics=diagnostics)


def _allocate_group_counts(group_size: int, ratios: tuple[float, float, float]) -> list[int]:
    raw_counts = [group_size * ratio for ratio in ratios]
    allocated = [math.floor(raw) for raw in raw_counts]
    remainder = group_size - sum(allocated)
    fractional_order = sorted(
        range(len(raw_counts)),
        key=lambda idx: (raw_counts[idx] - allocated[idx], raw_counts[idx]),
        reverse=True,
    )
    for index in fractional_order[:remainder]:
        allocated[index] += 1
    return allocated


def _ensure_non_empty_splits(frame: pd.DataFrame, diagnostics: dict[str, Any]) -> pd.DataFrame:
    working = frame.copy()
    counts = working["split"].value_counts().to_dict()
    missing = [split_name for split_name in SPLIT_NAMES if counts.get(split_name, 0) == 0]
    if not missing:
        return working

    for missing_split in missing:
        counts = working["split"].value_counts().to_dict()
        donors = [split_name for split_name in SPLIT_NAMES if counts.get(split_name, 0) > 1]
        if not donors:
            diagnostics["split_counts"] = counts
            raise CatalogSplitError(
                "Not enough rows to keep all train/validation/test splits non-empty after stratification.",
                diagnostics=diagnostics,
            )
        donor = max(donors, key=lambda split_name: counts[split_name])
        donor_rows = working.loc[working["split"] == donor].copy()
        donor_rows["_row_order"] = donor_rows["row_index"] if "row_index" in donor_rows.columns else donor_rows.index.to_series()
        donor_rows = donor_rows.sort_values(["geo_section", "_row_order"])
        working.at[donor_rows.index[-1], "split"] = missing_split
    return working


def _target_split_counts(total_rows: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    return dict(zip(SPLIT_NAMES, _allocate_group_counts(total_rows, ratios), strict=True))


def _rebalance_split_totals(frame: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Adjust split totals to the global target counts while preserving section stratification as much as possible."""

    working = frame.copy()
    target_counts = _target_split_counts(len(working), config.ratios)

    while True:
        counts = {split_name: int((working["split"] == split_name).sum()) for split_name in SPLIT_NAMES}
        donors = [split_name for split_name in SPLIT_NAMES if counts[split_name] > target_counts[split_name]]
        recipients = [
            split_name for split_name in SPLIT_NAMES if counts[split_name] < target_counts[split_name]
        ]
        if not donors or not recipients:
            break

        donor = max(donors, key=lambda split_name: counts[split_name] - target_counts[split_name])
        recipient = max(
            recipients,
            key=lambda split_name: target_counts[split_name] - counts[split_name],
        )

        donor_rows = working.loc[working["split"] == donor].copy()
        donor_rows["_row_order"] = donor_rows.index.to_series()
        donor_rows["donor_section_count"] = donor_rows["geo_section"].map(
            lambda section: int(
                ((working["geo_section"] == section) & (working["split"] == donor)).sum()
            )
        )
        donor_rows["recipient_section_count"] = donor_rows["geo_section"].map(
            lambda section: int(
                ((working["geo_section"] == section) & (working["split"] == recipient)).sum()
            )
        )
        donor_rows = donor_rows.sort_values(
            ["donor_section_count", "recipient_section_count", "_row_order"],
            ascending=[False, True, False],
        )
        candidate_index = donor_rows.index[0]
        working.at[candidate_index, "split"] = recipient

    return working


def assign_splits(frame: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Assign deterministic train/validation/test labels within each geographic section."""

    working = frame.copy()
    working["split"] = pd.Series(index=working.index, dtype="string")

    for section_offset, section_name in enumerate(sorted(working["geo_section"].dropna().unique())):
        group = working.loc[working["geo_section"] == section_name].copy()
        rng = np.random.default_rng(config.seed + section_offset)
        ordered_index = group.index.to_numpy()
        rng.shuffle(ordered_index)

        counts = _allocate_group_counts(len(group), config.ratios)
        labels: list[str] = []
        for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
            labels.extend([split_name] * count)
        working.loc[ordered_index, "split"] = labels

    return working


def _augment_table(table: pa.Table, frame: pd.DataFrame) -> pa.Table:
    enriched = table
    for name, data, value_type in (
        ("centroid_lon", frame["centroid_lon"].tolist(), pa.float64()),
        ("centroid_lat", frame["centroid_lat"].tolist(), pa.float64()),
        ("geo_section", frame["geo_section"].tolist(), pa.string()),
        ("split", frame["split"].tolist(), pa.string()),
    ):
        enriched = enriched.append_column(name, pa.array(data, type=value_type))
    return enriched


def _build_summary(frame: pd.DataFrame, diagnostics: dict[str, Any], config: SplitConfig) -> dict[str, Any]:
    split_counts = {split_name: int((frame["split"] == split_name).sum()) for split_name in SPLIT_NAMES}
    section_counts = frame["geo_section"].value_counts().sort_index().astype(int).to_dict()
    section_split_counts = (
        frame.groupby(["geo_section", "split"]).size().unstack(fill_value=0).sort_index().astype(int).to_dict(orient="index")
    )
    return {
        "status": "success",
        "input_path": diagnostics["input_path"],
        "rows": diagnostics["rows"],
        "unique_centroids": diagnostics["unique_centroids"],
        "unique_sections": diagnostics["unique_sections"],
        "split_counts": split_counts,
        "section_counts": section_counts,
        "section_split_counts": section_split_counts,
        "ratios": {
            "train": config.train_ratio,
            "validation": config.validation_ratio,
            "test": config.test_ratio,
        },
        "seed": config.seed,
    }


def _write_dataframe_outputs(frame: pd.DataFrame, target_prefix: Path) -> None:
    serializable = frame.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].astype("string")
    serializable.to_csv(target_prefix.with_suffix(".csv"), index=False)
    serializable.to_parquet(target_prefix.with_suffix(".parquet"), index=False)


def _build_rfinject_summary(
    linked_parents: pd.DataFrame,
    linked_children: pd.DataFrame,
    unmatched_parents: pd.DataFrame,
    diagnostics: dict[str, Any],
    config: SplitConfig,
    bucket: str,
    child_source: str,
) -> dict[str, Any]:
    parent_split_counts = {
        split_name: int((linked_parents["split"] == split_name).sum()) for split_name in SPLIT_NAMES
    }
    child_split_counts = {
        split_name: int((linked_children["split"] == split_name).sum()) for split_name in SPLIT_NAMES
    }
    section_counts = linked_parents["geo_section"].value_counts().sort_index().astype(int).to_dict()
    section_split_counts = (
        linked_parents.groupby(["geo_section", "split"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
        .astype(int)
        .to_dict(orient="index")
    )
    return {
        "status": "success",
        "input_path": diagnostics["input_path"],
        "bucket": bucket,
        "child_source": child_source,
        "total_parents": diagnostics["rows"],
        "linked_parents": diagnostics["linked_parents"],
        "unmatched_parents": diagnostics["unmatched_parents"],
        "linked_children": diagnostics["linked_children"],
        "unique_centroids": diagnostics["unique_centroids"],
        "unique_sections": diagnostics["unique_sections"],
        "query_errors": len(diagnostics.get("query_errors", [])),
        "parent_split_counts": parent_split_counts,
        "child_split_counts": child_split_counts,
        "section_counts": section_counts,
        "section_split_counts": section_split_counts,
        "ratios": {
            "train": config.train_ratio,
            "validation": config.validation_ratio,
            "test": config.test_ratio,
        },
        "lat_sections": config.lat_sections,
        "lon_sections": config.lon_sections,
        "seed": config.seed,
        "match_rate": diagnostics["linked_parents"] / diagnostics["rows"] if diagnostics["rows"] else 0.0,
    }


def write_failure_report(output_dir: str | Path, diagnostics: dict[str, Any], error_message: str) -> Path:
    """Persist a failure report when the requested split cannot be produced."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "failed",
        "generated_at": datetime.now().astimezone().isoformat(),
        "error": error_message,
        **diagnostics,
    }
    report_path = target_dir / "split_failure.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path


def split_catalog_geographically(
    input_path: str | Path,
    output_dir: str | Path,
    config: SplitConfig | None = None,
) -> dict[str, Any]:
    """Create train/validation/test parquet partitions from a geographic catalog."""

    config = config or SplitConfig()
    config.validate()

    inspection = inspect_catalog(input_path)
    sectioned = _assign_geo_sections(inspection.frame, config)
    diagnostics = dict(inspection.diagnostics)
    _validate_split_inputs(sectioned, diagnostics, config)

    assigned = assign_splits(sectioned, config)
    assigned = _ensure_non_empty_splits(assigned, diagnostics)
    assigned = _rebalance_split_totals(assigned, config)

    if assigned["split"].isna().any():
        diagnostics["unassigned_rows"] = int(assigned["split"].isna().sum())
        raise CatalogSplitError("Split assignment left some rows unassigned.", diagnostics=diagnostics)

    split_counts = assigned["split"].value_counts().to_dict()
    if any(split_counts.get(split_name, 0) == 0 for split_name in SPLIT_NAMES):
        diagnostics["split_counts"] = split_counts
        raise CatalogSplitError("At least one split is empty after assignment.", diagnostics=diagnostics)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    enriched = _augment_table(inspection.table, assigned)
    pq.write_table(enriched, target_dir / "catalog_with_geo_splits.parquet")

    assigned.sort_values("row_index").to_csv(target_dir / "split_assignments.csv", index=False)

    for split_name in SPLIT_NAMES:
        indices = assigned.loc[assigned["split"] == split_name, "row_index"].tolist()
        split_table = enriched.take(pa.array(indices, type=pa.int64()))
        pq.write_table(split_table, target_dir / f"{split_name}.parquet")

    summary = _build_summary(assigned, diagnostics, config)
    summary["generated_at"] = datetime.now().astimezone().isoformat()
    summary_path = target_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def split_rfinject_csv_catalog(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    config: SplitConfig | None = None,
    bucket: str = "ESA-philab/RFInject-v1-L0",
    child_source: str = "bucket",
    child_products: Iterable[str] | None = None,
    max_workers: int = 8,
    fetcher: Any = None,
) -> dict[str, Any]:
    """Split the RFInject CSV catalog after linking each parent to bucket children."""

    config = config or SplitConfig(lat_sections=3, lon_sections=3)
    config.validate()

    csv_frame = load_rfinject_csv_catalog(input_path)
    if child_source == "bucket":
        if child_products is None:
            from .utils import list_hf_bucket_zarrs

            child_products = list_hf_bucket_zarrs(bucket)
        linked_frames = link_rfinject_catalog_to_children(csv_frame, child_products, input_path=input_path)
    elif child_source == "cdse-nodes":
        linked_frames = link_rfinject_catalog_to_cdse_nodes(
            csv_frame,
            input_path=input_path,
            max_workers=max_workers,
            fetcher=fetcher,
        )
    else:
        raise ValueError("child_source must be either 'bucket' or 'cdse-nodes'.")
    diagnostics = dict(linked_frames.diagnostics)

    if linked_frames.linked_parents.empty:
        diagnostics["validation_errors"] = ["No CSV parents could be linked to bucket child products."]
        raise CatalogSplitError(
            "No CSV parents could be linked to bucket child products.",
            diagnostics=diagnostics,
        )

    sectioned = _assign_geo_sections(linked_frames.linked_parents, config)
    _validate_split_inputs(sectioned, diagnostics, config)

    assigned_parents = assign_splits(sectioned, config)
    assigned_parents = _ensure_non_empty_splits(assigned_parents, diagnostics)
    assigned_parents = _rebalance_split_totals(assigned_parents, config)
    if assigned_parents["split"].isna().any():
        diagnostics["unassigned_rows"] = int(assigned_parents["split"].isna().sum())
        raise CatalogSplitError("Split assignment left some linked parents unassigned.", diagnostics=diagnostics)

    assigned_parents = assigned_parents.sort_values(["split", "geo_section", "Name"]).reset_index(drop=True)
    assigned_parents["child_products_json"] = assigned_parents["child_products"].map(json.dumps)
    if "child_records" in assigned_parents.columns:
        assigned_parents["child_records_json"] = assigned_parents["child_records"].map(json.dumps)

    linked_children = linked_frames.children.merge(
        assigned_parents.loc[
            :,
            [
                "Id",
                "match_key",
                "geo_section",
                "split",
                "child_products_json",
                *([ "child_records_json" ] if "child_records_json" in assigned_parents.columns else []),
            ],
        ],
        left_on=["parent_id", "match_key"],
        right_on=["Id", "match_key"],
        how="inner",
        validate="many_to_one",
    )
    drop_columns = ["Id", "child_products_json"]
    if "child_records_json" in linked_children.columns:
        drop_columns.append("child_records_json")
    linked_children = linked_children.drop(columns=drop_columns).sort_values(
        ["split", "geo_section", "parent_name", "child_name"]
    )

    unmatched_parents = linked_frames.unmatched_parents.sort_values(["Name"]).reset_index(drop=True)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    parent_drop_columns = ["child_products"]
    if "child_records" in assigned_parents.columns:
        parent_drop_columns.append("child_records")
    unmatched_drop_columns = ["child_products"] + (["child_records"] if "child_records" in unmatched_parents.columns else [])

    parent_output = assigned_parents.drop(columns=parent_drop_columns).copy()
    unmatched_output = unmatched_parents.drop(columns=unmatched_drop_columns).copy()

    _write_dataframe_outputs(parent_output, target_dir / "linked_parents")
    _write_dataframe_outputs(linked_children, target_dir / "linked_children")
    _write_dataframe_outputs(unmatched_output, target_dir / "unmatched_parents")

    for split_name in SPLIT_NAMES:
        split_parents = parent_output.loc[parent_output["split"] == split_name].reset_index(drop=True)
        split_children = linked_children.loc[linked_children["split"] == split_name].reset_index(drop=True)
        _write_dataframe_outputs(split_parents, target_dir / f"{split_name}_parents")
        _write_dataframe_outputs(split_children, target_dir / f"{split_name}_children")

    summary = _build_rfinject_summary(
        parent_output,
        linked_children,
        unmatched_output,
        diagnostics,
        config,
        bucket,
        child_source,
    )
    summary["generated_at"] = datetime.now().astimezone().isoformat()
    summary_path = target_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", help="Path to the source catalog parquet file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the split parquet files and reports will be written.",
    )
    parser.add_argument(
        "--catalog-format",
        choices=("auto", "parquet", "rfinject-csv"),
        default="auto",
        help="How to interpret the input catalog. Defaults to auto-detection from the file suffix.",
    )
    parser.add_argument(
        "--bucket",
        default="ESA-philab/RFInject-v1-L0",
        help="Bucket ID used to resolve RFInject child products when the input is an RFInject CSV.",
    )
    parser.add_argument(
        "--child-source",
        choices=("bucket", "cdse-nodes"),
        default="bucket",
        help="How to resolve RFInject child products for CSV catalogs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum worker threads for live CDSE node lookups.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--lat-sections", type=int, default=4)
    parser.add_argument("--lon-sections", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rows", type=int, default=3)
    parser.add_argument("--min-unique-sections", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the geographic splitter CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = Path(args.input_path)
    catalog_format = args.catalog_format
    if catalog_format == "auto":
        catalog_format = "rfinject-csv" if input_path.suffix.lower() == ".csv" else "parquet"

    lat_sections = args.lat_sections
    lon_sections = args.lon_sections
    if catalog_format == "rfinject-csv" and lat_sections == 4 and lon_sections == 4:
        lat_sections = lon_sections = 3

    config = SplitConfig(
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        lat_sections=lat_sections,
        lon_sections=lon_sections,
        seed=args.seed,
        min_rows=args.min_rows,
        min_unique_sections=args.min_unique_sections,
    )

    try:
        if catalog_format == "rfinject-csv":
            summary = split_rfinject_csv_catalog(
                args.input_path,
                args.output_dir,
                config=config,
                bucket=args.bucket,
                child_source=args.child_source,
                max_workers=args.max_workers,
            )
        else:
            summary = split_catalog_geographically(args.input_path, args.output_dir, config=config)
    except CatalogSplitError as exc:
        report_path = write_failure_report(args.output_dir, exc.diagnostics, str(exc))
        print(f"Split failed. Diagnostics written to {report_path}")
        return 2

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
