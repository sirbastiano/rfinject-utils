from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rfinject.catalog_split import (
    CatalogSplitError,
    SplitConfig,
    extract_online_child_records,
    split_catalog_geographically,
    split_rfinject_csv_catalog,
)


def _bbox_array(records: list[dict[str, float]]) -> pa.Array:
    return pa.array(
        records,
        type=pa.struct(
            [
                ("xmax", pa.float64()),
                ("xmin", pa.float64()),
                ("ymax", pa.float64()),
                ("ymin", pa.float64()),
            ]
        ),
    )


def test_split_catalog_geographically_writes_expected_splits(tmp_path: Path) -> None:
    rows = []
    item_ids = []
    for lat_center in (10.0, 20.0):
        for lon_center in (30.0, 40.0):
            for sample_index in range(4):
                item_ids.append(f"item-{lat_center}-{lon_center}-{sample_index}")
                rows.append(
                    {
                        "xmax": lon_center + 0.5,
                        "xmin": lon_center - 0.5,
                        "ymax": lat_center + 0.5,
                        "ymin": lat_center - 0.5,
                    }
                )

    table = pa.table(
        {
            "id": pa.array(item_ids, type=pa.string()),
            "bbox": _bbox_array(rows),
        }
    )
    input_path = tmp_path / "catalog.parquet"
    pq.write_table(table, input_path)

    output_dir = tmp_path / "split"
    summary = split_catalog_geographically(
        input_path,
        output_dir,
        config=SplitConfig(
            train_ratio=0.5,
            validation_ratio=0.25,
            test_ratio=0.25,
            lat_sections=2,
            lon_sections=2,
            seed=7,
        ),
    )

    assert summary["status"] == "success"
    assert summary["rows"] == 16
    assert summary["unique_sections"] == 4
    assert summary["split_counts"] == {"train": 8, "validation": 4, "test": 4}

    assignment_rows = output_dir.joinpath("split_assignments.csv").read_text().strip().splitlines()
    assert len(assignment_rows) == 17
    assert pq.read_table(output_dir / "train.parquet").num_rows == 8
    assert pq.read_table(output_dir / "validation.parquet").num_rows == 4
    assert pq.read_table(output_dir / "test.parquet").num_rows == 4

    summary_path = output_dir / "split_summary.json"
    assert json.loads(summary_path.read_text())["section_counts"] == {
        "lat0_lon0": 4,
        "lat0_lon1": 4,
        "lat1_lon0": 4,
        "lat1_lon1": 4,
    }


def test_split_catalog_geographically_rejects_catalog_without_geography(tmp_path: Path) -> None:
    table = pa.table(
        {
            "id": pa.array(["README.md", "RFInject.tar.zst"], type=pa.string()),
            "bbox": _bbox_array(
                [
                    {"xmax": 0.0, "xmin": 0.0, "ymax": 0.0, "ymin": 0.0},
                    {"xmax": 0.0, "xmin": 0.0, "ymax": 0.0, "ymin": 0.0},
                ]
            ),
        }
    )
    input_path = tmp_path / "catalog.parquet"
    pq.write_table(table, input_path)

    with pytest.raises(CatalogSplitError, match="usable geography"):
        split_catalog_geographically(input_path, tmp_path / "split")


def _make_geofootprint(lon_center: float, lat_center: float) -> str:
    lon_min = lon_center - 0.5
    lon_max = lon_center + 0.5
    lat_min = lat_center - 0.5
    lat_max = lat_center + 0.5
    return str(
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [lon_min, lat_min],
                    [lon_max, lat_min],
                    [lon_max, lat_max],
                    [lon_min, lat_max],
                    [lon_min, lat_min],
                ]
            ],
        }
    )


def _make_parent_name(start: str, end: str, orbit: str, code: str, crc: str) -> str:
    return f"S1A_IW_RAW__0SDV_{start}_{end}_{orbit}_{code}_{crc}.SAFE"


def _make_child_name(start: str, end: str, orbit: str, code: str) -> str:
    return f"s1a-iw-raw-s-vh-{start.lower()}-{end.lower()}-{orbit}-{code.lower()}.zarr"


def test_split_rfinject_csv_catalog_links_parents_and_children(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    child_products: list[str] = []

    section_specs = [
        (-20.0, -40.0),
        (-20.0, 40.0),
        (20.0, -40.0),
        (20.0, 40.0),
    ]

    sample_id = 0
    for lat_center, lon_center in section_specs:
        for section_index in range(4):
            start = f"202401{sample_id + 1:02d}T010101"
            end = f"202401{sample_id + 1:02d}T010133"
            orbit = f"{50000 + sample_id:06d}"
            code = f"{600000 + sample_id:06X}"[-6:]
            crc = f"{7000 + sample_id:04X}"[-4:]
            rows.append(
                {
                    "Id": f"parent-{sample_id}",
                    "Name": _make_parent_name(start, end, orbit, code, crc),
                    "GeoFootprint": _make_geofootprint(lon_center + section_index * 0.01, lat_center + section_index * 0.01),
                }
            )
            child_products.append(_make_child_name(start, end, orbit, code))
            sample_id += 1

    for unmatched_index in range(2):
        rows.append(
            {
                "Id": f"unmatched-{unmatched_index}",
                "Name": _make_parent_name(
                    f"202402{unmatched_index + 1:02d}T020202",
                    f"202402{unmatched_index + 1:02d}T020234",
                    f"{60000 + unmatched_index:06d}",
                    f"{610000 + unmatched_index:06X}"[-6:],
                    f"{7100 + unmatched_index:04X}"[-4:],
                ),
                "GeoFootprint": _make_geofootprint(80.0 + unmatched_index, 50.0 + unmatched_index),
            }
        )

    input_path = tmp_path / "RFInject.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    output_dir = tmp_path / "split"
    summary = split_rfinject_csv_catalog(
        input_path,
        output_dir,
        config=SplitConfig(
            train_ratio=0.5,
            validation_ratio=0.25,
            test_ratio=0.25,
            lat_sections=2,
            lon_sections=2,
            seed=11,
        ),
        child_products=child_products,
    )

    assert summary["status"] == "success"
    assert summary["total_parents"] == 18
    assert summary["linked_parents"] == 16
    assert summary["unmatched_parents"] == 2
    assert summary["linked_children"] == 16
    assert summary["parent_split_counts"] == {"train": 8, "validation": 4, "test": 4}
    assert summary["child_split_counts"] == {"train": 8, "validation": 4, "test": 4}

    linked_parents = pd.read_csv(output_dir / "linked_parents.csv")
    linked_children = pd.read_csv(output_dir / "linked_children.csv")
    unmatched_parents = pd.read_csv(output_dir / "unmatched_parents.csv")

    assert len(linked_parents) == 16
    assert len(linked_children) == 16
    assert len(unmatched_parents) == 2
    assert set(linked_children["parent_name"]) == set(linked_parents["Name"])
    assert set(linked_parents["split"]) == {"train", "validation", "test"}


def test_extract_online_child_records_groups_data_annotation_and_index() -> None:
    records = extract_online_child_records(
        [
            {"Name": "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0.dat"},
            {"Name": "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0-annot.dat"},
            {"Name": "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0-index.dat"},
            {"Name": "manifest.safe"},
        ]
    )

    assert records == [
        {
            "child_product": "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0",
            "child_name": "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0",
            "has_data": True,
            "has_annot": True,
            "has_index": True,
            "node_names": [
                "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0-annot.dat",
                "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0-index.dat",
                "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0.dat",
            ],
            "node_names_json": json.dumps(
                [
                    "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0-annot.dat",
                    "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0-index.dat",
                    "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0.dat",
                ]
            ),
        }
    ]


def test_split_rfinject_csv_catalog_with_cdse_nodes_fetcher(tmp_path: Path) -> None:
    rows = [
        {
            "Id": "parent-1",
            "Name": "S1A_IW_RAW__0SDV_20240101T010101_20240101T010133_050000_0927C0_1AAA.SAFE",
            "GeoFootprint": _make_geofootprint(-10.0, -10.0),
        },
        {
            "Id": "parent-2",
            "Name": "S1A_IW_RAW__0SDV_20240102T010101_20240102T010133_050001_0927C1_1AAB.SAFE",
            "GeoFootprint": _make_geofootprint(10.0, -10.0),
        },
        {
            "Id": "parent-3",
            "Name": "S1A_IW_RAW__0SDV_20240103T010101_20240103T010133_050002_0927C2_1AAC.SAFE",
            "GeoFootprint": _make_geofootprint(-10.0, 10.0),
        },
        {
            "Id": "parent-4",
            "Name": "S1A_IW_RAW__0SDV_20240104T010101_20240104T010133_050003_0927C3_1AAD.SAFE",
            "GeoFootprint": _make_geofootprint(10.0, 10.0),
        },
    ]
    input_path = tmp_path / "RFInject.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    payloads = {
        "parent-1": [
            {"Name": "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0.dat"},
            {"Name": "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0-annot.dat"},
            {"Name": "s1a-iw-raw-s-vh-20240101t010101-20240101t010133-050000-0927c0-index.dat"},
        ],
        "parent-2": [
            {"Name": "s1a-iw-raw-s-vv-20240102t010101-20240102t010133-050001-0927c1.dat"},
            {"Name": "s1a-iw-raw-s-vv-20240102t010101-20240102t010133-050001-0927c1-annot.dat"},
            {"Name": "s1a-iw-raw-s-vv-20240102t010101-20240102t010133-050001-0927c1-index.dat"},
        ],
        "parent-3": [
            {"Name": "s1a-iw-raw-s-vh-20240103t010101-20240103t010133-050002-0927c2.dat"},
            {"Name": "s1a-iw-raw-s-vh-20240103t010101-20240103t010133-050002-0927c2-annot.dat"},
            {"Name": "s1a-iw-raw-s-vh-20240103t010101-20240103t010133-050002-0927c2-index.dat"},
        ],
        "parent-4": [
            {"Name": "s1a-iw-raw-s-vv-20240104t010101-20240104t010133-050003-0927c3.dat"},
            {"Name": "s1a-iw-raw-s-vv-20240104t010101-20240104t010133-050003-0927c3-annot.dat"},
            {"Name": "s1a-iw-raw-s-vv-20240104t010101-20240104t010133-050003-0927c3-index.dat"},
        ],
    }

    def fake_fetcher(product_id: str, product_name: str) -> list[dict[str, str]]:
        return payloads[product_id]

    output_dir = tmp_path / "split_cdse"
    summary = split_rfinject_csv_catalog(
        input_path,
        output_dir,
        config=SplitConfig(
            train_ratio=0.5,
            validation_ratio=0.25,
            test_ratio=0.25,
            lat_sections=2,
            lon_sections=2,
            seed=5,
        ),
        child_source="cdse-nodes",
        fetcher=fake_fetcher,
        max_workers=2,
    )

    assert summary["status"] == "success"
    assert summary["child_source"] == "cdse-nodes"
    assert summary["linked_parents"] == 4
    assert summary["linked_children"] == 4
    assert summary["parent_split_counts"] == {"train": 2, "validation": 1, "test": 1}
    assert summary["child_split_counts"] == {"train": 2, "validation": 1, "test": 1}

    linked_children = pd.read_csv(output_dir / "linked_children.csv")
    assert linked_children["has_data"].all()
    assert linked_children["has_annot"].all()
    assert linked_children["has_index"].all()
