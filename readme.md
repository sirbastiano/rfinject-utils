# rfinject-utils

<p align="center">
  <img src="./src/rfinject.png" alt="rfinject-utils logo" width="420"/>
</p>

<p align="center">
  <a href="https://github.com/sirbastiano/rfinject-utils/blob/main/readme.md"><img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge&logo=gitbook" alt="Status"/></a>
  <a href="https://github.com/sirbastiano/rfinject-utils/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge&logo=apache" alt="License"/></a>
  <a href="https://github.com/sirbastiano/rfinject-utils/stargazers"><img src="https://img.shields.io/github/stars/sirbastiano/rfinject-utils?style=for-the-badge" alt="GitHub stars"/></a>
  <a href="https://github.com/sirbastiano/rfinject-utils/forks"><img src="https://img.shields.io/github/forks/sirbastiano/rfinject-utils?style=for-the-badge" alt="GitHub forks"/></a>
  <a href="https://github.com/sirbastiano/rfinject-utils/commits"><img src="https://img.shields.io/github/last-commit/sirbastiano/rfinject-utils?style=for-the-badge" alt="Last commit"/></a>
  <a href="https://github.com/sirbastiano/rfinject-utils/issues"><img src="https://img.shields.io/github/issues/sirbastiano/rfinject-utils?style=for-the-badge" alt="Open issues"/></a>
  <a href="https://github.com/sirbastiano/rfinject-utils"><img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python" alt="Python version"/></a>
</p>
<p align="center">
  <a href="https://github.com/sirbastiano/rfinject-utils"><img src="https://img.shields.io/badge/Repository-Open%20on%20GitHub-181717?style=for-the-badge&logo=github" alt="Repository"/></a>
  <a href="docs/index.html"><img src="https://img.shields.io/badge/Docs-Available-brightgreen?style=for-the-badge&logo=readthedocs" alt="Documentation"/></a>
  <a href="https://huggingface.co/datasets/RFInject/v1"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface" alt="Hugging Face dataset"/></a>
</p>

## RF RFI analysis toolkit for SAR Zarr workflows

rfinject-utils is a Python toolkit for radio-frequency interference (RFI) analysis and injection workflows on synthetic aperture radar (SAR) data.

Developed at ESA Φ-lab, it provides practical utilities for:
- Navigating complex Zarr datasets
- Handling burst-level SAR products (echo, RFI, and fused views)
- Accessing and filtering metadata at group/attribute level
- Building reproducible, analysis-ready data pipelines

Repository: [https://github.com/sirbastiano/rfinject-utils.git](https://github.com/sirbastiano/rfinject-utils.git)

## Key features

- **Structured Zarr exploration** for fast inspection of nested SAR products  
- **Burst-aware access** to echo, RFI, and fused groups  
- **Metadata extraction helpers** for consistent experiment tracking  
- **Efficient slicing utilities** for repeatable, scalable analysis  
- **Optional interactive tooling** via environment extras

## Evaluation examples

- [`notebooks/rfi_iou_evaluation.ipynb`](notebooks/rfi_iou_evaluation.ipynb): end-to-end example for evaluating burst/segment detections with IoU, precision, recall, F1, and mean IoU.

## Quick start

### 1. Install PDM

```bash
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

### 2. Install project dependencies

```bash
pdm install
```

### 3. Install optional extras (as needed)

```bash
pdm install -G jupyter_env
pdm install -G viz
pdm install -G docs
```

## Data download

Example datasets are available on Hugging Face:

```bash
hf download RFInject/v1 --repo-type dataset --max-workers 64 --local-dir /path/to/folder
7z x -r -y /path/to/folder/RFInject.zip
```

## Basic usage

```python
import zarr

from rfinject.utils import explore_zarr_structure, access_attributes, get_burst_info

zarr_data = zarr.open("path/to/your/data.zarr", mode="r")

explore_zarr_structure(zarr_data)
burst_info = get_burst_info(zarr_data)
attrs = access_attributes(zarr_data, "burst_0")
```

Run notebook or script workflows with PDM:

```bash
pdm run python your_script.py
```

## Project structure

```text
rfinject/
├── pyproject.toml        # Project metadata and dependency definitions
├── rfinject/
│   ├── utils.py          # Core utilities for Zarr handling
│   ├── viz.py            # Visualization helpers
│   └── __init__.py
└── readme.md             # Project documentation
```

## Documentation

Static docs are available at:

- `docs/index.html` (home)
- `docs/getting-started.html`
- `docs/api-reference.html`
- `docs/visualization.html`

The documentation package includes three alternate display themes (dark, cool, aurora) for different working environments.

## Development and maintenance

rfinject-utils is maintained at ESA Φ-lab to support Earth observation and radar research workflows.

## License

Apache License 2.0

## Contact

For questions and support, contact `roberto.delprete@esa.int`.
