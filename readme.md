# rfinject-utils

<p align="right">
  <img src="./docs/ESAΦ-lab_WT.png" alt="ESA Φ-lab logo" width="240"/>
</p>
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
  <a href="https://huggingface.co/buckets/ESA-philab/RFInject-v1-L0"><img src="https://img.shields.io/badge/Bucket-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface" alt="Hugging Face bucket"/></a>
</p>

## RF RFI analysis toolkit for SAR Zarr workflows

rfinject-utils is a Python toolkit for radio-frequency interference (RFI) analysis and injection workflows on synthetic aperture radar (SAR) data.

In this repository:
- a SAR product is organized as a sequence of burst groups
- each burst contains complex arrays such as `echo` and `rfi`
- metadata and numeric payload live together in Zarr, so you can inspect structure first and mirror only the chunks you need

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

## SAR and Zarr primer

- **SAR**: Synthetic aperture radar measures complex radar echoes while the satellite moves along its orbit. In RFInject, those echoes are grouped into bursts, which are the natural units used for browsing, sampling, visualization, and machine-learning pipelines.
- **RFI**: Radio-frequency interference is unwanted energy that contaminates the SAR signal. RFInject stores interference-aligned arrays next to the clean or baseline echo arrays so analysis and evaluation can happen on the same burst grid.
- **Zarr**: Zarr is a hierarchical, chunked array format. It behaves like a directory tree of groups, arrays, and attributes. That makes it a good fit for SAR because you can open metadata quickly, inspect shapes and attributes, and then download only the burst payload you actually need.

## Notebooks

- [`notebooks/how_to_start.ipynb`](notebooks/how_to_start.ipynb): guided first walkthrough of one SAR Zarr product, including burst structure, metadata, echo arrays, and RFI views.
- [`notebooks/how_to_pytorch_dataloader.ipynb`](notebooks/how_to_pytorch_dataloader.ipynb): burst-level PyTorch `Dataset` and `DataLoader` example that mirrors the selected train, validation, and test burst subset under `./data` by default, with optional full-scene caching and fractional product sampling.
- [`notebooks/rfi_iou_evaluation.ipynb`](notebooks/rfi_iou_evaluation.ipynb): end-to-end example for evaluating burst/segment detections with IoU, precision, recall, F1, and mean IoU.

For a quick smoke run of the PyTorch notebook, start with a small `sample_fraction` such as `0.001` or cap `max_scenes_per_split`, because even a small fraction of a large raw scene can still be several GiB.

## Install the environment

Choose one workflow. All three install the same project; the difference is only the environment manager.

### Option A: PDM

```bash
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
pdm install
pdm install -G jupyter_env -G viz -G docs
```

For the PyTorch notebook, add:

```bash
pdm run python -m pip install torch
```

### Option B: virtualenv

```bash
python3 -m pip install --user virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[jupyter_env,viz,docs]"
```

For the PyTorch notebook, add:

```bash
python -m pip install torch
```

### Option C: uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra jupyter_env --extra viz --extra docs
```

For the PyTorch notebook, add:

```bash
uv pip install --python .venv/bin/python torch
```

### Jupyter kernel registration

After installing dependencies, register the environment you want to use in Jupyter:

```bash
# PDM
pdm run python -m ipykernel install --user --name rfinject-pdm --display-name "Python (rfinject-pdm)"

# virtualenv
python -m ipykernel install --user --name rfinject-venv --display-name "Python (rfinject-venv)"

# uv
uv run python -m ipykernel install --user --name rfinject-uv --display-name "Python (rfinject-uv)"
```

Then select the matching kernel inside JupyterLab or Notebook before running the notebooks in [`notebooks/`](notebooks/).

## Bucket-native data access

RFInject sample products are published directly in the Hugging Face bucket
[`ESA-philab/RFInject-v1-L0`](https://huggingface.co/buckets/ESA-philab/RFInject-v1-L0).

List the top-level Zarr products:

```python
from rfinject import DEFAULT_HF_BUCKET_ID, list_hf_bucket_zarrs

products = list_hf_bucket_zarrs(DEFAULT_HF_BUCKET_ID, limit=5)
print(products)
```

Open one product for metadata-only inspection without downloading chunk payloads:

```python
from rfinject import DEFAULT_HF_BUCKET_ID, open_hf_bucket_zarr

zarr_data = open_hf_bucket_zarr(
    DEFAULT_HF_BUCKET_ID,
    "s1a-iw-raw-s-hh-20240116t204634-20240116t204707-052137-064d52.zarr",
    metadata_only=True,
)
```

Mirror a full file or a complete Zarr product locally:

```python
from rfinject import DEFAULT_HF_BUCKET_ID, download_hf_bucket_path

download_hf_bucket_path(
    DEFAULT_HF_BUCKET_ID,
    "s1a-iw-raw-s-hh-20240116t204634-20240116t204707-052137-064d52.zarr",
    "/path/to/folder",
)
```

## Basic usage

```python
from rfinject import (
    DEFAULT_HF_BUCKET_ID,
    access_attributes,
    explore_zarr_structure,
    get_burst_info,
    open_hf_bucket_zarr,
)

zarr_data = open_hf_bucket_zarr(
    DEFAULT_HF_BUCKET_ID,
    "s1a-iw-raw-s-hh-20240116t204634-20240116t204707-052137-064d52.zarr",
    metadata_only=True,
)

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
├── notebooks/            # Example notebooks and guided workflows
├── rfinject/
│   ├── __init__.py       # Public exports
│   ├── utils.py          # Core utilities for Zarr handling
│   ├── viz.py            # Visualization helpers
├── docs/                 # Static HTML documentation
├── tests/
│   └── test_utils.py     # Bucket and Zarr regression tests
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
