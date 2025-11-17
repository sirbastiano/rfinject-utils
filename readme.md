# RFInject

A radio frequency interference (RFI) injection and analysis tool developed at ESA Φ-lab.

## Overview

RFInject is a Python toolkit for working with synthetic aperture radar (SAR) data, specifically designed for RFI analysis and injection. The tool provides utilities for exploring Zarr data structures, accessing SAR burst data, and analyzing RFI patterns in radar imagery.

## Features

- **Zarr Data Exploration**: Comprehensive tools for exploring and navigating Zarr data structures
- **SAR Data Access**: Easy access to burst data including echo, RFI, and combined datasets
- **Attribute Management**: Flexible attribute access across the data hierarchy
- **Data Slicing**: Efficient data slicing and inspection utilities

## Installation

This project uses [PDM](https://pdm-project.org/) for dependency management.

1.  **Install PDM**:
    Follow the official instructions to install PDM on your system. For example:
    ```bash
    curl -sSL https://pdm-project.org/install-pdm.py | python3 -
    ```

2.  **Install dependencies**:
    Clone the repository and run `pdm install` in the project root directory:
    ```bash
    cd rfinject
    pdm install
    ```
    This will create a virtual environment and install all necessary packages like `numpy`, `matplotlib`, and `zarr`.

## Usage

To run scripts or notebooks, use `pdm run`:

```python
import zarr
from rfinject.utils import explore_zarr_structure, access_attributes, get_burst_info

# Open your Zarr dataset
zarr_data = zarr.open('path/to/your/data.zarr', mode='r')

# Explore the structure
explore_zarr_structure(zarr_data)

# Get burst information
burst_info = get_burst_info(zarr_data)

# Access specific attributes
attrs = access_attributes(zarr_data, 'burst_0')
```

## Project Structure

```
rfinject/
├── pyproject.toml    # Project metadata and dependencies for PDM
├── rfinject/
│   └── utils.py      # Core utilities for Zarr data handling
│   └── viz.py        # Core utilities for plotting
└── readme.md         # This file
```

## Development

This project is developed at **ESA Φ-lab** (Phi-lab), the European Space Agency's exploration and innovation lab focused on Earth observation and artificial intelligence.

## License

APACHE-2.0. Developed for ESA Φ-lab research purposes.

## Contact

For questions and support, please contact roberto.delprete@esa.int or ESA Φ-lab.
