# AIr-wise: AI-Based Uncertainty-Aware Air Quality Assessment

This repository contains the development code for the ECMWF Code for Earth 2026 project **AIr-wise**, which aims to support uncertainty-aware air quality assessment and automated reporting using CAMS air quality forecast and analysis products.

## Project overview

AIr-wise develops an AI-driven workflow to move beyond deterministic AQI estimation by combining forecast error modelling, AQI computation, confidence score estimation, and automated country-level air quality reporting.

The workflow is structured into four main components:

1. Data preparation and harmonisation of CAMS forecast, analysis, meteorological, and geospatial data.
2. Forecast error modelling using machine learning and deep learning approaches.
3. AQI computation and confidence score estimation.
4. Automated report generation for country-level and city-level air quality summaries.

## Repository structure

```text
src/airwise/
    cli/                # Argument parsing and exit codes
    pipelines/          # Explicit acquisition, AQI, confidence, and report flows
    domain/             # AQI, pollutant, region, and bulletin models
    config/             # External YAML loading and typed settings
    resources/          # Installed AQI, template, and JSON-schema definitions
    data/
        download/       # Network-facing provider adapters
        io/             # Local NetCDF and Zarr products
        preprocessing/  # Regridding and model-independent transforms
    modelling/          # Features, models, inference, and training subsystem
    uncertainty/        # Probability and AQI-confidence algorithms
    reporting/          # JSON/PDF presentation
```

See [docs/architecture.md](docs/architecture.md) for dependency rules and the
daily production flow. Production OpenIFS inference uses the control forecast
only (`number=0`).

## Install

Copy `configs/local.example.yaml` to `configs/local.yaml` and set the data directories for this machine. `configs/default.yaml` keeps portable relative paths. `configs/local.yaml` is not committed.

Daily acquisition, pretrained-model inference, and reporting do not require
model retraining or CUDA. For a CPU-only report environment, install the
CPU-only PyTorch wheel before the report extra:

```bash
python -m pip install torch \
  --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[report]"
```

Retraining is an independent optional profile containing PyTorch, TensorBoard,
Dask, and Zarr:

```bash
pip install -e ".[train]"
```

Install `.[report,train]` only when both workflows are needed. See the
[installation profiles](docs/installation.md) for CPU-only reporting, optional
GPU training, development dependencies, and the complete daily command
sequence.

## Daily commands

Run the full daily report in one command:

```bash
airwise-daily --date 20240430 --device cpu
```

The same steps remain available separately:

```bash
airwise-acquire-daily-inputs --date 20240430
airwise-compute-aqi --date 20240430
airwise-compute-confidence --date 20240430 --device cpu
airwise-report \
  --date 20240430 \
  --region-config configs/regions/germany.yaml \
  --output reports/germany_20240430.json \
  --device cpu
```