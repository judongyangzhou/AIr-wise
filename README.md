# AIr-wise: AI-Based Uncertainty-Aware Air Quality Assessment

This repository contains the development code for the ECMWF Code for Earth 2026 project **AIr-wise**, which aims to support uncertainty-aware air quality assessment and automated reporting using CAMS air quality forecast and analysis products.

## Project overview

AIr-wise develops an AI-driven workflow to move beyond deterministic AQI estimation by combining forecast error modelling, AQI computation, confidence score estimation, and automated country-level air quality reporting.

The workflow combines:

- CAMS, OpenIFS, ERA5, and geospatial data preparation;
- machine-learning forecast error modelling;
- AQI and confidence score estimation; and
- automated regional and city-level air-quality reporting.

Production reporting currently supports Germany and uses the OpenIFS control
forecast (`number=0`).

## Choose your workflow

For either workflow, create and activate the Python 3.10 Conda environment
from a repository checkout:

```bash
conda create -n airwise python=3.10
conda activate airwise
python -m pip install "pip==26.2.1"
```

### I want to generate an air-quality report

Install the pinned CPU-only PyTorch wheel and the report dependencies using
the validated Python 3.10 constraints:

```bash
python -m pip install "torch==2.14.1" \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install \
  -c constraints/py310.txt \
  -e ".[report]"
```

Before the first report, accept the CAMS Europe dataset licence and save the
Atmosphere Data Store token in `$HOME/.cdsapirc`:

```text
url: https://ads.atmosphere.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

See [CAMS credentials](docs/downloading.md#cams-credentials) for account and
licence setup. OpenIFS and CAMS Policy do not require additional credentials
with the default download configuration.

Generate the daily German JSON and PDF bulletin:

```bash
airwise-daily --date YYYY-MM-DD --device cpu
```

See the [quick start](docs/quickstart.md) for data-access prerequisites,
configuration, outputs, and the individual pipeline steps.

### I want to retrain the AI models

Install the independent training profile:

```bash
python -m pip install \
  -c constraints/py310.txt \
  -e ".[train]"
```

After downloading and aligning the historical CAMS and ERA5 data, build the
training stores and train a model:

```bash
airwise-build-training-zarr --years 2023 2024 2025
airwise-train-model
```

See the [training guide](docs/training.md) for the complete data preparation,
regridding, training, checkpoint, and TensorBoard workflow.

## Documentation

- [Installation profiles](docs/installation.md)
- [Daily-report quick start](docs/quickstart.md)
- [Model training guide](docs/training.md)
- [Downloading data](docs/downloading.md)
- [Configuration](docs/configuration.md)
- [Architecture](docs/architecture.md)
