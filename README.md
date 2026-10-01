# AIr-wise: AI-Based Uncertainty-Aware Air Quality Assessment

This repository contains the development code for the ECMWF Code for Earth 2026 project **AIr-wise**, which aims to support uncertainty-aware air quality assessment and automated reporting using CAMS air quality forecast and analysis products.

## Project overview

AIr-wise combines data preparation, forecast-error modelling, AQI and confidence estimation, and automated reporting:

- CAMS, OpenIFS, ERA5, and geospatial data preparation;
- machine-learning forecast-error modelling;
- AQI and confidence estimation; and
- automated regional and city-level air-quality reporting.

Production reporting currently supports Germany and uses the OpenIFS control
forecast (`oper/fc`).

## How to run it

### Prerequisites

- **Python 3.10.**
- **A free Copernicus Atmosphere Data Store account** for CAMS downloads.
  Register at <https://ads.atmosphere.copernicus.eu/>, accept the CAMS Europe
  dataset licence, and copy your personal access token from your profile page.
- OpenIFS and CAMS Policy do not require additional credentials with the
  default download configuration.

### Step 1 — Create a clean environment

From a repository checkout:

```bash
conda create -n airwise python=3.10
conda activate airwise
python -m pip install --upgrade pip
```

Activate this environment before the later commands. In a new terminal, run
`conda activate airwise` again first.

### Step 2 — Configure your CAMS credentials

1. Register and log in at the
   [Atmosphere Data Store](https://ads.atmosphere.copernicus.eu/).
2. Open [How to use the ADS API](https://ads.atmosphere.copernicus.eu/how-to-api)
   and copy the personal access token shown there.
3. Accept the dataset licence on the CAMS Europe forecast download form.
4. Save the ADS token in `$HOME/.cdsapirc`:

```text
url: https://ads.atmosphere.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

See [CAMS credentials](docs/downloading.md#cams-credentials) for the same setup.

### Step 3 — Choose what to run

Generating a report uses the pretrained models included in this repository.
Retraining replaces those models with ones you train yourself.

#### I want to generate an air-quality report

Install PyTorch from the CPU package index first. Do not pin its version;
that index then supplies a build for this computer. Install the report
dependencies afterwards:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[report]"
```

Generate the daily German JSON and PDF bulletin:

```bash
airwise-daily --date YYYY-MM-DD --device cpu
```

See the [quick start](docs/quickstart.md) for data-access prerequisites,
configuration, outputs, and the individual pipeline steps.

#### I want to retrain the AI models

Install the independent training profile:

```bash
python -m pip install -e ".[train]"
```

After downloading and aligning the historical CAMS and ERA5 data, build the
training stores and train a model:

```bash
airwise-build-training-zarr --years 2023 2024 2025
airwise-train-model
```

See the [training guide](docs/training.md) for the complete data preparation,
regridding, training, checkpoint, and TensorBoard monitoring.

## Documentation

- [Installation profiles](docs/installation.md)
- [Daily-report quick start](docs/quickstart.md)
- [Model training guide](docs/training.md)
- [Downloading data](docs/downloading.md)
- [Configuration](docs/configuration.md)
- [Architecture](docs/architecture.md)
