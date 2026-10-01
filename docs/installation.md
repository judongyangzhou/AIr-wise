# Installation profiles

This page is for users choosing an environment. For the commands to run after
installation, continue to the [daily-report quick start](quickstart.md) or the
[model training guide](training.md).

AIr-wise has two independent dependency profiles:

- `report` downloads daily inputs, runs the bundled pretrained models, and
  writes JSON/PDF reports.
- `train` prepares reusable Zarr stores and retrains the neural models.

## Common setup

AIr-wise uses Python 3.10. From a repository checkout, create and activate the
Conda environment:

```bash
conda create -n airwise python=3.10
conda activate airwise
python -m pip install --upgrade pip
```

`xarray` is pinned to 2025.6.1 in `pyproject.toml`. Other dependencies declare
a minimum version, so pip can select a build for the current platform.

## Reporting

Install PyTorch from the CPU package index first, without a version pin. Then
install the report dependencies. The second command reuses the installed
PyTorch build:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[report]"
```

Daily reporting does not need TensorBoard, Dask, Zarr, a CUDA toolkit, or a
CUDA-enabled GPU.

## Reporting credentials

Installing the `report` profile does not configure remote data access. The
standard `airwise-daily` workflow starts by downloading a CAMS Europe forecast,
so first accept the dataset licence and save the Atmosphere Data Store token in
`$HOME/.cdsapirc`:

```text
url: https://ads.atmosphere.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

See [CAMS credentials](downloading.md#cams-credentials) for the account,
licence, and token setup. OpenIFS and CAMS Policy do not require additional
credentials with the default download configuration.

## Training

Training dependencies are isolated in the `train` extra:

```bash
python -m pip install -e ".[train]"
```

This installs TensorBoard, Dask, and Zarr, and reuses the PyTorch build from
the reporting step. CPU-only training is supported but may be slow. If this
environment does not already contain PyTorch, run the CPU install command in
[Reporting](#reporting) first.

## Combined and test environments

Install both profiles when one environment must generate reports and retrain
models:

```bash
python -m pip install -e ".[report,train]"
```

Add the test dependencies for repository development:

```bash
python -m pip install -e ".[report,train,test]"
```

## Next steps

- [Generate a daily report](quickstart.md)
- [Prepare data and train a model](training.md)
- [Configure paths and model files](configuration.md)

