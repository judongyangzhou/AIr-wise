# Installation profiles

This page is for users choosing an environment. For the commands to run after
installation, continue to the [daily-report quick start](quickstart.md) or the
[model training guide](training.md).

AIr-wise has two independent dependency profiles:

- `report` downloads daily inputs, runs the bundled pretrained models, and
  writes JSON/PDF reports.
- `train` prepares reusable Zarr stores and retrains the neural models.

## Common setup

AIr-wise uses a validated Python 3.10 dependency set. From a repository
checkout, create and activate the Conda environment:

```bash
conda create -n airwise python=3.10
conda activate airwise
python -m pip install "pip==26.2.1"
```

Direct dependencies are pinned in `pyproject.toml`.
`constraints/py310.txt` additionally pins their transitive dependencies to the
versions tested together on Linux. Use that constraints file in every project
installation command below.

## Reporting

Install the report dependencies. PyTorch is not pinned, so pip selects a build
for the current platform:

```bash
python -m pip install \
  -c constraints/py310.txt \
  -e ".[report]"
```

The Python running this command must be able to install a current PyTorch
wheel. An Intel Mac, or an Apple Silicon Mac using an `x86_64` Conda
environment, only receives PyTorch 2.2.2. That release does not support the
pinned NumPy 2.2.6, so the installation stops. Use a native `arm64` environment
on Apple Silicon.

Daily reporting does not need TensorBoard, Dask, Zarr, a CUDA toolkit, or a
CUDA-enabled GPU. On Linux, the default PyPI build may still include CUDA
libraries. To avoid that, install a CPU build before the command above:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

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
python -m pip install \
  -c constraints/py310.txt \
  -e ".[train]"
```

This installs PyTorch, TensorBoard, Dask, and Zarr. An already installed
PyTorch build is reused. CPU-only training is supported but may be slow.

## Combined and test environments

Install both profiles when one environment must generate reports and retrain
models:

```bash
python -m pip install \
  -c constraints/py310.txt \
  -e ".[report,train]"
```

Add the test dependencies for repository development:

```bash
python -m pip install \
  -c constraints/py310.txt \
  -e ".[report,train,test]"
```

## Next steps

- [Generate a daily report](quickstart.md)
- [Prepare data and train a model](training.md)
- [Configure paths and model files](configuration.md)

