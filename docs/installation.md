# Installation profiles

AIr-wise has two independent user workflows:

1. **Daily reporting** downloads data for a date, runs the bundled pretrained
  models for uncertainty inference, and writes JSON/PDF reports.
2. **Model training** rebuilds the neural models and is optional.

Daily reporting does not require model training, TensorBoard, Dask, Zarr, or
CUDA. Every published report requires neural confidence inference, which uses
PyTorch but can run entirely on CPU. Report generation fails instead of writing
null confidence values when the model or its inputs are unavailable.

## Common setup

Python 3.9 or newer is required. From a repository checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
cp configs/local.example.yaml configs/local.yaml
```

`configs/local.example.yaml` shows placeholder paths on a separate data disk.
Replace `/mnt/data-disk/airwise` with this machine's directories before use.
Keys left out of that file, including the shapefile, land-sea mask, and model
checkpoints, stay on the paths in `configs/default.yaml`. Configure CDS API
credentials before downloading CAMS data.

## Daily reports on CPU (recommended for report users)

Install the official CPU-only PyTorch wheel first, then install the report
dependencies:

```bash
python -m pip install torch \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[report]"
```

Installing PyTorch first is intentional: the project can declare a PyTorch
version requirement, but Python package metadata cannot select PyTorch's
CPU-only package index. Once the CPU wheel is installed, the second command
reuses it. No CUDA toolkit or CUDA-enabled GPU is required.

Run one date from acquisition through PDF generation:

```bash
airwise-daily --date 20240430 --device cpu
```

That writes `reports/germany_20240430.json` and `reports/germany_20240430.pdf`.
The same work can still be run as four separate commands:

```bash
REPORT_DATE=20240430

airwise-acquire-daily-inputs --date "${REPORT_DATE}"
airwise-compute-aqi --date "${REPORT_DATE}"
airwise-compute-confidence --date "${REPORT_DATE}" --device cpu
airwise-report \
  --date "${REPORT_DATE}" \
  --region-config configs/regions/germany.yaml \
  --output "reports/germany_${REPORT_DATE}.json" \
  --device cpu
```

`configs/regions/germany.yaml` defines which NUTS regions appear in the report,
their ordering, display names, and representative cities. Their polygon
geometry is loaded separately from the configured
`data/shapefile/NUTS_RG_10M_2024_3035.gpkg`.

`airwise-report` writes the bulletin JSON to `--output` and the PDF to the
same path with a `.pdf` suffix. Use `--pdf-output` when the PDF should go
elsewhere. It reuses an existing confidence product. If one is absent,
it can run the same pretrained-model inference itself. Running
`airwise-compute-confidence` explicitly makes that step and any failure easier
to inspect.

## Retraining models

Training dependencies are isolated in the `train` extra:

```bash
python -m pip install -e ".[train]"
```

This installs PyTorch, TensorBoard, Dask, and Zarr. CUDA remains optional.
CPU-only training is supported but may be slow; to guarantee a CPU-only
environment, install the CPU PyTorch wheel first as shown above.

For GPU training, install the PyTorch build recommended for the machine's
driver and platform, then install the project:

```bash
# First install the appropriate PyTorch build from pytorch.org.
python -m pip install -e ".[train]"
```

The repository does not pin a CUDA toolkit because the correct PyTorch/CUDA
combination depends on the host GPU and driver.

Users who need both daily reporting and retraining can install both profiles:

```bash
python -m pip install -e ".[report,train]"
```

