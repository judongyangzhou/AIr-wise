# Model training guide

This guide is for model developers who want to rebuild the CAMS forecast-error
models. Training is separate from daily reporting and requires historical CAMS
analysis/forecast data plus ERA5 meteorology.

The examples below use 2023–2025 as the training and validation period, matching
`error_modelling.train_val_years` in `configs/default.yaml`.

## 1. Install the training environment

Follow the [training installation profile](installation.md#training). GPU
training is recommended for the larger architectures, but CUDA is optional.

Historical downloads are large. Before downloading, review and, if necessary,
override the configured data directories:

- `paths.cams_data_raw`
- `paths.cams_data_processed`
- `paths.era5_data_raw`
- `paths.era5_data_cams_grid`
- `paths.training_zarr`
- `paths.checkpoints`
- `paths.tensorboard`

See [Configuration](configuration.md) for the override rules.

## 2. Download historical CAMS data

Configure Atmosphere Data Store access first, then download monthly CAMS Europe
analysis and forecast files:

```bash
airwise-download-cams-archive --start 2023-01 --end 2025-12
```

Each month produces one analysis file and one 00Z forecast file under
`paths.cams_data_raw/analysis` and `paths.cams_data_raw/forecast`. Existing
files are skipped; use `--overwrite` to replace them.

## 3. Preprocess CAMS by year

The training cache reads yearly, preprocessed CAMS files. There is currently no
dedicated console command for this stage, so call the package function from the
repository environment:

```bash
python - <<'PY'
from airwise.modelling.features import preprocess_cams_years

years = [2023, 2024, 2025]
for data_type in ("analysis", "forecast"):
    preprocess_cams_years(years, data_type=data_type)
PY
```

The default recipe writes hourly, untransformed NetCDF files under
`paths.cams_data_processed`. Pollutant transforms are applied later when the
training data are loaded.

## 4. Download and regrid ERA5

Download every configured ERA5 variable for each year:

```bash
for year in 2023 2024 2025; do
  airwise-download-era5 --year "${year}" --all
done
```

ERA5 is downloaded on its native grid under `paths.era5_data_raw`. Resample it
onto the CAMS Europe grid:

```bash
airwise-regrid-era5 --years 2023 2024 2025
```

By default, the target grid comes from the CAMS forecast file named by
`era5_euro.regrid.reference_cams` in `configs/default.yaml`. Use
`--reference-cams PATH` if that file is not available. Regridded fields are
written under `paths.era5_data_cams_grid`.

The default training recipe derives relative humidity from `t2m` and `d2m` and
uses the ERA5 variables configured in `error_modelling.era5.variables`.

## 5. Build the training Zarr stores

Combine the processed CAMS data and regridded ERA5 fields into aligned,
chunked stores:

```bash
airwise-build-training-zarr --years 2023 2024 2025
```

The command writes CAMS forecast, CAMS analysis, ERA5, and metadata products
under `paths.training_zarr`. It stores physical CAMS concentrations; configured
pollutant transforms and z-score statistics are applied by the training
pipeline.

Useful variants include:

```bash
# Replace all selected stores.
airwise-build-training-zarr --years 2023 2024 2025 --overwrite

# Rebuild ERA5 only while reusing existing CAMS stores for time alignment.
airwise-build-training-zarr \
  --years 2023 2024 2025 \
  --stores era5 \
  --overwrite

# Build a separate cache.
airwise-build-training-zarr \
  --years 2026 \
  --output-root data/training_zarr_2026
```

## 6. Train a model

Run a basic PM10 experiment from the Zarr cache:

```bash
airwise-train-model \
  --data-source zarr \
  --pollutant pm10_conc \
  --epochs 10 \
  --device cuda \
  --run-name pm10-baseline
```

Use `--device cpu` when CUDA is unavailable. The default model is
`pointwise_temporal`; select another implemented architecture with `--model`.
Checkpoints are written below `paths.checkpoints`, and TensorBoard event files
below `paths.tensorboard`.

To resume the model and optimizer state:

```bash
airwise-train-model \
  --data-source zarr \
  --pollutant pm10_conc \
  --device cuda \
  --resume-from models/checkpoints/PATH_TO_CHECKPOINT.pt
```

Inspect the available options before defining a production experiment:

```bash
airwise-train-model --help
```

## 7. Monitor training

Unless `--disable-tensorboard` is passed, start TensorBoard with:

```bash
tensorboard --logdir logs/tensorboard
```

If `paths.tensorboard` has been overridden, pass that directory instead.

## Data flow

```text
CAMS monthly NetCDF ──> yearly processed CAMS ──┐
                                                ├─> aligned Zarr ──> training
ERA5 yearly NetCDF ──> CAMS-grid ERA5 ──────────┘
```

## Next steps

- [Configure data, checkpoints, and TensorBoard paths](configuration.md)
- [Review data-source credentials and download commands](downloading.md)
- [Understand the training subsystem boundaries](architecture.md#training-boundary)
