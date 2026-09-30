# Airwise architecture

## Dependency direction

Production code follows this direction:

```text
cli -> pipelines -> domain / modelling / uncertainty / reporting
                    -> data -> config / resources
```

Lower layers must not import `cli` or `pipelines`.

## Package layout

```text
src/airwise/
  cli/              argument parsing, logging, exit codes
  pipelines/        explicit use-case orchestration
  domain/           pollutant, AQI, region, and bulletin data models
  config/           external YAML loading and typed settings
  resources/        immutable AQI, template, and schema package data
  data/
    download/       network-facing provider adapters
    io/             local NetCDF/Zarr product readers and writers
    preprocessing/  model-independent transforms and regridding
  modelling/        datasets, features, models, checkpoints, training, inference
  uncertainty/      probability, interpolation, and AQI-confidence algorithms
  reporting/        JSON/PDF presentation and visual components
```

## Configuration and resources

`configs/default.yaml`, optional `configs/local.yaml`, and
`configs/regions/*.yaml` remain outside the package because they describe a
deployment and its report regions. Pipelines load them at their boundary and
pass typed settings or explicit values inward.

Immutable application definitions are installed with the package:

- `resources/aqi/europe.yaml`
- `resources/reporting/overview_templates.yaml`
- `resources/schemas/daily_bulletin.schema.json`

## Daily production flow

1. `acquire_daily_inputs` explicitly downloads CAMS, OpenIFS control
   (`number=0`), and CAMS Policy city forecasts. The bulletin table requires
   the Policy files. Download sources are described in
   [Downloading data](downloading.md).
2. `compute_daily_forecast_aqi` reads local CAMS data and writes an AQI
   NetCDF.
3. `ensure_daily_uq_dataset` runs learned spatial-standard-deviation inference
   from the OpenIFS control forecast and writes a versioned uncertainty
   NetCDF.
4. `generate_report_json` combines local products into the bulletin contract.
5. Reporting code renders the JSON contract as PDF.

Report generation does not download data. EMOS, conformal calibration, and
51-member OpenIFS ensemble experiments are not part of this flow.

## Region definitions and geometry

Daily German reports use both `configs/regions/germany.yaml` and the configured
NUTS GeoPackage. They are not interchangeable:

- `germany.yaml` selects the country and report regions and supplies stable
  report IDs, display names, representative cities, coordinates, and NUTS IDs.
- `data/shapefile/NUTS_RG_10M_2024_3035.gpkg` supplies polygon geometry keyed
  by those NUTS IDs. The boundaries are Eurostat GISCO NUTS 2024.
  © EuroGeographics for the administrative boundaries.

The reporting pipeline rasterises the GeoPackage polygons onto the CAMS grid,
then uses the YAML metadata to label and order the resulting regional report
sections.

## Training boundary

`data/io/training_store.py` owns local Zarr access.
`modelling/training/data.py` aligns CAMS/ERA5 inputs and applies model recipes.
`modelling/training/datasets.py` owns PyTorch sample construction.
`modelling/training/losses.py` owns loss functions and AQI-aware weighting.
`modelling/training/metrics.py` owns streaming validation metrics used while training.
`modelling/training/trainer.py` owns training and validation loops.
`modelling/features.py` owns model-specific feature preparation.
`pipelines/prepare_training_data.py` and `pipelines/train_model.py` orchestrate
cache creation and training.
