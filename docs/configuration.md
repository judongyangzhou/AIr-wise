# Configuration

This page is for users who need to move data, select download settings, or
change model and report inputs. The repository defaults work without creating
a machine-specific configuration file.

## Configuration loading

The base configuration is selected in this order:

1. an explicit path passed by a command that supports `--config`;
2. the path in the `AIRWISE_CONFIG` environment variable; or
3. `configs/default.yaml`.

After loading the base file, AIr-wise also loads `local.yaml` from the same
directory when it exists. Mappings are merged recursively; scalar values and
lists in `local.yaml` replace the corresponding base values.

Relative paths are resolved from the repository root. Absolute paths are used
as written.

## Machine-specific overrides

`configs/default.yaml` uses portable paths inside the repository. Create a
local override only when this machine should use different storage:

```bash
cp configs/local.example.yaml configs/local.yaml
```

Edit the copied file and replace the `/mnt/data-disk/airwise` placeholders.
`configs/local.yaml` is ignored by Git. Keys omitted from it retain their
values from `configs/default.yaml`, so the file only needs to contain actual
overrides.

For example:

```yaml
paths:
  cams_data_raw: "/data/airwise/cams/raw"
  era5_data_raw: "/data/airwise/era5/raw"
  training_zarr: "/data/airwise/training_zarr"

open_ifs:
  client_source: ecmwf
```

Do not store CDS credentials or other secrets in project YAML files.

## Using another base configuration

Commands that expose `--config`, including ERA5 regridding and training-store
preparation, can load another base YAML directly:

```bash
airwise-build-training-zarr --config /path/to/experiment/default.yaml
```

For commands without a `--config` option, set the environment variable:

```bash
export AIRWISE_CONFIG=/path/to/experiment/default.yaml
airwise-train-model --data-source zarr --device cuda
```

If `/path/to/experiment/local.yaml` exists, it is merged into that base file
using the same rules.

## Path settings

The `paths` section separates downloaded inputs, generated products, models,
and presentation outputs.

Daily reporting primarily uses:

- `cams_forecast_daily`: downloaded daily CAMS forecasts;
- `cams_aqi_daily`: computed daily AQI products;
- `cams_policy_forecast`: Policy city forecast JSON;
- `open_ifs_data`: operational IFS forecast products from ECMWF Open Data;
- `bulletin_uq`: daily uncertainty/confidence products;
- `nuts_shapefile`: administrative polygon geometry;
- `land_sea_mask`: the mask used by confidence inference and training; and
- `checkpoints`: trained model weights.

The configuration also defines `paths.reports`, but the current
`airwise-daily` default is the repository's `reports/` directory. Use
`airwise-daily --output PATH` to select another report location.

Model training additionally uses:

- `cams_data_raw` and `cams_data_processed`;
- `era5_data_raw` and `era5_data_cams_grid`;
- `training_zarr`; and
- `tensorboard`.

The repository already includes the report checkpoints, model statistics,
Germany NUTS geometry, and land-sea mask at the default paths. When overriding
these paths, make sure the replacement files are present.

## Download and model settings

- `cams_europe` defines the Copernicus dataset, downloaded pollutant fields,
  coordinate names, and model-training variable names.
- `era5_euro` defines ERA5 fields, the Europe crop, and the CAMS-grid
  regridding methods.
- `open_ifs.channel` and `open_ifs.client_source` select how the operational
  IFS forecast is downloaded. The `open_ifs` key is a local configuration name.
- `error_modelling` defines the default pollutant, training/test years,
  temporal resolution, transforms, meteorological fields, and data backend.
- `reporting.confidence` maps report pollutants to their checkpoint and
  statistics files.
- `aqi` selects the AQI standard and optional threshold definition override.

CLI arguments take precedence over the corresponding defaults for that
invocation.

## Region configuration

Report regions are deployment metadata and are not part of the merged
application configuration. The current file,
`configs/regions/germany.yaml`, defines:

- the report country and country code;
- NUTS level and IDs;
- stable region IDs and display names; and
- representative cities and coordinates.

The region file selects and labels areas; `paths.nuts_shapefile` supplies the
polygon geometry for those NUTS IDs. Both are required for a report.

Pass another region file to commands that expose `--region-config`, but note
that the production daily command currently accepts Germany only.

## Installed resources

Definitions that are part of the application rather than a deployment are
installed inside the package:

- the European AQI definition;
- report overview templates; and
- the daily bulletin JSON schema.

Override these only through the dedicated command options, such as
`--aqi-config`; do not copy them into `configs/local.yaml`.

## Next steps

- [Install the required dependency profile](installation.md)
- [Review remote data sources](downloading.md)
- [Understand configuration boundaries](architecture.md#configuration-and-resources)
