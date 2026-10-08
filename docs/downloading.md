# Downloading data

This page describes the remote data sources used by daily reporting and model
training. Output directories come from the project configuration; see
[Configuration](configuration.md) before downloading large archives.

## CAMS credentials

AIr-wise uses `cdsapi` for:

- CAMS Europe daily air-quality forecasts;
- historical CAMS Europe analysis and forecasts.

These downloads use the Copernicus Atmosphere Data Store dataset
[`cams-europe-air-quality-forecasts`](https://ads.atmosphere.copernicus.eu/datasets/cams-europe-air-quality-forecasts).
Configure access as follows:

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

`cdsapi.Client()` reads `~/.cdsapirc`. Keep this credential file outside the
repository. A Climate Data Store token for ERA5 is a different key and does not
authorise the CAMS Atmosphere Data Store dataset.

## Download all daily-report inputs

The normal reporting workflow downloads all three daily sources in one step:

```bash
airwise-acquire-daily-inputs --date 2026-09-29 --country Germany
```

This downloads:

1. the CAMS Europe air-quality forecast through `cdsapi`;
2. the 00 UTC operational high-resolution IFS forecast (`stream=oper`,
   `type=fc`) through the configured Open Data channel;
   and
3. CAMS Policy source-receptor forecasts for German cities.

The command currently accepts Germany only. Existing files are reused; pass
`--overwrite` to replace them.

### CAMS Europe

Download one daily forecast:

```bash
airwise-download-cams-forecast --date 2026-09-29
```

The daily product is the 00:00 UTC surface ensemble forecast at lead hours
0–23 for nitrogen dioxide, ozone, PM2.5, PM10, and sulphur dioxide. The file is
written under `paths.cams_forecast_daily`. If `--date` is omitted, the command
uses the current UTC date.

For model training, download monthly analysis and 00Z forecast files over an
inclusive range:

```bash
airwise-download-cams-archive --start 2023-01 --end 2025-12
```

The archive command writes NetCDF files under
`paths.cams_data_raw/analysis` and `paths.cams_data_raw/forecast`. Use
`--out-dir` to override the configured root for one invocation.

### ERA5

ERA5 uses the
[Climate Data Store API](https://cds.climate.copernicus.eu/how-to-api), not the
Atmosphere Data Store token described above. Accept the ERA5 dataset terms and
place the CDS URL and token in `~/.cdsapirc` before running an ERA5 download.
Users who download both CAMS and ERA5 must switch that file to the credentials
for the service currently being queried.

Download one configured ERA5 field for a calendar year:

```bash
airwise-download-era5 --year 2024 --variable t2m
```

Download every field listed in `era5_euro.variables`:

```bash
airwise-download-era5 --year 2024 --all
```

Use `--months 1 2 3` to request only selected months. Raw files are written
under `paths.era5_data_raw`. Regridding is a separate local operation:

```bash
airwise-regrid-era5 --years 2023 2024 2025
```

See the [training guide](training.md#4-download-and-regrid-era5) for the full
preparation sequence.

### Operational IFS forecast

Production reporting uses the ECMWF operational high-resolution IFS forecast
from the Open Data service. Each request is the 00 UTC run at 0.25°
(`model=ifs`, `stream=oper`, `type=fc`). This is not an OpenIFS experiment.
Downloaded fields are stored locally with `number=0`:

```bash
airwise-download-open-ifs --date 2026-09-29
```

The default configuration uses the `client` channel with the Google mirror
through `ecmwf-opendata`. No API credential is required for this public Open
Data client. The resulting NetCDF files are cropped to Europe and written under
`paths.open_ifs_data`.

Available channels are:

- `client`: use `ecmwf-opendata`; select `google`, `aws`, `azure`, or `ecmwf`
  with `--client-source`.
- `https`: download with `curl`, or `wget` when `curl` is absent, from
  `data.ecmwf.int`; this route can be slower or rate limited.
- `gsutil`: download from the public `ecmwf-open-data` Google Cloud Storage
  bucket; the `gsutil` executable must be on `PATH`.

Set the normal default in `open_ifs.channel` and
`open_ifs.client_source`, or override it for one command:

```bash
airwise-download-open-ifs \
  --date 2026-09-29 \
  --channel client \
  --client-source ecmwf
```

Use `--dry-run` to inspect an operational IFS request without downloading it.

Existing operational IFS NetCDF files are validated before reuse. Valid files
are kept;
an unreadable or incomplete file stops acquisition and prints the recovery
command. To force a fresh download and atomically replace every lead for one
date, run:

```bash
airwise-download-open-ifs --date 2026-09-29 --overwrite
```

## CAMS Policy city forecasts

Download the public source-receptor forecast files for German cities:

```bash
airwise-download-policy-forecast \
  --date 2026-09-29 \
  --country Germany
```

The files are written under `paths.cams_policy_forecast`. The standalone
downloader can query other country names or two-letter codes exposed by the
Policy API, but the published AIr-wise bulletin currently has a Germany-only
region configuration. No API token is required. The default request uses the
TNO inventory, and bulletin generation fails rather than silently omitting the
transboundary table when these files are unavailable.

## Reusing and replacing downloads

Download commands skip files that already exist. Pass `--overwrite` only when
the remote product must be fetched again. Downloaded and generated datasets are
ignored by Git; the repository tracks only the small static assets needed by
the report.

## Next steps

- [Generate the daily bulletin](quickstart.md)
- [Prepare the historical training dataset](training.md)
- [Change download and output directories](configuration.md)
