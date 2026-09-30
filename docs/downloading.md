# Downloading data

`airwise-daily` downloads the three inputs and then continues through AQI,
confidence, and the PDF. To download only, use one command for all three
products or one command for each product. Existing files are skipped unless
`--overwrite` is set. Report rendering does not download data.

Download CAMS, the OpenIFS 00z control forecast, and CAMS Policy city
forecasts for one date:

```bash
airwise-acquire-daily-inputs --date 20260929
```

Download one product at a time:

```bash
airwise-download-cams-forecast --date 20260929
airwise-download-open-ifs --date 20260929
airwise-download-policy-forecast --date 20260929
```

`--date` accepts `YYYYMMDD` or `YYYY-MM-DD`. `airwise-download-open-ifs` also
accepts `--channel` for that one run. See the sections below for credentials,
OpenIFS channels, and the Policy city list.

## CAMS forecast

The daily forecast is the Copernicus Atmosphere Data Store dataset
[`cams-europe-air-quality-forecasts`](https://ads.atmosphere.copernicus.eu/datasets/cams-europe-air-quality-forecasts):
the 00:00 UTC ensemble, surface level, lead hours 0–23, for nitrogen dioxide,
ozone, PM2.5, PM10, and sulphur dioxide. The file is written to
`paths.cams_forecast_daily`.

Downloads use the [cdsapi](https://github.com/ecmwf/cdsapi) client. It is
installed with the project (`pip install -e ".[report]"`). Credentials are
separate and come from the Atmosphere Data Store:

1. Register and log in at the
   [Atmosphere Data Store](https://ads.atmosphere.copernicus.eu/).
2. Open [How to use the ADS API](https://ads.atmosphere.copernicus.eu/how-to-api)
   and copy the personal access token shown there.
3. Accept the dataset licence on the CAMS Europe forecast download form.
   The API refuses the request until that licence has been accepted in the
   browser.
4. Save the token in `$HOME/.cdsapirc`:

```text
url: https://ads.atmosphere.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

`cdsapi.Client()` reads `~/.cdsapirc`. The URL above must be the Atmosphere
Data Store. A Climate Data Store token
([CDS API setup](https://cds.climate.copernicus.eu/how-to-api)) is a different
key and does not authorise this CAMS dataset.

## OpenIFS control forecast

The daily report uses the IFS 00z deterministic control forecast (`oper/fc`,
member 0), cropped to Europe. It is not the 51-member ensemble. The same GRIB
fields can be fetched in three ways. `open_ifs.channel` and
`open_ifs.client_source` in `configs/default.yaml` are the repository
defaults. `configs/local.yaml` overrides them for one machine.
`airwise-download-open-ifs --channel` overrides them for one command.

| `channel` | What it uses | `client_source` |
| --- | --- | --- |
| `client` | the [ecmwf-opendata](https://confluence.ecmwf.int/spaces/OIFS/pages/19661477/OpenIFS+Home) Python client, installed with `.[report]` | `google`, `aws`, `azure`, or `ecmwf` |
| `https` | `curl`, or `wget` if `curl` is absent, from <https://data.ecmwf.int/forecasts> | ignored |
| `gsutil` | the public `ecmwf-open-data` bucket; `gsutil` must be on `PATH` | ignored |

`client_source` selects the mirror only when `channel` is `client`.

`channel: https` is the simplest choice. It uses `curl` or `wget` and does
not need the `ecmwf-opendata` package. Downloads come from `data.ecmwf.int`,
so they can be slower and can be rate limited. Put this in `configs/local.yaml`
to use it for daily downloads, for example:

```yaml
open_ifs:
  channel: https
```

`configs/default.yaml` still defaults to `channel: client` and
`client_source: google`. That Google mirror avoids the `data.ecmwf.int`
request limit, but it needs `ecmwf-opendata`. `aws` and `azure` are the other
mirrors for `channel: client`. `configs/local.example.yaml` shows
`client_source: ecmwf`, which talks to the ECMWF server and can return HTTP
429. `channel: gsutil` fits a machine that already has `gsutil`; it reads the
same public files.

One download without editing the config:

```bash
airwise-download-open-ifs --date 20260929 --channel https
airwise-download-open-ifs --date 20260929 --channel client --client-source aws
```

## CAMS Policy forecasts

City source-receptor forecasts come from the
[CAMS Policy website](https://policy.atmosphere.copernicus.eu/). No API token
is required. The daily commands always download them. For the requested
country (Germany by default) the download reads the public city catalogue
and each city's daily forecast JSON from that site, using the TNO inventory.
Files are written under `paths.cams_policy_forecast`. The bulletin fails
instead of omitting the transboundary table when these files are missing.
