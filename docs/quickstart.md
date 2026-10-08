# Daily-report quick start

This guide is for users who want to download one day's inputs and generate the
published air-quality bulletin. The production workflow currently supports
Germany only.

## Prerequisites

1. Work from a repository checkout with Python 3.10 or newer.
2. Install the `report` profile. The
   [installation guide](installation.md#reporting) shows the recommended
   CPU-only setup.
3. Configure Atmosphere Data Store access for the CAMS download as described in
   [Downloading data](downloading.md#cams-credentials).
4. Keep the bundled model checkpoints, statistics, NUTS geometry, and land-sea
   mask at their default paths, or override their locations as described in
   [Configuration](configuration.md).

The operational IFS download and CAMS Policy download also require network
access, but not additional credentials in the default configuration.

## Generate one report

From the repository root, run:

```bash
airwise-daily --date 2026-09-29 --device cpu
```

Both `YYYY-MM-DD` and `YYYYMMDD` date formats are accepted. The command:

1. downloads the CAMS Europe forecast, 00 UTC operational IFS forecast, and CAMS
   Policy city forecasts;
2. computes the daily European AQI product;
3. runs the bundled neural models to estimate AQI confidence; and
4. writes the bulletin as JSON and PDF.

With the default configuration, the outputs are:

```text
reports/germany_20260929.json
reports/germany_20260929.pdf
```

Use `--output` to choose another JSON path. The PDF is written beside it with
the same filename stem:

```bash
airwise-daily \
  --date 2026-09-29 \
  --device cpu \
  --output reports/custom-name.json
```

Existing downloads and AQI products are reused. Existing operational IFS files
are validated before reuse. Pass `--overwrite` to re-download all remote inputs,
atomically replace the operational IFS files, and recompute AQI.

## Run the steps separately

The individual commands make it easier to identify a failed stage:

```bash
REPORT_DATE=2026-09-29

airwise-acquire-daily-inputs --date "${REPORT_DATE}" --country Germany
airwise-compute-aqi --date "${REPORT_DATE}"
airwise-compute-confidence --date "${REPORT_DATE}" --device cpu
airwise-report \
  --date "${REPORT_DATE}" \
  --region-config configs/regions/germany.yaml \
  --output "reports/germany_${REPORT_DATE//-/}.json" \
  --device cpu
```

`airwise-report` writes the JSON path given by `--output` and a PDF with the
same stem. Use `--pdf-output` to place the PDF elsewhere. If no confidence
product exists, the report command can run the same pretrained-model inference
itself; running `airwise-compute-confidence` explicitly makes that stage and
its errors visible.

## Current scope

- The daily bulletin supports Germany and rejects other `--country` values.
- `configs/regions/germany.yaml` selects and labels the NUTS regions and
  representative cities.
- `data/shapefile/NUTS_RG_10M_2024_3035.gpkg` supplies their polygon geometry.
- Production uncertainty inference uses the operational high-resolution IFS
  forecast (`oper`/`fc`, 00 UTC), not a 51-member ensemble.

## Next steps

- [Understand the download sources and credentials](downloading.md)
- [Move data or report outputs to other directories](configuration.md)
- [Understand the production pipeline](architecture.md#daily-production-flow)
