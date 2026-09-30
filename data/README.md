# Bundled static data

`shapefile/NUTS_RG_10M_2024_3035.gpkg` contains the NUTS 2024 administrative
boundaries (1:10 million, EPSG:3035) used to build regional masks for daily
reports. The application resolves this file through `paths.nuts_shapefile` in
`configs/default.yaml`.

Source: Eurostat GISCO, territorial units for statistics (NUTS), 2024.
https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units

When these boundaries are used in a printed or electronic publication, acknowledge
the source in the map legend and on the introductory page:

- EN: © EuroGeographics for the administrative boundaries
- FR: © EuroGeographics pour les limites administratives
- DE: © EuroGeographics bezüglich der Verwaltungsgrenzen

GISCO provides this dataset for non-commercial use. Commercial use requires a
licence from EuroGeographics.

`land_sea_mask/land_sea_mask.nc` is the CAMS regional land-sea mask. It is
downloaded from
<https://confluence.ecmwf.int/download/attachments/202173092/land_sea_mask.nc?version=1&modificationDate=1746010579837&api=v2>.
The application resolves it through `paths.land_sea_mask` in `configs/default.yaml`.
