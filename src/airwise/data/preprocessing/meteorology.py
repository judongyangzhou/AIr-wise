"""Model-independent meteorological derived variables."""

from __future__ import annotations

import numpy as np
import xarray as xr


_MAGNUS_A = 17.625
_MAGNUS_B = 243.04


def relative_humidity_from_dewpoint(
    t2m: xr.DataArray | np.ndarray,
    d2m: xr.DataArray | np.ndarray,
) -> xr.DataArray | np.ndarray:
    """Compute relative humidity (%) from temperature and dewpoint in Kelvin."""
    temperature_c = t2m - 273.15
    dewpoint_c = d2m - 273.15
    humidity = 100.0 * np.exp(
        _MAGNUS_A
        * (
            dewpoint_c / (_MAGNUS_B + dewpoint_c)
            - temperature_c / (_MAGNUS_B + temperature_c)
        )
    )
    if isinstance(humidity, xr.DataArray):
        return humidity.clip(min=0.0, max=100.0)
    return np.clip(humidity, 0.0, 100.0)
