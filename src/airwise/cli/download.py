"""Compatibility CLIs for downloading individual input products."""

from __future__ import annotations

from collections.abc import Sequence


def openifs_main(argv: Sequence[str] | None = None):
    from airwise.data.download.openifs import main

    return main(argv)


def cams_forecast_main(argv: list[str] | None = None):
    from airwise.data.download.cams_forecast import main

    return main(argv)


def cams_archive_main(argv: list[str] | None = None):
    from airwise.data.download.cams import main

    return main(argv)


def policy_main(argv: list[str] | None = None):
    from airwise.data.download.policy import main

    return main(argv)


def era5_main(argv: list[str] | None = None):
    from airwise.data.download.era5 import main

    return main(argv)
