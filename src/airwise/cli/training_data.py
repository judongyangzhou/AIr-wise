"""CLI adapters for training-data preparation and model training."""

from __future__ import annotations

from collections.abc import Sequence


def prepare_main(argv: Sequence[str] | None = None) -> int:
    from airwise.pipelines.prepare_training_data import main

    main(argv)
    return 0


def regrid_main(argv: Sequence[str] | None = None) -> int:
    from airwise.data.preprocessing.regridding import main

    main(argv)
    return 0


def train_main() -> int:
    from airwise.pipelines.train_model import main

    main()
    return 0
