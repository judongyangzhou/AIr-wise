"""Serialize and validate daily bulletin JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from airwise.domain.bulletin import parse_report


def write_report_json(
    output_path: str | Path,
    report_data: dict[str, Any],
    *,
    validate: bool = True,
) -> Path:
    if validate:
        parse_report(report_data)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(output)
    return output
