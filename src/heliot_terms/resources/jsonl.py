from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> None:
    """Write Pydantic records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(record.model_dump_json(exclude_none=True))
            file.write("\n")


def write_json(path: str | Path, record: BaseModel) -> None:
    """Write a Pydantic model to a pretty JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(record.model_dump(mode="json", exclude_none=True), file, ensure_ascii=False, indent=2)