from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write(path: Path, items: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")
            n += 1
    return n


def read(path: Path, model: type[T]) -> Iterator[T]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield model.model_validate_json(line)


def read_all(path: Path, model: type[T]) -> list[T]:
    return list(read(path, model))


def dump_json(path: Path, obj: BaseModel | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    path.write_text(json.dumps(data, indent=2))
