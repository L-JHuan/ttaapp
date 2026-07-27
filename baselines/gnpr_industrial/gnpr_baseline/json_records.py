from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def _data_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = [
        item
        for item in path.rglob("*")
        if item.is_file() and not item.name.startswith(("_", "."))
    ]
    if not files:
        raise ValueError(f"{path} 不包含数据文件")
    return sorted(files)


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    """读取 JSON 对象列表、JSONL 文件或 Spark JSON 输出目录。"""
    for data_file in _data_files(path):
        with data_file.open("r", encoding="utf-8") as source:
            first_character = ""
            while True:
                character = source.read(1)
                if not character:
                    break
                if not character.isspace():
                    first_character = character
                    break
            if not first_character:
                continue
            source.seek(0)
            if first_character == "[":
                value = json.load(source)
                if not isinstance(value, list):
                    raise ValueError(f"{data_file} 必须是 JSON 对象列表")
                for row in value:
                    if not isinstance(row, dict):
                        raise ValueError(f"{data_file} 包含非对象记录")
                    yield row
                continue
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{data_file}:{line_number} 包含非对象记录")
                yield row


def load_json_records(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_json_records(path):
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    return rows
