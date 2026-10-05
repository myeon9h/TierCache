from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.utils.io import load_json


def _require(path: Path | str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _safe_get(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_distribution_templates(
    distribution_file: Path | str,
    benchmark_type: str,
    split: str,
    target_db: str,
) -> list[dict[str, Any]]:
    data = load_json(_require(distribution_file))

    if split == "Combined":
        templates: list[dict[str, Any]] = []
        for sub_split in ["Train", "Dev", "Test"]:
            rows = _safe_get(data, benchmark_type, sub_split, target_db, "templates")
            if rows:
                for template in rows:
                    row = dict(template)
                    row["source_split"] = sub_split
                    templates.append(row)
        return templates

    rows = _safe_get(data, benchmark_type, split, target_db, "templates")
    if rows is None:
        raise ValueError(
            f"Cannot find templates for benchmark={benchmark_type}, split={split}, db={target_db}"
        )
    return rows


def _merge_literal_sources(merged: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, info in incoming.items():
        merged.setdefault(key, {"existing_values": []})
        current = set(merged[key].get("existing_values", []))
        current.update(info.get("existing_values", []))
        merged[key]["existing_values"] = sorted(current)


def load_literals_data(
    literals_file: Path | str,
    benchmark_type: str,
    split: str,
    target_db: str,
) -> dict[str, Any] | None:
    path = Path(literals_file)
    if not path.exists():
        return None
    data = load_json(path)

    if split == "Combined":
        merged: dict[str, Any] = {}
        for sub_split in ["Train", "Dev", "Test"]:
            row = _safe_get(data, benchmark_type, sub_split, target_db)
            if row:
                _merge_literal_sources(merged, row)
        return merged or None

    row = _safe_get(data, benchmark_type, split, target_db)
    return row if isinstance(row, dict) else None
