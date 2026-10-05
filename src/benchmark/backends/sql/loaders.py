from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.utils.io import load_json
from benchmark.constants import BIRD_DOMAIN_DBS


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


def _copy_with_source_db(templates: list[dict[str, Any]], source_db: str) -> list[dict[str, Any]]:
    out = []
    for t in templates:
        row = dict(t)
        row["source_db"] = source_db
        row["template_id"] = f"{source_db}_{t.get('template_id')}"
        out.append(row)
    return out


def load_distribution_templates(
    distribution_file: Path | str,
    benchmark_type: str,
    split: str,
    target_db: str,
) -> list[dict[str, Any]]:
    data = load_json(_require(distribution_file))

    if split == "Combined" and benchmark_type == "BIRD" and target_db in BIRD_DOMAIN_DBS:
        templates: list[dict[str, Any]] = []
        for sub_split in ["Train", "Dev"]:
            for db in BIRD_DOMAIN_DBS[target_db].get(sub_split, []):
                rows = _safe_get(data, "BIRD", sub_split, db, "templates")
                if rows:
                    with_source = _copy_with_source_db(rows, db)
                    for row in with_source:
                        row["source_split"] = sub_split
                    templates.extend(with_source)
        return templates

    if split == "Combined":
        templates: list[dict[str, Any]] = []
        for sub_split in ["Train", "Dev"]:
            rows = _safe_get(data, benchmark_type, sub_split, target_db, "templates")
            if rows:
                for t in rows:
                    row = dict(t)
                    row["source_split"] = sub_split
                    templates.append(row)
        return templates

    if benchmark_type == "BIRD" and target_db in BIRD_DOMAIN_DBS:
        dbs = BIRD_DOMAIN_DBS[target_db].get(split, [])
        templates = []
        for db in dbs:
            rows = _safe_get(data, "BIRD", split, db, "templates")
            if rows:
                templates.extend(_copy_with_source_db(rows, db))
        return templates

    rows = _safe_get(data, benchmark_type, split, target_db, "templates")
    if rows is None:
        raise ValueError(
            f"Cannot find templates for benchmark={benchmark_type}, split={split}, db={target_db}"
        )
    return rows


def _merge_literal_sources(merged: dict[str, Any], incoming: dict[str, Any]) -> None:
    for column, info in incoming.items():
        merged.setdefault(column, {"existing_values": []})
        current = set(merged[column].get("existing_values", []))
        current.update(info.get("existing_values", []))
        merged[column]["existing_values"] = sorted(current)


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

    if benchmark_type == "BIRD" and target_db in BIRD_DOMAIN_DBS and split != "Combined":
        merged: dict[str, Any] = {}
        for db in BIRD_DOMAIN_DBS[target_db].get(split, []):
            row = _safe_get(data, "BIRD", split, db)
            if row:
                _merge_literal_sources(merged, row)
        return merged or None

    if benchmark_type == "BIRD" and target_db in BIRD_DOMAIN_DBS and split == "Combined":
        merged: dict[str, Any] = {}
        for sub_split in ["Train", "Dev"]:
            for db in BIRD_DOMAIN_DBS[target_db].get(sub_split, []):
                row = _safe_get(data, "BIRD", sub_split, db)
                if row:
                    _merge_literal_sources(merged, row)
        return merged or None

    if split == "Combined":
        merged = {}
        for sub_split in ["Train", "Dev"]:
            row = _safe_get(data, benchmark_type, sub_split, target_db)
            if row:
                _merge_literal_sources(merged, row)
        return merged or None

    row = _safe_get(data, benchmark_type, split, target_db)
    return row if isinstance(row, dict) else None
