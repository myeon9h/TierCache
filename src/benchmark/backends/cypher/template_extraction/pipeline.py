from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from benchmark.backends.cypher.template_extraction.masking import (
    convert_literal_placeholders,
    placeholder_index,
    process_record,
    to_m2_placeholder,
)
from benchmark.utils.io import dump_json, ensure_dir, load_json_or_jsonl


from benchmark.paths import REPO_ROOT as PROJECT_ROOT


def _resolve_project_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_sources_config(config_path: Path | str) -> dict[str, Any]:
    path = _resolve_project_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing sources config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _split_prefix(split_name: str) -> str:
    if split_name.lower() == "train":
        return "train"
    if split_name.lower() == "test":
        return "test"
    return "dev"


def configured_dbs(config: dict[str, Any], db_split: str | None = None) -> list[str]:
    dbs = config.get("dbs", [])
    if isinstance(dbs, dict):
        if db_split and db_split != "all":
            return [str(db) for db in dbs.get(db_split, [])]
        out: list[str] = []
        for group in ["train", "test"]:
            out.extend(str(db) for db in dbs.get(group, []))
        return out
    if isinstance(dbs, list):
        return [str(db) for db in dbs]
    return []


def _normalize_sampling_returns(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows or []):
        item = dict(row)
        item["slot"] = to_m2_placeholder(str(item.get("slot", "")), idx)
        normalized.append(item)
    return normalized


def _normalize_literal_slot(slot: dict[str, Any], index: int) -> dict[str, Any]:
    source_slot = str(slot.get("slot", f"[LITERAL_{index}]"))
    m2_slot = to_m2_placeholder(source_slot, index)
    literal_index = placeholder_index(source_slot, index)
    value = slot.get("value", slot.get("example_value"))

    normalized = {
        "slot": m2_slot,
        "source_slot": source_slot,
        "index": literal_index,
        "value": value,
        "example_value": value,
        "type": slot.get("type", slot.get("literal_type", "string")),
        "raw_expressions": slot.get("raw_expressions", []),
        "contexts": slot.get("contexts", []),
        "sample_query": slot.get("sample_query"),
    }
    return {key: val for key, val in normalized.items() if val is not None}


def _ensure_preprocessed(
    record: dict[str, Any],
    *,
    sample_limit: int,
    ignore_case: bool,
) -> dict[str, Any]:
    if record.get("masked_cypher") is not None and record.get("literal_slots") is not None:
        return record
    return process_record(
        record,
        ignore_case=ignore_case,
        sample_limit=sample_limit,
        numbered_placeholders=True,
        include_sampling=True,
    )


def _to_structural_record(
    record: dict[str, Any],
    *,
    row_id: int,
    source: str,
) -> dict[str, Any]:
    graph = record.get("graph") or record.get("db")
    if not graph:
        raise ValueError(f"Missing graph/db field at row {row_id}")

    literal_slots = [
        _normalize_literal_slot(slot, idx)
        for idx, slot in enumerate(record.get("literal_slots", []))
    ]

    structural: dict[str, Any] = {
        "id": row_id,
        "qid": record.get("qid"),
        "source": source,
        "db": graph,
        "question": {
            "original": record.get("nl_question", record.get("question", "")),
            "semi_template_m2": convert_literal_placeholders(record.get("masked_question")),
        },
        "cypher": {
            "original": record.get("gold_cypher", record.get("cypher", "")),
            "semi_template_m2": convert_literal_placeholders(record.get("masked_cypher")),
        },
        "literals": {
            "m2": literal_slots,
        },
        "sampling": {
            "query": record.get("sampling_query"),
            "returns": _normalize_sampling_returns(record.get("sampling_returns")),
            "policy": record.get("sampling_policy", {}),
        },
        "meta": {
            "from_template": record.get("from_template"),
            "masking_policy": record.get("masking_policy", {}),
        },
    }
    if not structural["qid"]:
        structural.pop("qid")
    if structural["meta"]["from_template"] is None:
        structural["meta"].pop("from_template")
    return structural


def extract_cypherbench(
    config: dict[str, Any],
    output_dir: Path | str,
    *,
    split: str = "Dev",
    db_split: str | None = None,
    input_file: Path | str | None = None,
    selected_dbs: list[str] | None = None,
    sample_limit: int = 100,
    ignore_case: bool = False,
) -> dict[str, int]:
    output_dir = ensure_dir(output_dir)

    if input_file is None:
        preprocessed = config.get("preprocessed", {})
        raw = config.get("raw", {})
        candidates = [preprocessed.get(split), raw.get(split)]
        input_file = next((value for value in candidates if value and _resolve_project_path(value).is_file()), None)
    if input_file is None:
        raise ValueError(f"No input file configured for CypherBench split={split}")

    src = _resolve_project_path(input_file)
    if not src.exists():
        raise FileNotFoundError(f"Missing CypherBench input: {src}")

    selected = set(selected_dbs or configured_dbs(config, db_split))
    source_name = f"CypherBench-{split}"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    rows = load_json_or_jsonl(src)
    for row_idx, record in enumerate(rows):
        preprocessed = _ensure_preprocessed(
            record,
            sample_limit=sample_limit,
            ignore_case=ignore_case,
        )
        graph = preprocessed.get("graph") or preprocessed.get("db")
        if selected and graph not in selected:
            continue
        structural = _to_structural_record(
            preprocessed,
            row_id=row_idx,
            source=source_name,
        )
        grouped[str(graph)].append(structural)

    counts: dict[str, int] = {}
    for db_id, db_rows in sorted(grouped.items()):
        out = Path(output_dir) / f"{_split_prefix(split)}_{db_id}.json"
        dump_json(db_rows, out)
        counts[db_id] = len(db_rows)
        print(f"[CypherBench] {split}/{db_id}: {len(db_rows)} rows -> {out}")

    return counts


def run_template_extraction(
    benchmark: str,
    sources_config_path: Path | str,
    output_dir: Path | str,
    *,
    split: str = "Dev",
    db_split: str | None = None,
    input_file: Path | str | None = None,
    selected_dbs: list[str] | None = None,
    sample_limit: int = 100,
    ignore_case: bool = False,
) -> dict[str, int]:
    cfg = load_sources_config(sources_config_path)
    output_dir = ensure_dir(output_dir)

    if benchmark == "CypherBench":
        return extract_cypherbench(
            cfg["CypherBench"],
            output_dir,
            split=split,
            db_split=db_split,
            input_file=input_file,
            selected_dbs=selected_dbs,
            sample_limit=sample_limit,
            ignore_case=ignore_case,
        )

    raise ValueError(f"Unsupported benchmark: {benchmark}")
