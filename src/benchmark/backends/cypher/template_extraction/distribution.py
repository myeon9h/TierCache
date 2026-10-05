from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from benchmark.utils.io import dump_json, iter_json_files, load_json


def _normalize_literal_examples(literals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(literals)
    for literal in normalized:
        if "value" in literal and "example_value" not in literal:
            literal["example_value"] = literal["value"]
    return normalized


def _literal_key(literal: dict[str, Any]) -> str:
    contexts = literal.get("contexts") or []
    context = contexts[0] if contexts and isinstance(contexts[0], dict) else {}
    prop = context.get("property")
    if prop and context.get("owner_type") == "node" and context.get("label"):
        return f"node:{context['label']}.{prop}"
    if prop and context.get("owner_type") == "relationship" and context.get("relation_type"):
        return f"relationship:{context['relation_type']}.{prop}"
    if context.get("expression"):
        return f"expression:{context['expression']}"
    return str(literal.get("slot", literal.get("index", "unknown")))


def analyze_preprocessed_file(file_path: Path, *, include_meta: bool = True) -> dict[str, Any]:
    rows = load_json(file_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list JSON: {file_path}")

    template_info: dict[str, dict[str, Any]] = {}
    literal_values: dict[str, set[str]] = {}

    for item in rows:
        literals = item.get("literals", {}).get("m2", [])
        for literal in literals:
            value = literal.get("value", literal.get("example_value"))
            if value is None:
                continue
            literal_values.setdefault(_literal_key(literal), set()).add(str(value))

        cypher_template = item.get("cypher", {}).get("semi_template_m2", "")
        if not cypher_template or not str(cypher_template).strip() or cypher_template == "NULL":
            continue

        question_template = item.get("question", {}).get("semi_template_m2", "")
        sampling = item.get("sampling", {})

        if cypher_template not in template_info:
            template_info[cypher_template] = {
                "count": 0,
                "question_semi_template": [question_template] if question_template else [],
                "cypher_semi_template": cypher_template,
                "literals": _normalize_literal_examples(literals),
                "sampling_query": sampling.get("query"),
                "sampling_returns": sampling.get("returns", []),
                "sampling_policy": sampling.get("policy", {}),
            }
            if include_meta:
                template_info[cypher_template]["meta"] = item.get("meta", {})
        elif question_template and question_template not in template_info[cypher_template]["question_semi_template"]:
            template_info[cypher_template]["question_semi_template"].append(question_template)

        template_info[cypher_template]["count"] += 1

    sorted_templates = sorted(template_info.values(), key=lambda x: x["count"], reverse=True)
    templates = []
    for idx, info in enumerate(sorted_templates, start=1):
        row = {
            "template_id": idx,
            "question_semi_template": info["question_semi_template"],
            "cypher_semi_template": info["cypher_semi_template"],
            "cnt": info["count"],
            "literals": info["literals"],
        }
        if info.get("sampling_query"):
            row["sampling_query"] = info["sampling_query"]
            row["sampling_returns"] = info.get("sampling_returns", [])
        if info.get("sampling_policy"):
            row["sampling_policy"] = info["sampling_policy"]
        if include_meta and info.get("meta"):
            row["meta"] = info["meta"]
        templates.append(row)

    sorted_literal_values = {k: sorted(v) for k, v in sorted(literal_values.items())}
    return {
        "unique_template": len(templates),
        "cnt": sum(t["cnt"] for t in templates),
        "templates": templates,
        "literal_values": sorted_literal_values,
        "column_values": sorted_literal_values,
    }


def _parse_split_and_db(filename: str) -> tuple[str, str] | None:
    stem = Path(filename).stem
    if stem.startswith("train_"):
        return "Train", stem.removeprefix("train_")
    if stem.startswith("test_"):
        return "Test", stem.removeprefix("test_")
    if stem.startswith("dev_"):
        return "Dev", stem.removeprefix("dev_")
    return None


def build_distribution(
    benchmark_name: str,
    preprocessed_dir: Path | str,
    output_file: Path | str,
    *,
    include_meta: bool = True,
) -> dict[str, Any]:
    preprocessed_dir = Path(preprocessed_dir)
    result: dict[str, Any] = {benchmark_name: {"Train": {}, "Dev": {}, "Test": {}}}

    for file_path in iter_json_files(preprocessed_dir):
        parsed = _parse_split_and_db(file_path.name)
        if not parsed:
            continue
        split, db_id = parsed
        summary = analyze_preprocessed_file(file_path, include_meta=include_meta)
        result[benchmark_name][split][db_id] = summary
        print(
            f"[DIST] {split}/{db_id}: "
            f"templates={summary['unique_template']}, rows={summary['cnt']}"
        )

    dump_json(result, output_file)
    print(f"[DIST] Saved distribution: {output_file}")
    return result


def build_template_pool(
    benchmark_name: str,
    repaired_dir: Path | str,
    output_file: Path | str,
    *,
    include_meta: bool = True,
) -> dict[str, Any]:
    result = build_distribution(
        benchmark_name=benchmark_name,
        preprocessed_dir=repaired_dir,
        output_file=output_file,
        include_meta=include_meta,
    )
    print(f"[POOL] Saved template pool: {output_file}")
    return result


def build_literals_cache(template_pool: dict[str, Any], output_file: Path | str) -> dict[str, Any]:
    if not template_pool:
        raise ValueError("Empty template pool")

    benchmark_name = next(iter(template_pool.keys()))
    cache: dict[str, Any] = {benchmark_name: {"Train": {}, "Dev": {}, "Test": {}}}

    for split in ["Train", "Dev", "Test"]:
        split_rows = template_pool.get(benchmark_name, {}).get(split, {})
        for db_id, summary in split_rows.items():
            values = summary.get("literal_values", summary.get("column_values", {}))
            cache[benchmark_name][split][db_id] = {
                key: {"existing_values": sorted(existing)}
                for key, existing in values.items()
            }

    dump_json(cache, output_file)
    print(f"[LITERALS] Saved literals cache: {output_file}")
    return cache
