from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
from benchmark.utils.io import dump_json, iter_json_files, load_json

# Json 을 읽어서, 템플릿 별 빈도와 관측 값 목록을 집계

def _normalize_literal_examples(literals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(literals)
    for literal in normalized:
        if "value" in literal and "example_value" not in literal:
            literal["example_value"] = literal.pop("value")
    return normalized


def analyze_preprocessed_file(
    file_path: Path,
    *,
    include_evidence: bool = False,
) -> dict[str, Any]:
    rows = load_json(file_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list JSON: {file_path}")

    sql_key = "semi_template_m2"
    question_key = "semi_template_m2"

    template_info: dict[str, dict[str, Any]] = {}
    column_values: dict[str, set[str]] = {}

    for item in rows:
        literals = item.get("literals", {}).get("m2", [])
        for literal in literals:
            column_name = literal.get("column_name")
            if not column_name:
                continue
            value = literal.get("value", literal.get("example_value"))
            if value is None:
                continue
            column_values.setdefault(column_name, set()).add(str(value))

        sql_template = item.get("sql", {}).get(sql_key, "")
        if not sql_template or not sql_template.strip() or sql_template == "NULL":
            continue

        question_template = item.get("question", {}).get(question_key, "")
        evidence_template = item.get("evidence", {}).get(question_key, "")

        if sql_template not in template_info:
            template_info[sql_template] = {
                "count": 0,
                "question_semi_template": [question_template] if question_template else [],
                "sql_semi_template": sql_template,
                "literals": _normalize_literal_examples(literals),
            }
            if include_evidence:
                template_info[sql_template]["evidence_semi_template"] = evidence_template
        elif question_template and question_template not in template_info[sql_template]["question_semi_template"]:
            template_info[sql_template]["question_semi_template"].append(question_template)

        template_info[sql_template]["count"] += 1

    sorted_templates = sorted(template_info.values(), key=lambda x: x["count"], reverse=True)
    templates = []
    for idx, info in enumerate(sorted_templates, start=1):
        row = {
            "template_id": idx,
            "question_semi_template": info["question_semi_template"],
            "sql_semi_template": info["sql_semi_template"],
            "cnt": info["count"],
            "literals": info["literals"],
        }
        if include_evidence and "evidence_semi_template" in info:
            row["evidence_semi_template"] = info["evidence_semi_template"]
        templates.append(row)

    return {
        "unique_template": len(templates),
        "cnt": sum(t["cnt"] for t in templates),
        "templates": templates,
        "column_values": {k: sorted(v) for k, v in sorted(column_values.items())},
    }


def _parse_split_and_db(filename: str) -> tuple[str, str] | None:
    stem = Path(filename).stem
    if stem.startswith("train_"):
        return "Train", stem.removeprefix("train_")
    if stem.startswith("dev_"):
        return "Dev", stem.removeprefix("dev_")
    return None


def build_distribution(
    benchmark_name: str,
    preprocessed_dir: Path | str,
    output_file: Path | str,
    *,
    include_evidence: bool = False,
) -> dict[str, Any]:
    preprocessed_dir = Path(preprocessed_dir)

    result: dict[str, Any] = {
        benchmark_name: {
            "Train": {},
            "Dev": {},
        }
    }

    for file_path in iter_json_files(preprocessed_dir):
        parsed = _parse_split_and_db(file_path.name)
        if not parsed:
            continue
        split, db_id = parsed

        summary = analyze_preprocessed_file(
            file_path,
            include_evidence=include_evidence,
        )
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
    include_evidence: bool = False,
) -> dict[str, Any]:
    result = build_distribution(
        benchmark_name=benchmark_name,
        preprocessed_dir=repaired_dir,
        output_file=output_file,
        include_evidence=include_evidence,
    )
    print(f"[POOL] Saved template pool: {output_file}")
    return result


def build_literals_cache(
    template_pool: dict[str, Any],
    output_file: Path | str,
) -> dict[str, Any]:
    if not template_pool:
        raise ValueError("Empty template pool")

    benchmark_name = next(iter(template_pool.keys()))
    cache: dict[str, Any] = {benchmark_name: {"Train": {}, "Dev": {}}}

    for split in ["Train", "Dev"]:
        split_rows = template_pool.get(benchmark_name, {}).get(split, {})
        for db_id, summary in split_rows.items():
            column_values = summary.get("column_values", {})
            cache[benchmark_name][split][db_id] = {
                column: {"existing_values": sorted(values)}
                for column, values in column_values.items()
            }

    dump_json(cache, output_file)
    print(f"[LITERALS] Saved literals cache: {output_file}")
    return cache
