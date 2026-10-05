from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.utils.io import dump_json, iter_json_files, load_json


def extract_template_tokens(template: str) -> set[str]:
    return set(re.findall(r"\[m2_\d+\]", template or ""))


def count_masking_tokens(template: str) -> int:
    return len(extract_template_tokens(template))


def restore_missing_tokens(
    template: str,
    literals: list[dict[str, Any]],
    missing_tokens: set[str],
) -> str:
    restored = template
    for idx, literal in enumerate(literals):
        token = f"[m2_{idx}]"
        if token not in missing_tokens:
            continue

        raw_value = str(literal.get("value", literal.get("example_value", "")))
        literal_type = literal.get("type", literal.get("original_type", "string"))
        if literal_type in {"number", "float", "int"}:
            restored = restored.replace(token, raw_value)
        else:
            restored = restored.replace(f"'{token}'", f"'{raw_value}'")
            restored = restored.replace(f'"{token}"', f'"{raw_value}"')
            restored = restored.replace(token, raw_value)
    return restored


def fix_record(record: dict[str, Any]) -> int:
    question_template = record.get("question", {}).get("semi_template_m2", "")
    cypher_template = record.get("cypher", {}).get("semi_template_m2", "")
    literals = record.get("literals", {}).get("m2", [])

    if not question_template or not cypher_template or not literals:
        return 0

    question_tokens = extract_template_tokens(question_template)
    cypher_tokens = extract_template_tokens(cypher_template)
    missing_in_question = cypher_tokens - question_tokens
    if not missing_in_question:
        return 0

    record["cypher"]["semi_template_m2"] = restore_missing_tokens(
        cypher_template,
        literals,
        missing_in_question,
    )
    return 1


def validate_and_fix_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    dry_run: bool = False,
    write_all_files: bool = True,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_fixed = 0
    per_file_stats: dict[str, dict[str, Any]] = {}
    overall_masking = defaultdict(int)

    for json_file in iter_json_files(input_dir):
        rows = load_json(json_file)
        if not isinstance(rows, list):
            print(f"[WARN] Skip non-list file: {json_file}")
            continue

        file_fixed = 0
        masking_stats = defaultdict(int)
        for record in rows:
            file_fixed += fix_record(record)
            q_tmpl = record.get("question", {}).get("semi_template_m2", "")
            c_tmpl = record.get("cypher", {}).get("semi_template_m2", "")
            masking_cnt = max(count_masking_tokens(q_tmpl), count_masking_tokens(c_tmpl))
            masking_stats[masking_cnt] += 1

        total_rows += len(rows)
        total_fixed += file_fixed
        for k, v in masking_stats.items():
            overall_masking[k] += v

        if not dry_run and (write_all_files or file_fixed > 0):
            dump_json(rows, output_dir / json_file.name)

        per_file_stats[json_file.name] = {
            "rows": len(rows),
            "fixed": file_fixed,
            "masking_stats": dict(sorted(masking_stats.items())),
        }
        print(f"[VALIDATE] {json_file.name}: rows={len(rows)}, fixed={file_fixed}")

    return {
        "total_rows": total_rows,
        "total_fixed": total_fixed,
        "files": per_file_stats,
        "overall_masking": dict(sorted(overall_masking.items())),
        "masking": "m2",
        "dry_run": dry_run,
    }
