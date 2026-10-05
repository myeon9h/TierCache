from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from benchmark.utils.io import dump_json, iter_json_files, load_json

# Template Consistency 보정 (질문 Template 에 없는 [MASK] 가 SQL Template 에 있을 경우 채워넣기)

def extract_template_tokens(template: str) -> set[str]:
    pattern = r"\[m2_\d+\]"
    return set(re.findall(pattern, template or ""))


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
        original_type = literal.get("original_type", "str")

        if original_type in {"int", "float"}:
            restored = restored.replace(token, raw_value)
        else:
            restored = restored.replace(f"'{token}'", f"'{raw_value}'")
            restored = restored.replace(f'"{token}"', f'"{raw_value}"')

    return restored


def fix_record(record: dict[str, Any], include_evidence: bool) -> int:
    question_semi_template = record.get("question", {}).get("semi_template_m2", "")
    sql_semi_template = record.get("sql", {}).get("semi_template_m2", "")
    literals = record.get("literals", {}).get("m2", [])

    if not question_semi_template or not sql_semi_template or not literals:
        return 0

    fixed = 0
    question_tokens = extract_template_tokens(question_semi_template)
    sql_tokens = extract_template_tokens(sql_semi_template)

    missing_in_question = sql_tokens - question_tokens
    if missing_in_question:
        record["sql"]["semi_template_m2"] = restore_missing_tokens(
            sql_semi_template,
            literals,
            missing_in_question,
        )
        fixed += 1

    if include_evidence and "evidence" in record:
        evidence_semi_template = record.get("evidence", {}).get("semi_template_m2", "")
        if evidence_semi_template:
            evidence_tokens = extract_template_tokens(evidence_semi_template)
            missing_in_evidence = evidence_tokens - question_tokens
            if missing_in_evidence:
                record["evidence"]["semi_template_m2"] = restore_missing_tokens(
                    evidence_semi_template,
                    literals,
                    missing_in_evidence,
                )
                fixed += 1

    return fixed


def validate_and_fix_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    include_evidence: bool = False,
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
            file_fixed += fix_record(record, include_evidence=include_evidence)

            q_tmpl = record.get("question", {}).get("semi_template_m2", "")
            masking_cnt = count_masking_tokens(q_tmpl)
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
        "include_evidence": include_evidence,
        "masking": "m2",
        "dry_run": dry_run,
    }
