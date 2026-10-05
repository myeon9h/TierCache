from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml
from sqlglot import parse_one

from benchmark.utils.io import dump_json, ensure_dir, load_json
from benchmark.backends.sql.template_extraction.masking import (
    collect_sql_literals,
    collect_text_literals,
    create_indexed_sql_template,
    create_indexed_template,
    extract_literals_with_columns,
)
from benchmark.backends.sql.template_extraction.schema import SchemaStore, resolve_column_with_schema


from benchmark.paths import REPO_ROOT


def _resolve_repo_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_sources_config(config_path: Path | str) -> dict[str, Any]:
    path = _resolve_repo_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing sources config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_literal_set(values: Iterable[str], case_sensitive: bool) -> set[str]:
    if case_sensitive:
        return set(values)
    return {v.lower() for v in values}


def _build_record(
    *,
    row_id: int,
    source: str,
    db_id: str,
    question: str,
    sql_query: str,
    dialect: str,
    case_sensitive: bool,
    resolve_column,
    evidence: str = "",
) -> dict[str, Any] | None:
    try:
        tree = parse_one(sql_query, dialect=dialect)
    except Exception:
        return None

    m2_targets = collect_sql_literals(sql_query, dialect, case_sensitive)
    if evidence:
        evidence_literals = collect_text_literals(evidence)
        m2_targets.update(_normalize_literal_set(evidence_literals, case_sensitive))

    m2_literals = extract_literals_with_columns(
        tree,
        m2_targets,
        sql_query,
        resolve_column=lambda c: resolve_column(c, tree),
        case_sensitive=case_sensitive,
    )
    m2_sql_template = create_indexed_sql_template(
        tree,
        m2_literals,
        dialect=dialect,
        case_sensitive=case_sensitive,
    )

    record: dict[str, Any] = {
        "id": row_id,
        "source": source,
        "db": db_id,
        "question": {
            "original": question,
            "semi_template_m2": create_indexed_template(question, m2_literals, case_sensitive),
        },
        "sql": {
            "original": sql_query,
            "semi_template_m2": m2_sql_template,
        },
        "literals": {
            "m2": m2_literals,
        },
    }

    if evidence:
        record["evidence"] = {
            "original": evidence,
            "semi_template_m2": create_indexed_template(evidence, m2_literals, case_sensitive),
        }

    return record


def _split_prefix(split_name: str) -> str:
    return "train" if split_name.lower() == "train" else "dev"


def extract_ehrsql(config: dict[str, Any], output_dir: Path, case_sensitive: bool) -> None:
    root = _resolve_repo_path(config["root"])
    dbs = config.get("dbs", ["eicu", "mimic_iii"])
    splits = config.get("splits", {"Train": "train", "Dev": "valid"})
    schema_root = _resolve_repo_path(config.get("schema_root", config["root"]))
    schema_store = SchemaStore("EHRSQL", schema_root, tables_json=config.get("tables_json", "tables.json"))

    for split_name, source_split in splits.items():
        for db_id in dbs:
            src = root / db_id / f"{source_split}.json"
            if not src.exists():
                print(f"[WARN] Missing EHRSQL source: {src}")
                continue

            rows = load_json(src)
            processed = []

            for idx, ex in enumerate(rows):
                question = ex.get("question", "")
                sql_query = ex.get("query", "")
                if not question or not sql_query:
                    continue

                schema = schema_store.get(db_id)
                rec = _build_record(
                    row_id=idx,
                    source=f"EHRSQL-{split_name}",
                    db_id=db_id,
                    question=question,
                    sql_query=sql_query,
                    dialect="sqlite",
                    case_sensitive=case_sensitive,
                    resolve_column=lambda c, t, s=schema: resolve_column_with_schema(c, t, s),
                )
                if rec:
                    processed.append(rec)

            out = output_dir / f"{_split_prefix(split_name)}_{db_id}.json"
            dump_json(processed, out)
            print(f"[EHRSQL] {split_name}/{db_id}: {len(processed)} rows -> {out}")


def extract_sciencebenchmark(config: dict[str, Any], output_dir: Path, case_sensitive: bool) -> None:
    root = _resolve_repo_path(config["root"])
    dbs = config.get("dbs", ["cordis", "oncomx"])
    split_map = config.get("splits", {"Train": ["seed", "synth"], "Dev": ["dev"]})
    schema_root = _resolve_repo_path(config.get("schema_root", config["root"]))
    schema_store = SchemaStore("ScienceBenchmark", schema_root)

    for db_id in dbs:
        for split_name, source_splits in split_map.items():
            if isinstance(source_splits, str):
                source_splits = [source_splits]

            processed = []
            row_id = 0

            for source_split in source_splits:
                src = root / db_id / f"{source_split}.json"
                if not src.exists():
                    print(f"[WARN] Missing ScienceBenchmark source: {src}")
                    continue

                rows = load_json(src)
                for ex in rows:
                    question = ex.get("question", "")
                    sql_query = ex.get("query", "")
                    ex_db_id = ex.get("db_id", db_id)
                    if not question or not sql_query:
                        continue

                    schema = schema_store.get(ex_db_id)

                    rec = _build_record(
                        row_id=row_id,
                        source=f"ScienceBenchmark-{source_split}",
                        db_id=ex_db_id,
                        question=question,
                        sql_query=sql_query,
                        dialect="postgres",
                        case_sensitive=case_sensitive,
                        resolve_column=lambda c, t, s=schema: resolve_column_with_schema(c, t, s),
                    )
                    if rec:
                        processed.append(rec)
                        row_id += 1

            out = output_dir / f"{_split_prefix(split_name)}_{db_id}.json"
            dump_json(processed, out)
            print(f"[ScienceBenchmark] {split_name}/{db_id}: {len(processed)} rows -> {out}")


def extract_bird(
    config: dict[str, Any],
    output_dir: Path,
    case_sensitive: bool,
    split: str,
    selected_dbs: list[str] | None = None,
) -> None:
    root = _resolve_repo_path(config["root"])
    split_map = config.get("splits", {"Dev": "dev.json"})
    tables_json = config.get("tables_json", "dev_tables.json")
    schema_root = _resolve_repo_path(config.get("schema_root", config["root"]))
    schema_store = SchemaStore("BIRD", schema_root, tables_json=tables_json)

    if split not in split_map:
        raise ValueError(f"Unsupported BIRD split: {split}")

    if selected_dbs is None:
        selected_dbs = config.get("dbs")
    if not selected_dbs:
        raise ValueError("[BIRD] 'dbs' must be provided in sources config")

    selected_set = set(selected_dbs)
    combined_path = root / split_map[split]
    combined_rows = load_json(combined_path) if combined_path.is_file() else None
    for db_id in sorted(selected_set):
        src = root / db_id / split_map[split]
        if combined_rows is None and not src.exists():
            print(f"[WARN] Missing BIRD source: {src}")
            continue

        rows = [row for row in combined_rows if row.get("db_id") == db_id] if combined_rows is not None else load_json(src)
        if not isinstance(rows, list):
            continue

        schema = schema_store.get(db_id)
        processed: list[dict[str, Any]] = []
        row_id = 0

        for ex in rows:
            if not isinstance(ex, dict):
                continue
            question = ex.get("question", "")
            sql_query = ex.get("SQL") or ex.get("query", "")
            evidence = ex.get("evidence", "")
            if not question or not sql_query:
                continue

            rec = _build_record(
                row_id=row_id,
                source=f"BIRD-{split}",
                db_id=db_id,
                question=question,
                sql_query=sql_query,
                dialect="sqlite",
                case_sensitive=case_sensitive,
                resolve_column=lambda c, t, s=schema: resolve_column_with_schema(c, t, s),
                evidence=evidence,
            )
            if rec:
                processed.append(rec)
                row_id += 1

        out = output_dir / f"{_split_prefix(split)}_{db_id}.json"
        dump_json(processed, out)
        print(f"[BIRD] {split}/{db_id}: {len(processed)} rows -> {out}")


def run_template_extraction(
    benchmark: str,
    sources_config_path: Path | str,
    output_dir: Path | str,
    *,
    case_sensitive: bool = False,
    split: str = "Dev",
    bird_split: str = "Dev",
    bird_dbs: list[str] | None = None,
) -> None:
    cfg = load_sources_config(sources_config_path)
    if benchmark not in cfg:
        raise ValueError(f"Benchmark {benchmark} is missing from {sources_config_path}")
    cfg[benchmark] = dict(cfg[benchmark])
    split_map = cfg[benchmark].get("splits", {})
    if split not in split_map:
        raise ValueError(f"Split {split} is not configured for {benchmark}")
    cfg[benchmark]["splits"] = {split: split_map[split]}
    output_dir = ensure_dir(output_dir)

    if benchmark == "EHRSQL":
        extract_ehrsql(cfg["EHRSQL"], output_dir, case_sensitive)
        return

    if benchmark == "ScienceBenchmark":
        extract_sciencebenchmark(cfg["ScienceBenchmark"], output_dir, case_sensitive)
        return

    if benchmark == "BIRD":
        extract_bird(
            cfg["BIRD"],
            output_dir,
            case_sensitive,
            split=bird_split,
            selected_dbs=bird_dbs,
        )
        return

    raise ValueError(f"Unsupported benchmark: {benchmark}")
