#!/usr/bin/env python3
"""Check SQStream source files and SQL/Neo4j connection configuration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.catalog import BENCHMARKS
from benchmark.paths import CONFIG_ROOT, resolve_path


def load_config(path):
    with resolve_path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected nonempty YAML mapping: {path}")
    return data


def source_files(name, cfg, split):
    if name == "CypherBench":
        dbs = cfg.get("dbs", [])
        if isinstance(dbs, dict):
            dbs = dbs.get(split.lower(), [])
        candidates = [cfg.get(kind, {}).get(split) for kind in ("preprocessed", "raw")]
        path = next((resolve_path(p) for p in candidates if p and resolve_path(p).is_file()), None)
        if path is None:
            raise ValueError(f"No existing CypherBench {split} input in raw/preprocessed config")
        return [path], dbs
    root = resolve_path(cfg["root"])
    dbs = cfg.get("dbs", [])
    split_spec = cfg.get("splits", {}).get(split)
    if not split_spec:
        raise ValueError(f"Split {split} is not configured for {name}")
    files = []
    for db in dbs:
        if name == "BIRD":
            combined = root / split_spec
            files.append(combined if combined.is_file() else root / db / split_spec)
        else:
            stems = split_spec if isinstance(split_spec, list) else [split_spec]
            files.extend(root / db / f"{stem}.json" for stem in stems)
    return files, dbs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-config", default=str(CONFIG_ROOT / "benchmark_sources.yaml"))
    parser.add_argument("--db-config", default=str(CONFIG_ROOT / "db_config.yaml"))
    parser.add_argument("--benchmark", choices=BENCHMARKS, help="Check only this source")
    parser.add_argument("--split", choices=["Train", "Dev", "Test"])
    parser.add_argument("--skip-db-check", action="store_true")
    parser.add_argument("--check-connections", action="store_true", help="Also execute SELECT 1 / RETURN 1")
    args = parser.parse_args()
    if args.skip_db_check and args.check_connections:
        parser.error("--skip-db-check and --check-connections are mutually exclusive")
    try:
        sources = load_config(args.sources_config)
        db_config = load_config(args.db_config) if not args.skip_db_check else {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f"[FAIL] {exc}\nCopy the corresponding .example.yaml files first.\n")
    selected = [args.benchmark] if args.benchmark else list(sources)
    errors = []
    used_dbs = set()
    for name in selected:
        try:
            if name not in BENCHMARKS:
                raise ValueError(f"Unsupported source: {name}")
            files, dbs = source_files(name, sources[name], args.split or ("Test" if name == "CypherBench" else "Dev"))
            if not dbs:
                raise ValueError(f"No DBs configured for {name}")
            for path in set(files):
                if not path.is_file():
                    errors.append(f"Missing source: {path}")
            used_dbs.update(dbs)
            if not args.skip_db_check:
                expected = "neo4j" if name == "CypherBench" else ("postgresql" if name == "ScienceBenchmark" else "sqlite")
                for db in dbs:
                    if db_config.get(db, {}).get("type") != expected:
                        errors.append(f"{name}/{db}: expected DB type {expected}")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    if not args.skip_db_check:
        for db in sorted(used_dbs):
            cfg = db_config.get(db, {})
            kind = cfg.get("type")
            if kind == "sqlite":
                if not cfg.get("path") or not resolve_path(cfg["path"]).is_file():
                    errors.append(f"{db}: missing SQLite path/file")
            elif kind == "postgresql":
                if not cfg.get("url"):
                    errors.append(f"{db}: missing PostgreSQL URL")
            elif kind == "neo4j":
                if not (cfg.get("uri") or (cfg.get("host") and cfg.get("port"))):
                    errors.append(f"{db}: missing Neo4j URI or host/port")
                if not cfg.get("username") or not cfg.get("password"):
                    errors.append(f"{db}: missing Neo4j credentials")
            else:
                errors.append(f"{db}: unsupported or missing DB type {kind}")
    if args.check_connections and not errors:
        from benchmark.backends.sql.db import DBSampler
        from benchmark.backends.cypher.db import Neo4jSampler
        sql_sampler, cypher_sampler = DBSampler(args.db_config), Neo4jSampler(args.db_config)
        try:
            for db in sorted(used_dbs):
                try:
                    if db_config[db]["type"] == "neo4j":
                        cypher_sampler.verify(db)
                    else:
                        sql_sampler._get_conn(db).execute("SELECT 1").fetchone()
                except Exception as exc:
                    # Keep connection URLs/passwords out of diagnostic output.
                    errors.append(f"{db}: connection check failed ({type(exc).__name__})")
        finally:
            sql_sampler.close()
            cypher_sampler.close()
    if errors:
        parser.exit(1, "[FAIL]\n" + "\n".join(f"- {error}" for error in errors) + "\n")
    print("[OK] SQStream setup check passed")


if __name__ == "__main__":
    main()
