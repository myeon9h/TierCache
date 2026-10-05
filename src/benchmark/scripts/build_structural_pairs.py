#!/usr/bin/env python3
"""Build SQStream template and literal pools for SQL or Cypher sources."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.catalog import BENCHMARKS
from benchmark.paths import CONFIG_ROOT, POOL_ROOT, resolve_path
from benchmark.utils.io import dump_json, load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--sources-config", default=str(CONFIG_ROOT / "benchmark_sources.yaml"))
    parser.add_argument("--pool-root", default=str(POOL_ROOT))
    parser.add_argument("--split", choices=["Train", "Dev", "Test"])
    parser.add_argument("--bird-split", choices=["Train", "Dev"], help="Alias for --split for BIRD")
    parser.add_argument("--input", help="Override CypherBench raw/preprocessed JSON or JSONL")
    parser.add_argument("--sample-limit", default=100, type=int)
    parser.add_argument("--ignore-case", action="store_true", help="Cypher literal matching")
    args = parser.parse_args()
    if args.bird_split and args.benchmark != "BIRD":
        parser.error("--bird-split is only supported for BIRD")
    if args.split and args.bird_split and args.split != args.bird_split:
        parser.error("--split and --bird-split disagree")
    is_cypher = args.benchmark == "CypherBench"
    split = args.split or args.bird_split or ("Test" if is_cypher else "Dev")
    if args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    if not is_cypher and (split == "Test" or args.input):
        parser.error("SQL pool building supports Train/Dev and uses --sources-config")
    source_config = resolve_path(args.sources_config)
    if not source_config.is_file():
        parser.error(f"Missing config: {source_config}; copy benchmark_sources.example.yaml first")
    backend = "cypher" if is_cypher else "sql"
    prefix = f"benchmark.backends.{backend}.template_extraction"
    pipeline = importlib.import_module(f"{prefix}.pipeline")
    validation = importlib.import_module(f"{prefix}.validation")
    distribution = importlib.import_module(f"{prefix}.distribution")
    pool = resolve_path(args.pool_root)
    template_path = pool / "template_pool" / f"{args.benchmark}.json"
    literal_path = pool / "literals" / f"{args.benchmark}.json"
    with tempfile.TemporaryDirectory(prefix="sqstream_build_") as temp:
        raw, repaired = Path(temp) / "raw", Path(temp) / "repaired"
        if is_cypher:
            pipeline.run_template_extraction(
                args.benchmark, source_config, raw, split=split,
                input_file=resolve_path(args.input) if args.input else None,
                sample_limit=args.sample_limit, ignore_case=args.ignore_case,
            )
            summary = validation.validate_and_fix_directory(raw, repaired)
            built = distribution.build_template_pool(args.benchmark, repaired, Path(temp) / "pool.json")
        else:
            pipeline.run_template_extraction(
                args.benchmark, source_config, raw, split=split, bird_split=split,
            )
            summary = validation.validate_and_fix_directory(raw, repaired, include_evidence=True)
            built = distribution.build_template_pool(args.benchmark, repaired, Path(temp) / "pool.json", include_evidence=True)
        generated = built[args.benchmark].get(split, {})
        if not generated or not any(row.get("templates") for row in generated.values()):
            raise ValueError(f"No templates extracted for {args.benchmark}/{split}; check source files and DB filters")
        # Retain other splits and DBs when building a selected subset again.
        merged = load_json(template_path) if template_path.exists() else {args.benchmark: {}}
        merged.setdefault(args.benchmark, {}).setdefault(split, {}).update(generated)
        cache = distribution.build_literals_cache(merged, Path(temp) / "literals.json")
        dump_json(merged, template_path)
        dump_json(cache, literal_path)
        print(json.dumps(summary, indent=2))
        print(f"[SQStream] Saved {template_path} and {literal_path}")


if __name__ == "__main__":
    main()
