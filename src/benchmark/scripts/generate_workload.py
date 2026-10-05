#!/usr/bin/env python3
"""Generate SQL or Cypher SQStream workloads through the same CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.catalog import BENCHMARKS, database_name, workload_name
from benchmark.paths import CONFIG_ROOT, POOL_ROOT, WORKLOAD_ROOT, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_type", "--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--split", choices=["Train", "Dev", "Test", "Combined"])
    parser.add_argument("--target_db", required=True, help="Source DB name or paper workload name")
    parser.add_argument("--distribution_type", default="query_len", choices=["query_len", "rank", "uniform"])
    parser.add_argument("--num_queries", default=1000, type=int)
    parser.add_argument("--power_law_s", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--max_retries", default=25, type=int)
    parser.add_argument("--sampling_mode", default="db", choices=["db", "hybrid"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--pool-root", default=str(POOL_ROOT))
    parser.add_argument("--db_config_file", default=str(CONFIG_ROOT / "db_config.yaml"))
    parser.add_argument("--output_file", help="Default: data/SQStream/{Source}-{Workload}.json")
    parser.add_argument("--sample_pool_size", default=100, type=int, help="Cypher DB candidate rows")
    parser.add_argument("--connection_timeout", default=2.0, type=float)
    parser.add_argument("--query_timeout", default=10.0, type=float)
    args = parser.parse_args()
    is_cypher = args.benchmark_type == "CypherBench"
    split = args.split or ("Test" if is_cypher else "Dev")
    if not is_cypher and split == "Test":
        parser.error("SQL sources support Train, Dev or Combined; use Dev for paper workloads")
    if args.num_queries <= 0 or args.max_retries <= 0 or args.sample_pool_size <= 0:
        parser.error("num_queries, max_retries and sample_pool_size must be positive")
    if args.connection_timeout <= 0 or args.query_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.distribution_type != "rank" and args.alpha is not None:
        parser.error("--alpha requires --distribution_type rank")
    if args.distribution_type != "query_len" and args.power_law_s is not None:
        parser.error("--power_law_s requires --distribution_type query_len")
    if (args.alpha is not None and args.alpha <= 0) or (args.power_law_s is not None and args.power_law_s <= 0):
        parser.error("distribution exponents must be positive")
    target_db = database_name(args.benchmark_type, args.target_db)
    try:
        output = resolve_path(args.output_file) if args.output_file else WORKLOAD_ROOT / f"{args.benchmark_type}-{workload_name(args.benchmark_type, target_db)}.json"
    except ValueError as exc:
        parser.error(str(exc))
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose a new --output_file to preserve existing workloads.")
    pool = resolve_path(args.pool_root)
    templates = pool / "template_pool" / f"{args.benchmark_type}.json"
    literals = pool / "literals" / f"{args.benchmark_type}.json"
    db_config = resolve_path(args.db_config_file)
    for path in (templates, db_config):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if is_cypher:
        from benchmark.backends.cypher.generator import WorkloadGenConfig, generate_workload
    else:
        from benchmark.backends.sql.generator import WorkloadGenConfig, generate_workload
    config = dict(
        benchmark_type=args.benchmark_type, split=split, target_db=target_db,
        distribution_type="uniform" if args.distribution_type == "uniform" else "zipf",
        criterion="query_len" if args.distribution_type == "query_len" else "rank",
        alpha=args.alpha if args.alpha is not None else 1.0,
        power_law_s=args.power_law_s if args.power_law_s is not None else 2.0,
        num_queries=args.num_queries, max_retries=args.max_retries, seed=args.seed,
        output_file=str(output), template_pool_file=str(templates),
        literals_file=str(literals) if literals.is_file() else None,
        db_config_file=str(db_config), require_db_sampling=args.sampling_mode == "db",
    )
    if is_cypher:
        config.update(sample_pool_size=args.sample_pool_size,
                      connection_timeout=args.connection_timeout, query_timeout=args.query_timeout)
    result = generate_workload(WorkloadGenConfig(**config))
    print(json.dumps(result["statistics"], indent=2, ensure_ascii=False))
    print(f"[SQStream] Saved {output}")


if __name__ == "__main__":
    main()
