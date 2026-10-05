from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.utils.io import dump_json
from benchmark.backends.sql.db import DBSampler
from benchmark.backends.sql.loaders import (
    load_distribution_templates,
    load_literals_data,
)
from benchmark.workload_generation.sampling import (
    sample_templates_uniform,
    sample_templates_zipf,
)


@dataclass
class WorkloadGenConfig:
    benchmark_type: str
    split: str
    target_db: str
    distribution_type: str
    num_queries: int
    output_file: str
    template_pool_file: str
    literals_file: str | None
    db_config_file: str
    criterion: str = "rank"
    alpha: float = 1.0
    power_law_s: float = 2.0
    max_retries: int = 10
    require_db_sampling: bool = False
    seed: int = 42


def _count_masks(template: str) -> int:
    tokens = set(re.findall(r"\[m2_\d+\]", template or ""))
    return len(tokens)


def _choose_question_template(raw: Any) -> str:
    if isinstance(raw, list):
        if not raw:
            return ""
        return random.choice(raw)
    return raw or ""


def _apply_literals(template: str, literal_values: dict[int, Any]) -> str:
    output = template
    for idx, value in literal_values.items():
        output = output.replace(f"[m2_{idx}]", str(value))
    return output


def _apply_sql_literals(template: str, literal_values: dict[int, Any]) -> str:
    escaped = {idx: value.replace("'", "''") if isinstance(value, str) else value
               for idx, value in literal_values.items()}
    return _apply_literals(template, escaped)


def _top_k(counter: Counter, k: int = 20) -> dict[str, int]:
    return {k_: v for k_, v in counter.most_common(k)}


def generate_workload(cfg: WorkloadGenConfig) -> dict[str, Any]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    templates = load_distribution_templates(
        cfg.template_pool_file,
        cfg.benchmark_type,
        cfg.split,
        cfg.target_db,
    )
    if not templates:
        raise ValueError("No templates loaded. Check template-pool file and filters.")

    literals_data = (
        load_literals_data(cfg.literals_file, cfg.benchmark_type, cfg.split, cfg.target_db)
        if cfg.literals_file
        else None
    )

    if cfg.distribution_type == "zipf":
        sampled_templates = sample_templates_zipf(
            templates,
            cfg.num_queries,
            criterion=cfg.criterion,
            alpha=cfg.alpha,
            power_law_s=cfg.power_law_s,
        )
    elif cfg.distribution_type == "uniform":
        sampled_templates = sample_templates_uniform(templates, cfg.num_queries)
    else:
        raise ValueError(f"Unsupported distribution type: {cfg.distribution_type}")

    target_template_counter = Counter(t.get("template_id") for t in sampled_templates)

    generated_queries: list[dict[str, Any]] = []
    generated_template_counter: Counter = Counter()
    sampling_method_counter: Counter = Counter()
    masking_counter: Counter = Counter()

    total_failed_generations = 0
    total_attempts = 0
    rejected_non_db_sampling = 0

    sampler = DBSampler(cfg.db_config_file)
    try:
        idx = 0
        max_total_attempts = max(cfg.num_queries * max(cfg.max_retries, 1) * 5, cfg.num_queries)

        while len(generated_queries) < cfg.num_queries and total_attempts < max_total_attempts:
            template = sampled_templates[idx % len(sampled_templates)]
            idx += 1
            total_attempts += 1

            question_template = _choose_question_template(template.get("question_semi_template"))
            sql_template = template.get("sql_semi_template", "")
            evidence_template = template.get("evidence_semi_template", "")
            literals = template.get("literals", [])
            masking_cnt = _count_masks(question_template)

            question = question_template
            sql = sql_template
            evidence = evidence_template if evidence_template else None
            literal_values: dict[int, Any] = {}
            sampling_method = "original"

            has_masking = "[m2_" in question_template or "[m2_" in sql_template

            if has_masking and literals:
                actual_db = template.get("source_db", cfg.target_db)
                succeeded = False

                for _ in range(cfg.max_retries):
                    sampled_values, method, failed_cols = sampler.sample_literals(
                        actual_db,
                        literals,
                        literals_data,
                    )
                    if failed_cols:
                        continue
                    if cfg.require_db_sampling and method != "db":
                        rejected_non_db_sampling += 1
                        continue

                    candidate_question = _apply_literals(question_template, sampled_values)
                    candidate_sql = _apply_sql_literals(sql_template, sampled_values)
                    is_valid = sampler.validate_sql(actual_db, candidate_sql)
                    if not is_valid:
                        continue

                    question = candidate_question
                    sql = candidate_sql
                    evidence = _apply_literals(evidence_template, sampled_values) if evidence else None
                    literal_values = sampled_values
                    sampling_method = method
                    succeeded = True
                    break

                if not succeeded:
                    total_failed_generations += 1
                    continue
            elif not sampler.validate_sql(template.get("source_db", cfg.target_db), sql):
                total_failed_generations += 1
                continue

            query_obj = {
                "id": len(generated_queries) + 1,
                "template_id": template.get("template_id"),
                "question_semi_template": question_template,
                "question": question,
                "sql_semi_template": sql_template,
                "sql": sql,
                "sampling_method": sampling_method,
                "literal_values": literal_values,
                "masking_cnt": masking_cnt,
                "iteration": total_attempts,
            }

            if evidence is not None:
                query_obj["evidence_semi_template"] = evidence_template
                query_obj["evidence"] = evidence

            if "source_db" in template:
                query_obj["source_db"] = template["source_db"]

            generated_queries.append(query_obj)
            generated_template_counter[template.get("template_id")] += 1
            sampling_method_counter[sampling_method] += 1
            masking_counter[masking_cnt] += 1

    finally:
        sampler.close()

    if len(generated_queries) != cfg.num_queries:
        raise RuntimeError(
            f"Generated only {len(generated_queries)}/{cfg.num_queries} queries after "
            f"{total_attempts} attempts; check DB connections, templates and literal sampling. No workload saved."
        )

    statistics = {
        "generated_queries": len(generated_queries),
        "failed_generations": total_failed_generations,
        "total_attempts": total_attempts,
        "rejected_non_db_sampling": rejected_non_db_sampling,
        "queries_per_sampling_method": dict(sampling_method_counter),
        "queries_per_masking_cnt": dict(masking_counter),
        "target_template_counts": len(target_template_counter),
        "generated_template_counts": len(generated_template_counter),
        "target_top_20_template_distribution": _top_k(target_template_counter, 20),
        "generated_top_20_template_distribution": _top_k(generated_template_counter, 20),
    }

    result = {
        "config": {
            "benchmark_type": cfg.benchmark_type,
            "split": cfg.split,
            "target_db": cfg.target_db,
            "distribution_type": cfg.distribution_type,
            "criterion": cfg.criterion if cfg.distribution_type == "zipf" else None,
            "alpha": cfg.alpha if cfg.distribution_type == "zipf" else None,
            "power_law_s": cfg.power_law_s if cfg.criterion == "query_len" else None,
            "max_retries": cfg.max_retries,
            "sampling_mode": "db" if cfg.require_db_sampling else "hybrid",
            "num_queries": cfg.num_queries,
            "seed": cfg.seed,
        },
        "statistics": statistics,
        "queries": generated_queries,
    }

    # Persist only the workload payload; keep statistics as runtime metadata.
    output_payload = {
        "config": result["config"],
        "queries": result["queries"],
    }
    dump_json(output_payload, Path(cfg.output_file))
    return result
