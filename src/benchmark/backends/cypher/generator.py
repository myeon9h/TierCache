from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.utils.io import dump_json
from benchmark.utils.neo4j import format_cypher_slot_value, stringify_value
from benchmark.backends.cypher.db import Neo4jSampler
from benchmark.backends.cypher.loaders import (
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
    db_group: str = "full"
    criterion: str = "rank"
    alpha: float = 1.0
    power_law_s: float = 2.0
    max_retries: int = 10
    require_db_sampling: bool = False
    validate_generated: bool = True
    connection_timeout: float = 2.0
    query_timeout: float = 10.0
    sample_pool_size: int = 100
    seed: int = 42


def _count_masks(*templates: str) -> int:
    tokens: set[str] = set()
    for template in templates:
        tokens.update(re.findall(r"\[m2_\d+\]", template or ""))
    return len(tokens)


def _choose_question_template(raw: Any) -> str:
    if isinstance(raw, list):
        if not raw:
            return ""
        return random.choice(raw)
    return raw or ""


def _literal_type_by_index(literals: list[dict[str, Any]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for idx, literal in enumerate(literals):
        slot = str(literal.get("slot", f"[m2_{idx}]"))
        match = re.search(r"\[m2_(\d+)\]", slot)
        literal_idx = int(match.group(1)) if match else idx
        out[literal_idx] = str(literal.get("type", "string"))
    return out


def _apply_text_literals(template: str, literal_values: dict[int, Any]) -> str:
    output = template
    for idx, value in literal_values.items():
        output = output.replace(f"[m2_{idx}]", "" if value is None else str(value))
    return output


def _apply_cypher_literals(
    template: str,
    literal_values: dict[int, Any],
    literal_types: dict[int, str],
) -> str:
    output = template
    for idx, value in literal_values.items():
        output = output.replace(
            f"[m2_{idx}]",
            format_cypher_slot_value(value, literal_types.get(idx, "string")),
        )
    return output


def _top_k(counter: Counter, k: int = 20) -> dict[str, int]:
    return {str(key): value for key, value in counter.most_common(k)}


def _is_db_sampling(method: str) -> bool:
    return method in {"structure_db", "slot_db"}


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
    rejected_invalid_cypher = 0
    max_total_attempts = max(cfg.num_queries * max(cfg.max_retries, 1) * 5, cfg.num_queries)

    sampler = Neo4jSampler(
        cfg.db_config_file,
        db_group=cfg.db_group,
        connection_timeout=cfg.connection_timeout,
        query_timeout=cfg.query_timeout,
    )
    try:
        idx = 0
        while len(generated_queries) < cfg.num_queries and total_attempts < max_total_attempts:
            template = sampled_templates[idx % len(sampled_templates)]
            idx += 1
            total_attempts += 1

            question_template = _choose_question_template(template.get("question_semi_template"))
            cypher_template = template.get("cypher_semi_template", "")
            literals = template.get("literals", [])
            literal_types = _literal_type_by_index(literals)
            masking_cnt = _count_masks(question_template, cypher_template)

            question = question_template
            cypher = cypher_template
            literal_values: dict[int, Any] = {}
            sampling_method = "none"
            has_masking = "[m2_" in question_template or "[m2_" in cypher_template

            if has_masking and literals:
                actual_db = template.get("source_db", cfg.target_db)
                succeeded = False

                for _ in range(cfg.max_retries):
                    sampled_values, method, failed_slots = sampler.sample_literals(
                        actual_db,
                        template,
                        literals_data,
                        sample_pool_size=cfg.sample_pool_size,
                    )
                    if failed_slots:
                        continue
                    if cfg.require_db_sampling and not _is_db_sampling(method):
                        rejected_non_db_sampling += 1
                        continue

                    candidate_question = _apply_text_literals(question_template, sampled_values)
                    candidate_cypher = _apply_cypher_literals(
                        cypher_template,
                        sampled_values,
                        literal_types,
                    )
                    if cfg.validate_generated and not sampler.validate_cypher(actual_db, candidate_cypher):
                        rejected_invalid_cypher += 1
                        continue

                    question = candidate_question
                    cypher = candidate_cypher
                    literal_values = sampled_values
                    sampling_method = method
                    succeeded = True
                    break

                if not succeeded:
                    total_failed_generations += 1
                    continue
            elif cfg.validate_generated and not sampler.validate_cypher(template.get("source_db", cfg.target_db), cypher):
                rejected_invalid_cypher += 1
                total_failed_generations += 1
                continue

            query_obj = {
                "id": len(generated_queries) + 1,
                "template_id": template.get("template_id"),
                "question_semi_template": question_template,
                "question": question,
                "cypher_semi_template": cypher_template,
                "cypher": cypher,
                "sampling_method": sampling_method,
                "literal_values": {str(k): stringify_value(v) for k, v in literal_values.items()},
                "masking_cnt": masking_cnt,
                "iteration": total_attempts,
            }
            if "source_db" in template:
                query_obj["source_db"] = template["source_db"]
            if "source_split" in template:
                query_obj["source_split"] = template["source_split"]

            generated_queries.append(query_obj)
            generated_template_counter[template.get("template_id")] += 1
            sampling_method_counter[sampling_method] += 1
            masking_counter[masking_cnt] += 1

    finally:
        sampler.close()

    if len(generated_queries) != cfg.num_queries:
        raise RuntimeError(
            f"Generated only {len(generated_queries)}/{cfg.num_queries} queries after "
            f"{total_attempts} attempts; check Neo4j connections, templates and literal sampling. No workload saved."
        )

    statistics = {
        "generated_queries": len(generated_queries),
        "failed_generations": total_failed_generations,
        "total_attempts": total_attempts,
        "rejected_non_db_sampling": rejected_non_db_sampling,
        "rejected_invalid_cypher": rejected_invalid_cypher,
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
            "db_group": cfg.db_group,
            "distribution_type": cfg.distribution_type,
            "criterion": cfg.criterion if cfg.distribution_type == "zipf" else None,
            "alpha": cfg.alpha if cfg.distribution_type == "zipf" else None,
            "power_law_s": cfg.power_law_s if cfg.criterion == "query_len" else None,
            "max_retries": cfg.max_retries,
            "sampling_mode": "db" if cfg.require_db_sampling else "hybrid",
            "validate_generated": cfg.validate_generated,
            "sample_pool_size": cfg.sample_pool_size,
            "num_queries": cfg.num_queries,
            "seed": cfg.seed,
        },
        "statistics": statistics,
        "queries": generated_queries,
    }

    output_payload = {
        "config": result["config"],
        "queries": result["queries"],
    }
    dump_json(output_payload, Path(cfg.output_file))
    return result
