from __future__ import annotations

import random
from typing import Any

import numpy as np


def sample_templates_uniform(templates: list[dict[str, Any]], num_samples: int) -> list[dict[str, Any]]:
    if not templates:
        return []
    return random.choices(templates, k=num_samples)


def _question_len(template: dict[str, Any]) -> int:
    value = template.get("question_semi_template", "")
    if isinstance(value, list):
        if not value:
            return 0
        return len(value[0])
    return len(value or "")


def sample_templates_zipf(
    templates: list[dict[str, Any]],
    num_samples: int,
    *,
    criterion: str = "rank",
    alpha: float = 1.0,
    power_law_s: float = 2.0,
) -> list[dict[str, Any]]:
    if not templates:
        return []

    if criterion == "random":
        ordered = templates.copy()
        random.shuffle(ordered)
    elif criterion == "rank":
        ordered = templates
    elif criterion == "query_len":
        ordered = sorted(templates, key=_question_len)
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")

    n = len(ordered)
    if criterion == "query_len":
        lengths = np.array([max(_question_len(t), 1) for t in ordered], dtype=float)
        probs = lengths ** (-power_law_s)
    else:
        ranks = np.arange(1, n + 1, dtype=float)
        probs = 1.0 / (ranks ** alpha)

    probs = probs / probs.sum()
    sampled_idx = np.random.choice(np.arange(n), size=num_samples, p=probs, replace=True)
    return [ordered[i] for i in sampled_idx]
