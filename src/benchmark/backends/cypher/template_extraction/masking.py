from __future__ import annotations

import re


from benchmark.backends.cypher.template_extraction.literal_extractor import process_record, summarize


def to_m2_placeholder(slot: str, fallback_index: int = 0) -> str:
    match = re.search(r"\[(?:LITERAL|m2)_(\d+)\]", slot or "")
    if match:
        return f"[m2_{match.group(1)}]"
    if slot == "[LITERAL]":
        return f"[m2_{fallback_index}]"
    return slot or f"[m2_{fallback_index}]"


def placeholder_index(slot: str, fallback_index: int = 0) -> int:
    match = re.search(r"\[(?:LITERAL|m2)_(\d+)\]", slot or "")
    if match:
        return int(match.group(1))
    return fallback_index


def convert_literal_placeholders(template: str | None) -> str:
    if not template:
        return ""
    converted = re.sub(r"\[LITERAL_(\d+)\]", r"[m2_\1]", template)
    return converted.replace("[LITERAL]", "[m2_0]")
