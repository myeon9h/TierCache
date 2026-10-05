#!/usr/bin/env python3
"""Create aligned literal masks for CypherBench NLQ/Cypher pairs.

This script masks only literals that appear exactly in both `nl_question` and
`gold_cypher`. It also records enough metadata to sample replacement values
from Neo4j later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


PLACEHOLDER_PREFIX = "LITERAL"


@dataclass
class CypherLiteral:
    value: str
    literal_type: str
    cypher_span: tuple[int, int]
    raw_expression_span: tuple[int, int]
    raw_expression: str


@dataclass
class LiteralContext:
    variable: str | None = None
    label: str | None = None
    relation_type: str | None = None
    property: str | None = None
    owner_type: str | None = None
    operator: str | None = None
    expression: str | None = None


@dataclass
class LiteralSlot:
    slot: str
    value: str
    literal_type: str
    nl_spans: list[tuple[int, int]] = field(default_factory=list)
    cypher_spans: list[tuple[int, int]] = field(default_factory=list)
    raw_expressions: list[str] = field(default_factory=list)
    contexts: list[LiteralContext] = field(default_factory=list)
    sample_query: str | None = None


@dataclass
class MaskingResult:
    record: dict[str, Any]
    slots: list[LiteralSlot]


def is_inside(span: tuple[int, int], containers: list[tuple[int, int]]) -> bool:
    return any(start <= span[0] and span[1] <= end for start, end in containers)


def find_exact_spans(text: str, value: str, *, ignore_case: bool = False) -> list[tuple[int, int]]:
    if not value:
        return []

    flags = re.IGNORECASE if ignore_case else 0
    spans: list[tuple[int, int]] = []
    for match in re.finditer(re.escape(value), text, flags):
        start, end = match.span()
        if not has_token_boundary(text, start, end, value):
            continue
        spans.append((start, end))
    return spans


def has_token_boundary(text: str, start: int, end: int, value: str) -> bool:
    """Avoid matching short literals inside larger words, e.g. `US` in `USA`."""
    if value[0].isalnum() and start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
        return False
    if value[-1].isalnum() and end < len(text) and (text[end].isalnum() or text[end] == "_"):
        return False
    return True


# ---------------------------------------------------------------------------
# Masking phase
# ---------------------------------------------------------------------------


def extract_cypher_literals(cypher: str) -> list[CypherLiteral]:
    literals: list[CypherLiteral] = []
    occupied: list[tuple[int, int]] = []

    # Treat date('YYYY-MM-DD') as one logical literal. The replacement span is
    # only the date string so the Cypher keeps date('[LITERAL_0]').
    date_pattern = re.compile(r"date\('(?P<value>\d{4}-\d{2}-\d{2})'\)")
    for match in date_pattern.finditer(cypher):
        value_span = match.span("value")
        literals.append(
            CypherLiteral(
                value=match.group("value"),
                literal_type="date",
                cypher_span=value_span,
                raw_expression_span=match.span(),
                raw_expression=match.group(0),
            )
        )
        occupied.append(match.span())

    string_pattern = re.compile(r"'(?P<value>(?:\\'|[^'])*)'")
    for match in string_pattern.finditer(cypher):
        if is_inside(match.span(), occupied):
            continue
        value = match.group("value").replace("\\'", "'")
        literals.append(
            CypherLiteral(
                value=value,
                literal_type="string",
                cypher_span=match.span("value"),
                raw_expression_span=match.span(),
                raw_expression=match.group(0),
            )
        )
        occupied.append(match.span())

    number_pattern = re.compile(r"(?<![\w.])-?\b\d+(?:\.\d+)?\b")
    for match in number_pattern.finditer(cypher):
        if is_inside(match.span(), occupied):
            continue
        if is_limit_one(cypher, match):
            continue
        value = match.group(0)
        literal_type = "float" if "." in value else "number"
        literals.append(
            CypherLiteral(
                value=value,
                literal_type=literal_type,
                cypher_span=match.span(),
                raw_expression_span=match.span(),
                raw_expression=value,
            )
        )

    return sorted(literals, key=lambda item: item.cypher_span)


def is_limit_one(cypher: str, match: re.Match[str]) -> bool:
    if match.group(0) != "1":
        return False
    prefix = cypher[max(0, match.start() - 16) : match.start()]
    return bool(re.search(r"\bLIMIT\s*$", prefix, flags=re.IGNORECASE))


# ---------------------------------------------------------------------------
# Sampling phase
# ---------------------------------------------------------------------------


def parse_variable_labels(cypher: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for match in re.finditer(r"\((?P<var>[A-Za-z]\w*)\s*:\s*(?P<label>[A-Za-z]\w*)", cypher):
        labels.setdefault(match.group("var"), match.group("label"))
    return labels


def parse_relation_types(cypher: str) -> dict[str, str]:
    rels: dict[str, str] = {}
    for match in re.finditer(r"\[(?P<var>[A-Za-z]\w*)\s*:\s*(?P<label>[A-Za-z]\w*)", cypher):
        rels.setdefault(match.group("var"), match.group("label"))
    return rels


def extract_property_from_map(body: str, relative_literal_start: int) -> str | None:
    prefix = body[:relative_literal_start]
    match = re.search(r"([A-Za-z_]\w*)\s*:\s*(?:date\()?['\"]?$", prefix)
    return match.group(1) if match else None


def infer_literal_context(cypher: str, literal: CypherLiteral) -> LiteralContext:
    labels = parse_variable_labels(cypher)
    rels = parse_relation_types(cypher)
    start, end = literal.cypher_span

    node_context = infer_property_map_context(
        cypher,
        start,
        pattern=re.compile(r"\((?P<var>[A-Za-z]\w*)\s*:\s*(?P<label>[A-Za-z]\w*)\s*\{(?P<body>[^}]*)\}\)"),
        owner_type="node",
    )
    if node_context:
        return node_context

    rel_context = infer_property_map_context(
        cypher,
        start,
        pattern=re.compile(r"\[(?P<var>[A-Za-z]\w*)\s*:\s*(?P<label>[A-Za-z]\w*)\s*\{(?P<body>[^\]]*)\}\]"),
        owner_type="relationship",
    )
    if rel_context:
        return rel_context

    comparison_context = infer_comparison_context(cypher, literal, labels, rels)
    if comparison_context:
        return comparison_context

    return LiteralContext()


def infer_property_map_context(
    cypher: str,
    literal_start: int,
    *,
    pattern: re.Pattern[str],
    owner_type: str,
) -> LiteralContext | None:
    for match in pattern.finditer(cypher):
        body_start = match.start("body")
        body_end = match.end("body")
        if not (body_start <= literal_start <= body_end):
            continue
        prop = extract_property_from_map(match.group("body"), literal_start - body_start)
        label = match.group("label")
        var = match.group("var")
        if owner_type == "node":
            return LiteralContext(
                variable=var,
                label=label,
                property=prop,
                owner_type=owner_type,
                expression=f"{var}.{prop}" if prop else None,
            )
        return LiteralContext(
            variable=var,
            relation_type=label,
            property=prop,
            owner_type=owner_type,
            expression=f"{var}.{prop}" if prop else None,
        )
    return None


def infer_comparison_context(
    cypher: str,
    literal: CypherLiteral,
    labels: dict[str, str],
    rels: dict[str, str],
) -> LiteralContext | None:
    start, end = literal.cypher_span
    left_window = cypher[max(0, start - 160) : start]
    right_window = cypher[end : min(len(cypher), end + 160)]

    left_match = re.search(
        r"(?P<var>[A-Za-z]\w*)\.(?P<prop>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*(?P<op><=|>=|<>|!=|=|<|>|CONTAINS|STARTS\s+WITH|ENDS\s+WITH)\s*(?:date\()?['\"]?$",
        left_window,
        flags=re.IGNORECASE,
    )
    if left_match:
        return context_from_var_prop(left_match, labels, rels)

    right_match = re.match(
        r"['\"]?\)?\s*(?P<op>IN)\s+(?P<var>[A-Za-z]\w*)\.(?P<prop>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)",
        right_window,
        flags=re.IGNORECASE,
    )
    if right_match:
        return context_from_var_prop(right_match, labels, rels)

    return None


def context_from_var_prop(
    match: re.Match[str],
    labels: dict[str, str],
    rels: dict[str, str],
) -> LiteralContext:
    var = match.group("var")
    prop = match.group("prop")
    op = re.sub(r"\s+", " ", match.group("op").upper())
    if var in labels:
        return LiteralContext(
            variable=var,
            label=labels[var],
            property=prop,
            owner_type="node",
            operator=op,
            expression=f"{var}.{prop}",
        )
    if var in rels:
        return LiteralContext(
            variable=var,
            relation_type=rels[var],
            property=prop,
            owner_type="relationship",
            operator=op,
            expression=f"{var}.{prop}",
        )
    return LiteralContext(variable=var, property=prop, operator=op, expression=f"{var}.{prop}")


def make_sample_query(context: LiteralContext, literal_type: str, *, limit: int) -> str | None:
    if not context.property:
        return None

    if context.owner_type == "node" and context.label:
        accessor = f"x.{context.property}"
        if context.operator == "IN":
            return (
                f"MATCH (x:{context.label}) "
                f"WHERE {accessor} IS NOT NULL "
                f"UNWIND {accessor} AS value "
                f"RETURN DISTINCT value AS value LIMIT {limit}"
            )
        return (
            f"MATCH (x:{context.label}) "
            f"WHERE {accessor} IS NOT NULL "
            f"RETURN DISTINCT {sample_return_expr(accessor, literal_type)} AS value LIMIT {limit}"
        )

    if context.owner_type == "relationship" and context.relation_type:
        accessor = f"r.{context.property}"
        if context.operator == "IN":
            return (
                f"MATCH ()-[r:{context.relation_type}]-() "
                f"WHERE {accessor} IS NOT NULL "
                f"UNWIND {accessor} AS value "
                f"RETURN DISTINCT value AS value LIMIT {limit}"
            )
        return (
            f"MATCH ()-[r:{context.relation_type}]-() "
            f"WHERE {accessor} IS NOT NULL "
            f"RETURN DISTINCT {sample_return_expr(accessor, literal_type)} AS value LIMIT {limit}"
        )

    return None


def make_structure_sampling_query(
    masked_cypher: str,
    slots: list[LiteralSlot],
    *,
    limit: int,
) -> tuple[str | None, list[dict[str, str]]]:
    usable_slots = [slot for slot in slots if slot.contexts and slot.contexts[0].expression]
    if not usable_slots:
        return None, []
    if masked_cypher.lstrip().upper().startswith("CALL"):
        return None, []

    structural_cypher = extract_structural_cypher(masked_cypher)
    structural_cypher = remove_placeholder_property_maps(structural_cypher)
    structural_cypher = neutralize_placeholder_predicates(structural_cypher)
    structural_cypher = cleanup_cypher_spacing(structural_cypher)
    if not structural_cypher:
        return None, []

    conditions: list[str] = []
    returns: list[str] = []
    sampling_returns: list[dict[str, str]] = []
    seen_conditions: set[str] = set()
    seen_returns: set[str] = set()

    for slot in usable_slots:
        context = slot.contexts[0]
        if not context.expression:
            continue
        expr = context.expression
        non_null = f"{expr} IS NOT NULL"
        if non_null not in seen_conditions:
            conditions.append(non_null)
            seen_conditions.add(non_null)

        alias = slot.slot.strip("[]").lower()
        return_expr = sample_return_expr(expr, slot.literal_type)
        return_clause = f"{return_expr} AS {alias}"
        if return_clause not in seen_returns:
            returns.append(return_clause)
            seen_returns.add(return_clause)
            sampling_returns.append({"slot": slot.slot, "alias": alias, "expression": expr})

    for left_idx, left in enumerate(usable_slots):
        left_expr = left.contexts[0].expression
        if not left_expr:
            continue
        for right in usable_slots[left_idx + 1 :]:
            right_expr = right.contexts[0].expression
            if not right_expr:
                continue
            if left.literal_type != right.literal_type:
                continue
            condition = f"{left_expr} <> {right_expr}"
            if condition not in seen_conditions:
                conditions.append(condition)
                seen_conditions.add(condition)

    if not returns:
        return None, []

    where_prefix = " AND " if has_where_clause(structural_cypher) else " WHERE "
    query = structural_cypher
    if conditions:
        query += where_prefix + " AND ".join(conditions)
    query += " RETURN DISTINCT " + ", ".join(returns) + f" LIMIT {limit}"
    return query, sampling_returns


def extract_structural_cypher(masked_cypher: str) -> str:
    match = re.search(r"\s+WITH\s+|\s+RETURN\s+", masked_cypher, flags=re.IGNORECASE)
    if not match:
        return masked_cypher.strip()
    return masked_cypher[: match.start()].strip()


def remove_placeholder_property_maps(cypher: str) -> str:
    # Common benchmark slot form: (m0:Label {name: '[LITERAL_0]'}) -> (m0:Label)
    cypher = re.sub(
        r"\s*\{\s*[A-Za-z_]\w*\s*:\s*(?:date\()?['\"]?\[LITERAL_\d+\]['\"]?\)?\s*\}",
        "",
        cypher,
    )
    # Best-effort cleanup if a future property map contains multiple entries.
    cypher = re.sub(
        r"\{\s*([A-Za-z_]\w*\s*:\s*(?:date\()?['\"]?\[LITERAL_\d+\]['\"]?\)?\s*,\s*)+",
        "{",
        cypher,
    )
    cypher = re.sub(
        r",\s*[A-Za-z_]\w*\s*:\s*(?:date\()?['\"]?\[LITERAL_\d+\]['\"]?\)?\s*(?=})",
        "",
        cypher,
    )
    return cypher


def neutralize_placeholder_predicates(cypher: str) -> str:
    literal = r"(?:date\()?['\"]?\[LITERAL_\d+\]['\"]?\)?"
    prop = r"[A-Za-z]\w*(?:\.[A-Za-z_]\w*)+"
    cypher = re.sub(
        rf"{prop}\s*(?:<=|>=|<>|!=|=|<|>|CONTAINS|STARTS\s+WITH|ENDS\s+WITH)\s*{literal}",
        "TRUE",
        cypher,
        flags=re.IGNORECASE,
    )
    cypher = re.sub(
        rf"{literal}\s+IN\s+{prop}",
        "TRUE",
        cypher,
        flags=re.IGNORECASE,
    )
    return cypher


def cleanup_cypher_spacing(cypher: str) -> str:
    cypher = re.sub(r"\s+", " ", cypher).strip()
    cypher = re.sub(r"WHERE\s+TRUE\s+AND\s+", "WHERE ", cypher, flags=re.IGNORECASE)
    cypher = re.sub(r"WHERE\s+TRUE\s*$", "", cypher, flags=re.IGNORECASE).strip()
    cypher = re.sub(r"\(\s+", "(", cypher)
    cypher = re.sub(r"\s+\)", ")", cypher)
    return cypher


def has_where_clause(cypher: str) -> bool:
    return bool(re.search(r"\bWHERE\b", cypher, flags=re.IGNORECASE))


def sample_return_expr(accessor: str, literal_type: str) -> str:
    if literal_type == "date":
        return f"toString({accessor})"
    return accessor


def replace_spans(text: str, replacements: list[tuple[int, int, str]]) -> str:
    out = text
    for start, end, value in sorted(replacements, key=lambda item: item[0], reverse=True):
        out = out[:start] + value + out[end:]
    return out


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def filter_available_spans(
    spans: list[tuple[int, int]],
    reserved: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    return [span for span in spans if not any(spans_overlap(span, taken) for taken in reserved)]


def context_to_dict(context: LiteralContext) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "variable": context.variable,
            "label": context.label,
            "relation_type": context.relation_type,
            "property": context.property,
            "owner_type": context.owner_type,
            "operator": context.operator,
            "expression": context.expression,
        }.items()
        if value is not None
    }


def slot_to_dict(slot: LiteralSlot, *, include_sampling: bool) -> dict[str, Any]:
    data = {
        "slot": slot.slot,
        "value": slot.value,
        "type": slot.literal_type,
        "raw_expressions": slot.raw_expressions,
    }
    if include_sampling:
        data["contexts"] = [context_to_dict(context) for context in slot.contexts]
        data["sample_query"] = slot.sample_query
    return data


def apply_masking_phase(
    record: dict[str, Any],
    *,
    ignore_case: bool = False,
    numbered_placeholders: bool = True,
) -> MaskingResult:
    question = record["nl_question"]
    cypher = record["gold_cypher"]
    candidates = extract_cypher_literals(cypher)

    slots: list[LiteralSlot] = []
    slot_by_key: dict[tuple[str, str], LiteralSlot] = {}
    reserved_nl_spans: list[tuple[int, int]] = []

    # Longer literals must claim their NLQ spans first. Otherwise a literal like
    # `Oakpont` can corrupt the span already needed by `Oakpont Professionals`.
    prioritized_candidates = sorted(
        candidates,
        key=lambda item: (-len(item.value), item.cypher_span[0]),
    )

    for candidate in prioritized_candidates:
        all_nl_spans = find_exact_spans(question, candidate.value, ignore_case=ignore_case)
        nl_spans = filter_available_spans(all_nl_spans, reserved_nl_spans)
        if not nl_spans:
            continue

        key = (candidate.literal_type, candidate.value)
        if key not in slot_by_key:
            index = len(slots)
            slot_name = f"[{PLACEHOLDER_PREFIX}_{index}]" if numbered_placeholders else f"[{PLACEHOLDER_PREFIX}]"
            slot = LiteralSlot(
                slot=slot_name,
                value=candidate.value,
                literal_type=candidate.literal_type,
                nl_spans=nl_spans,
            )
            slot_by_key[key] = slot
            slots.append(slot)
            reserved_nl_spans.extend(nl_spans)

        slot = slot_by_key[key]
        slot.cypher_spans.append(candidate.cypher_span)
        slot.raw_expressions.append(candidate.raw_expression)

    slots = sorted(slots, key=lambda slot: min(slot.cypher_spans) if slot.cypher_spans else (10**9, 10**9))
    for index, slot in enumerate(slots):
        slot.slot = f"[{PLACEHOLDER_PREFIX}_{index}]" if numbered_placeholders else f"[{PLACEHOLDER_PREFIX}]"

    nl_replacements: list[tuple[int, int, str]] = []
    cypher_replacements: list[tuple[int, int, str]] = []
    for slot in slots:
        for start, end in slot.nl_spans:
            nl_replacements.append((start, end, slot.slot))
        for start, end in slot.cypher_spans:
            cypher_replacements.append((start, end, slot.slot))

    output = dict(record)
    output["masked_question"] = replace_spans(question, nl_replacements)
    output["masked_cypher"] = replace_spans(cypher, cypher_replacements)
    output["literal_slots"] = [slot_to_dict(slot, include_sampling=False) for slot in slots]
    output["masking_policy"] = {
        "mode": "exact_common_literal",
        "ignore_case": ignore_case,
        "numbered_placeholders": numbered_placeholders,
    }
    return MaskingResult(record=output, slots=slots)


def apply_sampling_phase(
    result: MaskingResult,
    *,
    sample_limit: int = 100,
) -> dict[str, Any]:
    cypher = result.record["gold_cypher"]

    for slot in result.slots:
        slot.contexts = []
        slot.sample_query = None
        for span, raw_expression in zip(slot.cypher_spans, slot.raw_expressions):
            literal = CypherLiteral(
                value=slot.value,
                literal_type=slot.literal_type,
                cypher_span=span,
                raw_expression_span=span,
                raw_expression=raw_expression,
            )
            context = infer_literal_context(cypher, literal)
            slot.contexts.append(context)
            if not slot.sample_query:
                slot.sample_query = make_sample_query(context, slot.literal_type, limit=sample_limit)

    output = dict(result.record)
    output["literal_slots"] = [slot_to_dict(slot, include_sampling=True) for slot in result.slots]
    structure_query, sampling_returns = make_structure_sampling_query(
        output["masked_cypher"],
        result.slots,
        limit=sample_limit,
    )
    output["sampling_query"] = structure_query
    output["sampling_returns"] = sampling_returns
    output["sampling_policy"] = {
        "mode": "structure_context",
        "sample_limit": sample_limit,
        "slot_sample_query_mode": "property_context_fallback",
    }
    return output


def process_record(
    record: dict[str, Any],
    *,
    ignore_case: bool = False,
    sample_limit: int = 100,
    numbered_placeholders: bool = True,
    include_sampling: bool = True,
) -> dict[str, Any]:
    masking_result = apply_masking_phase(
        record,
        ignore_case=ignore_case,
        numbered_placeholders=numbered_placeholders,
    )
    if include_sampling:
        return apply_sampling_phase(masking_result, sample_limit=sample_limit)
    return masking_result.record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_slots = sum(len(item.get("literal_slots", [])) for item in records)
    rows_with_slots = sum(1 for item in records if item.get("literal_slots"))
    rows_without_slots = len(records) - rows_with_slots
    rows_with_sample_query = sum(
        1
        for item in records
        if any(slot.get("sample_query") for slot in item.get("literal_slots", []))
    )
    rows_with_structure_sampling_query = sum(1 for item in records if item.get("sampling_query"))
    return {
        "rows": len(records),
        "rows_with_slots": rows_with_slots,
        "rows_without_slots": rows_without_slots,
        "total_slots": total_slots,
        "rows_with_sample_query": rows_with_sample_query,
        "rows_with_structure_sampling_query": rows_with_structure_sampling_query,
    }
