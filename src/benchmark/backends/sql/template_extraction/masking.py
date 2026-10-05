from __future__ import annotations

import re
from typing import Any, Callable

from sqlglot import TokenType, Tokenizer, expressions as exp


ComparisonExpr = (exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.NEQ)
MASK_PREFIX = "m2"


def get_literal_type(literal_node: exp.Literal, original_sql: str) -> str:
    value_str = str(literal_node.this)
    if f"'{value_str}'" in original_sql or f'"{value_str}"' in original_sql:
        return "str"
    try:
        if "." in value_str:
            float(value_str)
            return "float"
        int(value_str)
        return "int"
    except ValueError:
        return "str"


def collect_sql_literals(sql_query: str, dialect: str, case_sensitive: bool) -> set[str]:
    literals: set[str] = set()
    for tok in Tokenizer(dialect=dialect).tokenize(sql_query):
        if tok.token_type in {TokenType.STRING, TokenType.NUMBER, TokenType.BOOLEAN}:
            value = tok.text.strip("'\"")
            literals.add(value if case_sensitive else value.lower())
    return literals


def collect_text_literals(text: str) -> set[str]:
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    string_matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)

    values = set(numbers)
    for left, right in string_matches:
        value = left or right
        if value.strip():
            values.add(value)
    return values


def extract_literals_with_columns(
    tree: exp.Expression,
    target_literals: set[str],
    original_sql: str,
    resolve_column: Callable[[exp.Column], str],
    case_sensitive: bool,
) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    seen_values: set[str] = set()

    def normalize(v: str) -> str:
        return v if case_sensitive else v.lower()

    for comp in tree.find_all(ComparisonExpr):
        column_node = None
        literal_node = None

        if isinstance(comp.left, exp.Column) and isinstance(comp.right, exp.Literal):
            column_node, literal_node = comp.left, comp.right
        elif isinstance(comp.right, exp.Column) and isinstance(comp.left, exp.Literal):
            column_node, literal_node = comp.right, comp.left

        if not column_node or not literal_node:
            continue

        literal_value = str(literal_node.this)
        if normalize(literal_value) not in target_literals or literal_value in seen_values:
            continue

        extracted.append(
            {
                "column_name": resolve_column(column_node),
                "value": literal_node.this,
                "original_type": get_literal_type(literal_node, original_sql),
            }
        )
        seen_values.add(literal_value)

    for in_expr in tree.find_all(exp.In):
        if not isinstance(in_expr.this, exp.Column):
            continue
        column_node = in_expr.this

        for literal_node in in_expr.expressions:
            if not isinstance(literal_node, exp.Literal):
                continue
            literal_value = str(literal_node.this)
            if normalize(literal_value) not in target_literals or literal_value in seen_values:
                continue

            extracted.append(
                {
                    "column_name": resolve_column(column_node),
                    "value": literal_node.this,
                    "original_type": get_literal_type(literal_node, original_sql),
                }
            )
            seen_values.add(literal_value)

    return extracted


def create_indexed_template(
    text: str,
    literals: list[dict[str, Any]],
    case_sensitive: bool,
) -> str:
    if not text:
        return ""

    template = text
    indexed_literals = sorted(
        enumerate(literals),
        key=lambda item: len(str(item[1]["value"])),
        reverse=True,
    )

    for idx, item in indexed_literals:
        original_value = str(item["value"])
        pattern = r"\b" + re.escape(original_value) + r"\b"
        mask = f"[{MASK_PREFIX}_{idx}]"
        flags = 0 if case_sensitive else re.IGNORECASE
        template = re.sub(pattern, mask, template, flags=flags)

    return template


def create_indexed_sql_template(
    tree: exp.Expression,
    literals: list[dict[str, Any]],
    dialect: str,
    case_sensitive: bool,
) -> str:
    result_sql = tree.sql(dialect=dialect)

    sorted_literals = sorted(
        enumerate(literals),
        key=lambda item: len(str(item[1]["value"])),
        reverse=True,
    )

    for idx, literal_info in sorted_literals:
        original_value = str(literal_info["value"])
        original_type = literal_info.get("original_type", "str")
        mask = f"[{MASK_PREFIX}_{idx}]"

        if original_type in {"int", "float"}:
            pattern = re.escape(original_value)
            flags = 0 if case_sensitive else re.IGNORECASE
            result_sql = re.sub(pattern, mask, result_sql, flags=flags)
            continue

        if case_sensitive:
            result_sql = result_sql.replace(f"'{original_value}'", f"'{mask}'")
            result_sql = result_sql.replace(f'"{original_value}"', f'"{mask}"')
        else:
            result_sql = re.sub(re.escape(f"'{original_value}'"), f"'{mask}'", result_sql, flags=re.IGNORECASE)
            result_sql = re.sub(re.escape(f'"{original_value}"'), f'"{mask}"', result_sql, flags=re.IGNORECASE)

    return result_sql
