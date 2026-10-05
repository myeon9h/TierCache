"""Neo4j operations used by the Cypher workload sampler."""
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any


@dataclass
class Neo4jConnectionInfo:
    name: str
    uri: str
    username: str
    password: str


def make_driver(info: Neo4jConnectionInfo, *, timeout: float = 2.0) -> Any:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        info.uri, auth=(info.username, info.password), connection_timeout=timeout,
    )
    try:
        driver.verify_connectivity()
    except Exception:
        driver.close()
        raise
    return driver


def run_cypher(driver, cypher, *, timeout=10.0, max_records=None):
    from neo4j import Query, READ_ACCESS

    rows = []
    with driver.session(default_access_mode=READ_ACCESS) as session:
        result = session.run(Query(cypher, timeout=timeout))
        for record in result:
            rows.append({key: stringify_value(value) for key, value in record.data().items()})
            if max_records is not None and len(rows) >= max_records:
                break
    return rows


def stringify_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, list):
        return [stringify_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): stringify_value(val) for key, val in value.items()}
    return str(value)


def row_scalar(row):
    if "value" in row:
        return row["value"]
    return next(iter(row.values())) if len(row) == 1 else row


def format_cypher_slot_value(value, literal_type):
    if value is None:
        return ""
    if literal_type in {"number", "float", "int"}:
        return str(value)
    return str(value).replace("\\", "\\\\").replace("'", "\\'")
