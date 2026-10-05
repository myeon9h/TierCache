"""Validate the shared workload envelope and language-specific gold query."""
import json


def load_workload(path, structured_query_type):
    if structured_query_type not in {"sql", "cypher"}:
        raise ValueError(f"Unsupported structured query type: {structured_query_type}")
    with open(path, encoding="utf-8") as handle:
        workload = json.load(handle)
    if not isinstance(workload, dict) or not isinstance(workload.get("queries"), list):
        raise ValueError("Workload must contain a queries array")
    if not workload["queries"]:
        raise ValueError("Workload queries array is empty")
    for index, row in enumerate(workload["queries"], start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Query {index} must be an object")
        for field in ("question", structured_query_type):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Query {index}: missing or empty {field} string")
        if "evidence" in row and row["evidence"] is not None and not isinstance(row["evidence"], str):
            raise ValueError(f"Query {index}: evidence must be a string or null")
    return workload
