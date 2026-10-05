from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from sqlglot import expressions as exp
from benchmark.utils.io import load_json

# table.col 의 column 해석

@dataclass
class SchemaIndex:
    table_names: list[str]
    columns_by_table: dict[str, set[str]]
    columns_to_tables: dict[str, list[str]]


class SchemaStore:
    """Lazy schema loader for ScienceBenchmark and BIRD."""

    def __init__(
        self,
        benchmark: str,
        source_root: Path,
        tables_json: str | None = None,
        sqlite_paths: dict[str, str | Path] | None = None,
    ):
        self.benchmark = benchmark
        self.source_root = source_root
        self.tables_json = tables_json
        self.sqlite_paths = sqlite_paths or {}
        self._cache: dict[str, SchemaIndex] = {}
        self._bird_raw: dict[str, dict[str, Any]] | None = None

    def get(self, db_id: str) -> SchemaIndex | None:
        if db_id in self._cache:
            return self._cache[db_id]

        schema = self._load_raw_schema(db_id)
        if not schema:
            candidates = [self.source_root / db_id / "db.sqlite",
                          self.source_root / db_id / f"{db_id}.sqlite"]
            schema = next((self._sqlite_schema(path) for path in candidates if path.is_file()), None)
        if not schema:
            return None

        index = build_schema_index(schema)
        self._cache[db_id] = index
        return index

    def _load_raw_schema(self, db_id: str) -> dict[str, Any] | None:
        if self.benchmark == "EHRSQL":
            per_db_table_path = self.source_root / db_id / "tables.json"
            if per_db_table_path.exists():
                data = load_json(per_db_table_path)
                if isinstance(data, list) and data:
                    first = data[0]
                    if isinstance(first, dict):
                        return first
                if isinstance(data, dict):
                    return data
            combined = self.source_root / (self.tables_json or "tables.json")
            if combined.is_file():
                rows = load_json(combined)
                if isinstance(rows, list):
                    return next((row for row in rows if row.get("db_id") == db_id), None)
            return None

        if self.benchmark == "ScienceBenchmark":
            base_db = db_id.split("_")[0] if "_" in db_id else db_id
            schema_path = self.source_root / base_db / "tables.json"
            if not schema_path.exists():
                return None
            data = load_json(schema_path)
            if isinstance(data, list) and data:
                return data[0]
            if isinstance(data, dict):
                return data
            return None

        if self.benchmark == "BIRD":
            per_db_table_path = self.source_root / db_id / "tables.json"
            if per_db_table_path.exists():
                data = load_json(per_db_table_path)
                if isinstance(data, list) and data:
                    first = data[0]
                    if isinstance(first, dict):
                        return first
                if isinstance(data, dict):
                    return data

            if self._bird_raw is None:
                if not self.tables_json:
                    return None
                table_path = self.source_root / self.tables_json
                if not table_path.exists():
                    return None
                rows = load_json(table_path)
                self._bird_raw = {
                    row.get("db_id"): row
                    for row in rows
                    if isinstance(row, dict) and row.get("db_id")
                }
            return self._bird_raw.get(db_id)

        return None

    def _sqlite_schema(self, sqlite_path: Path) -> dict[str, Any] | None:
        def _normalize_sqlite_type(type_name: str) -> str:
            t = (type_name or "").strip().lower()
            if "int" in t or "real" in t or "float" in t or "double" in t or "numeric" in t or "decimal" in t:
                return "number"
            if "date" in t or "time" in t:
                return "time"
            if "bool" in t:
                return "boolean"
            if "char" in t or "clob" in t or "text" in t:
                return "text"
            if "blob" in t:
                return "others"
            return "text"

        try:
            conn = sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
        except Exception:
            return None

        try:
            table_rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            table_names = [str(row[0]) for row in table_rows if row and row[0]]
            if not table_names:
                return None

            table_names = [str(t) for t in table_names]
            table_names_normalized = [name.replace("_", " ") for name in table_names]

            column_names_original: list[list[Any]] = [[-1, "*"]]
            column_names: list[list[Any]] = [[-1, "*"]]
            column_types: list[str] = ["text"]
            primary_keys: list[int] = []
            foreign_keys: list[list[int]] = []
            column_idx_map: dict[tuple[int, str], int] = {}

            for table_idx, table_name in enumerate(table_names):
                pragma_rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
                for row in pragma_rows:
                    if len(row) < 6:
                        continue
                    col_name = str(row[1]) if row[1] is not None else ""
                    if not col_name:
                        continue
                    col_type = str(row[2]) if row[2] is not None else ""
                    is_pk = int(row[5]) > 0

                    global_idx = len(column_names_original)
                    column_names_original.append([table_idx, col_name])
                    column_names.append([table_idx, col_name.replace("_", " ")])
                    column_types.append(_normalize_sqlite_type(col_type))
                    column_idx_map[(table_idx, col_name)] = global_idx

                    if is_pk:
                        primary_keys.append(global_idx)

                fk_rows = conn.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()
                for fk in fk_rows:
                    if len(fk) < 5:
                        continue
                    ref_table_name = str(fk[2]) if fk[2] is not None else ""
                    from_col = str(fk[3]) if fk[3] is not None else ""
                    to_col = str(fk[4]) if fk[4] is not None else ""
                    if not ref_table_name or not from_col or not to_col:
                        continue
                    if ref_table_name not in table_names:
                        continue

                    ref_table_idx = table_names.index(ref_table_name)
                    src_idx = column_idx_map.get((table_idx, from_col))
                    dst_idx = column_idx_map.get((ref_table_idx, to_col))
                    if src_idx is not None and dst_idx is not None:
                        foreign_keys.append([src_idx, dst_idx])

            return {
                "db_id": sqlite_path.stem,
                "table_names": table_names_normalized,
                "table_names_original": table_names,
                "column_names": column_names,
                "column_names_original": column_names_original,
                "column_types": column_types,
                "primary_keys": sorted(set(primary_keys)),
                "foreign_keys": foreign_keys,
            }
        except Exception:
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass


def build_schema_index(schema: dict[str, Any]) -> SchemaIndex:
    # Convert raw Spider/BIRD-style schema JSON into fast lookup maps.
    # Example input:
    #   table_names_original=["patient","visit"]
    #   column_names_original=[[-1,"*"], [0,"id"], [1,"id"], [1,"patient_id"]]
    # Example output:
    #   columns_to_tables["id"] == ["patient", "visit"]
    #   columns_by_table["visit"] == {"id", "patient_id"}

    table_names = schema.get("table_names_original") or schema.get("table_names") or []

    raw_columns = (
        schema.get("column_names_original")
        or schema.get("column_names")
        or []
    )

    columns_by_table: dict[str, set[str]] = {name: set() for name in table_names}
    columns_to_tables: dict[str, list[str]] = {}

    for table_id, column_name in raw_columns:
        if table_id == -1:
            continue
        if table_id >= len(table_names):
            continue
        table = table_names[table_id]
        columns_by_table.setdefault(table, set()).add(column_name)
        columns_to_tables.setdefault(column_name, [])
        if table not in columns_to_tables[column_name]:
            columns_to_tables[column_name].append(table)

    return SchemaIndex(
        table_names=table_names,
        columns_by_table=columns_by_table,
        columns_to_tables=columns_to_tables,
    )


def _node_alias_name(node: exp.Expression) -> str | None:
    # Extract alias name from sqlglot node.
    # Example: "FROM patient AS p" -> "p"
    alias = getattr(node, "alias", None)
    if not alias:
        return None
    if hasattr(alias, "name") and alias.name:
        return str(alias.name)
    return str(alias)


def _column_name(column_node: exp.Column) -> str:
    # Get bare column identifier from sqlglot Column node.
    # Example: "p.age" -> "age"
    if hasattr(column_node, "name") and column_node.name:
        return str(column_node.name)
    if hasattr(column_node, "this") and column_node.this:
        return str(column_node.this)
    return column_node.sql()


def _collect_alias_and_tables(tree: exp.Expression) -> tuple[dict[str, str], list[str]]:
    # Collect:
    # 1) alias -> table map, 2) table order as they appear in SQL.
    # Example:
    #   SQL: FROM patient AS p JOIN visit v ...
    #   returns ({"p":"patient","v":"visit"}, ["patient","visit"])
    alias_to_table: dict[str, str] = {}
    ordered_tables: list[str] = []

    for table_node in tree.find_all(exp.Table):
        table_name = str(table_node.this)
        if table_name not in ordered_tables:
            ordered_tables.append(table_name)

        alias = _node_alias_name(table_node)
        if alias:
            alias_to_table[alias] = table_name

    return alias_to_table, ordered_tables


def _find_context_tables(column_node: exp.Column) -> list[str]:
    # Find FROM tables in the nearest SELECT scope of this column.
    # Example:
    #   SELECT id FROM (SELECT id FROM visit) t
    #   For inner "id" -> ["visit"], for outer "id" -> ["t"]
    current = column_node.parent
    select_node = None

    while current:
        if isinstance(current, exp.Select):
            select_node = current
            break
        current = current.parent

    if not select_node:
        return []

    from_node = select_node.find(exp.From)
    if not from_node:
        return []

    tables: list[str] = []
    for table in from_node.find_all(exp.Table):
        tname = str(table.this)
        if tname not in tables:
            tables.append(tname)
    return tables


def resolve_column_with_schema(
    column_node: exp.Column,
    tree: exp.Expression,
    schema: SchemaIndex | None,
) -> str:
    # Resolve a column to "table.column" using:
    # Option 1) explicit alias/table on column
    # Option 2) no schema fallback
    # Option 3) schema-guided heuristics

    col_name = _column_name(column_node)
    alias_to_table, ordered_tables = _collect_alias_and_tables(tree)
    table_alias = str(column_node.table) if getattr(column_node, "table", None) else None

    # Option 1) Explicit table/alias on the column (highest confidence).
    if table_alias:
        actual_table = alias_to_table.get(table_alias, table_alias)
        return f"{actual_table}.{col_name}"

    # Option 2) No schema available.
    if not schema:
        if ordered_tables:
            return f"{ordered_tables[0]}.{col_name}"
        return column_node.sql()

    # Option 3) Schema-guided resolution.
    candidate_tables = schema.columns_to_tables.get(col_name, [])

    context_tables = _find_context_tables(column_node)
    for table in context_tables:
        if table in candidate_tables:
            return f"{table}.{col_name}"

    for table in ordered_tables:
        if table in candidate_tables:
            return f"{table}.{col_name}"

    if len(candidate_tables) == 1:
        return f"{candidate_tables[0]}.{col_name}"

    if ordered_tables:
        return f"{ordered_tables[0]}.{col_name}"

    return column_node.sql()
