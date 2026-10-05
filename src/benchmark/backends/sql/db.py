from __future__ import annotations

import random
import sqlite3
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

try:
    import psycopg
except ImportError:  # pragma: no cover - optional dependency
    psycopg = None


from benchmark.paths import REPO_ROOT


def _resolve_repo_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


class DBSampler:
    def __init__(self, db_config_file: Path | str):
        path = _resolve_repo_path(db_config_file)
        if not path.exists():
            raise FileNotFoundError(f"Missing DB config: {path}")

        with path.open("r", encoding="utf-8") as f:
            self.db_configs: dict[str, dict[str, Any]] = yaml.safe_load(f) or {}

        self._connections: dict[str, Any] = {}

    def close(self) -> None:
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()

    def _get_conn(self, target_db: str):
        if target_db in self._connections:
            return self._connections[target_db]

        if target_db not in self.db_configs:
            raise KeyError(f"Database is not configured: {target_db}")

        cfg = self.db_configs[target_db]
        db_type = cfg.get("type")

        if db_type == "sqlite":
            sqlite_path = _resolve_repo_path(str(cfg["path"]))
            if not sqlite_path.is_file():
                raise FileNotFoundError(f"Missing SQLite database: {sqlite_path}")
            conn = sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)
        elif db_type == "postgresql":
            if psycopg is None:
                raise ImportError("psycopg is required for PostgreSQL")
            conn = psycopg.connect(cfg["url"], autocommit=True)
            conn.execute("SET default_transaction_read_only = on")
            schema = str(cfg.get("schema", "public"))
            from psycopg import sql
            conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        else:
            raise ValueError(f"Unsupported DB type: {db_type}")

        self._connections[target_db] = conn
        return conn

    def _split_column(self, column_name: str) -> tuple[str, str] | None:
        if "." not in column_name:
            return None
        table, column = column_name.split(".", 1)
        if not table or not column:
            return None
        return table, column

    def sample_column_value(self, target_db: str, column_name: str) -> Any | None:
        parsed = self._split_column(column_name)
        if not parsed:
            return None

        table_name, col_name = parsed
        cfg = self.db_configs.get(target_db, {})
        db_type = cfg.get("type")
        conn = self._get_conn(target_db)

        try:
            if db_type == "sqlite":
                query = (
                    f"SELECT DISTINCT {col_name} FROM {table_name} "
                    f"WHERE {col_name} IS NOT NULL ORDER BY RANDOM() LIMIT 1"
                )
                cur = conn.cursor()
                cur.execute(query)
                row = cur.fetchone()
                cur.close()
            else:
                schema = cfg.get("schema", "public")
                query = (
                    f"SELECT {col_name} FROM {schema}.{table_name} "
                    f"WHERE {col_name} IS NOT NULL ORDER BY RANDOM() LIMIT 1"
                )
                row = conn.execute(query).fetchone()

            if not row:
                return None
            return to_jsonable(row[0])

        except Exception:
            if db_type == "postgresql":
                try:
                    conn.rollback()
                except Exception:
                    pass
            return None

    def validate_sql(self, target_db: str, sql_query: str) -> bool:
        cfg = self.db_configs.get(target_db, {})
        db_type = cfg.get("type")

        try:
            conn = self._get_conn(target_db)
            if db_type == "sqlite":
                cur = conn.cursor()
                cur.execute(sql_query)
                row = cur.fetchone()
                cur.close()
                return row is not None

            row = conn.execute(sql_query).fetchone()
            return row is not None

        except Exception:
            if db_type == "postgresql":
                try:
                    self._get_conn(target_db).rollback()
                except Exception:
                    pass
            return False

    def sample_literals(
        self,
        target_db: str,
        literals: list[dict[str, Any]],
        literals_data: dict[str, Any] | None,
    ) -> tuple[dict[int, Any], str, list[str]]:
        sampled: dict[int, Any] = {}
        unresolved: list[int] = []

        for idx, literal in enumerate(literals):
            value = self.sample_column_value(target_db, literal.get("column_name", ""))
            if value is None:
                unresolved.append(idx)
            else:
                sampled[idx] = value

        source = "db" if len(sampled) == len(literals) else "db_partial"

        if unresolved and literals_data:
            for idx in unresolved[:]:
                column = literals[idx].get("column_name", "")
                candidates = literals_data.get(column, {}).get("existing_values", [])
                if candidates:
                    sampled[idx] = random.choice(candidates)
                    unresolved.remove(idx)
            if source == "db_partial":
                source = "hybrid_existing"
            elif not unresolved:
                source = "existing"

        if unresolved:
            for idx in unresolved[:]:
                example_value = literals[idx].get("example_value", literals[idx].get("value"))
                if example_value is not None:
                    sampled[idx] = example_value
                    unresolved.remove(idx)

            if source in {"db_partial", "hybrid_existing"}:
                source = "hybrid_example"
            elif source == "db":
                source = "example_value"
            else:
                source = "example_value"

        failed_columns = [literals[idx].get("column_name", "") for idx in unresolved]
        return sampled, source, failed_columns
