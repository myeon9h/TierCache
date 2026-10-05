from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import yaml

from benchmark.utils.neo4j import (
    Neo4jConnectionInfo,
    make_driver,
    row_scalar,
    run_cypher,
)


from benchmark.paths import REPO_ROOT as PROJECT_ROOT


def _resolve_project_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _slot_index(slot: str, fallback: int = 0) -> int:
    match = re.search(r"\[m2_(\d+)\]", slot or "")
    return int(match.group(1)) if match else fallback


def _literal_key(literal: dict[str, Any]) -> str:
    contexts = literal.get("contexts") or []
    context = contexts[0] if contexts and isinstance(contexts[0], dict) else {}
    prop = context.get("property")
    if prop and context.get("owner_type") == "node" and context.get("label"):
        return f"node:{context['label']}.{prop}"
    if prop and context.get("owner_type") == "relationship" and context.get("relation_type"):
        return f"relationship:{context['relation_type']}.{prop}"
    if context.get("expression"):
        return f"expression:{context['expression']}"
    return str(literal.get("slot", literal.get("index", "unknown")))


class Neo4jSampler:
    def __init__(
        self,
        db_config_file: Path | str,
        *,
        db_group: str = "full",
        connection_timeout: float = 2.0,
        query_timeout: float = 10.0,
    ) -> None:
        path = _resolve_project_path(db_config_file)
        if not path.exists():
            raise FileNotFoundError(f"Missing Neo4j config: {path}")
        with path.open("r", encoding="utf-8") as f:
            self.db_configs: dict[str, Any] = yaml.safe_load(f) or {}
        self.db_group = db_group
        self.connection_timeout = connection_timeout
        self.query_timeout = query_timeout
        self._drivers: dict[str, Any] = {}

    def close(self) -> None:
        for driver in self._drivers.values():
            try:
                driver.close()
            except Exception:
                pass
        self._drivers.clear()

    def _config_for(self, target_db: str) -> dict[str, Any]:
        grouped = self.db_configs.get(self.db_group)
        if isinstance(grouped, dict) and target_db in grouped:
            return grouped[target_db]
        if target_db in self.db_configs:
            return self.db_configs[target_db]
        raise KeyError(f"Database is not configured: group={self.db_group}, db={target_db}")

    def _connection_info(self, target_db: str) -> Neo4jConnectionInfo:
        cfg = self._config_for(target_db)
        uri = cfg.get("uri")
        if not uri:
            host = "127.0.0.1" if cfg.get("host", "localhost") == "localhost" else cfg.get("host")
            port = int(cfg["port"])
            uri = f"bolt://{host}:{port}"
        return Neo4jConnectionInfo(
            name=target_db,
            uri=str(uri),
            username=str(cfg.get("username", "neo4j")),
            password=str(cfg.get("password", "")),
        )

    def _driver(self, target_db: str) -> Any:
        if target_db not in self._drivers:
            self._drivers[target_db] = make_driver(
                self._connection_info(target_db),
                timeout=self.connection_timeout,
            )
        return self._drivers[target_db]

    def verify(self, target_db: str) -> bool:
        run_cypher(self._driver(target_db), "RETURN 1 AS ok", timeout=self.query_timeout, max_records=1)
        return True

    def validate_cypher(self, target_db: str, cypher: str) -> bool:
        try:
            run_cypher(self._driver(target_db), cypher, timeout=self.query_timeout, max_records=1)
            return True
        except Exception:
            return False

    def _sample_from_structure(
        self,
        target_db: str,
        template: dict[str, Any],
        *,
        sample_pool_size: int,
    ) -> dict[int, Any] | None:
        sampling_query = template.get("sampling_query")
        if not sampling_query:
            return None
        rows = run_cypher(
            self._driver(target_db),
            sampling_query,
            timeout=self.query_timeout,
            max_records=sample_pool_size,
        )
        if not rows:
            return None
        row = random.choice(rows)
        sampled: dict[int, Any] = {}
        returns = template.get("sampling_returns") or []
        literals = template.get("literals", [])

        if returns:
            for ret in returns:
                idx = _slot_index(str(ret.get("slot", "")), len(sampled))
                alias = ret.get("alias") or f"m2_{idx}"
                if alias in row:
                    sampled[idx] = row[alias]
                elif f"literal_{idx}" in row:
                    sampled[idx] = row[f"literal_{idx}"]
        else:
            for idx, _literal in enumerate(literals):
                for alias in (f"m2_{idx}", f"literal_{idx}", "value"):
                    if alias in row:
                        sampled[idx] = row[alias]
                        break

        if len(sampled) == len(literals):
            return sampled
        return None

    def _sample_from_slots(
        self,
        target_db: str,
        literals: list[dict[str, Any]],
        *,
        sample_pool_size: int,
    ) -> dict[int, Any] | None:
        sampled: dict[int, Any] = {}
        for idx, literal in enumerate(literals):
            sample_query = literal.get("sample_query")
            if not sample_query:
                return None
            rows = run_cypher(
                self._driver(target_db),
                sample_query,
                timeout=self.query_timeout,
                max_records=sample_pool_size,
            )
            if not rows:
                return None
            sampled[_slot_index(str(literal.get("slot", "")), idx)] = row_scalar(random.choice(rows))
        return sampled

    def sample_literals(
        self,
        target_db: str,
        template: dict[str, Any],
        literals_data: dict[str, Any] | None,
        *,
        sample_pool_size: int = 100,
    ) -> tuple[dict[int, Any], str, list[str]]:
        literals = template.get("literals", [])
        if not literals:
            return {}, "none", []

        try:
            sampled = self._sample_from_structure(
                target_db,
                template,
                sample_pool_size=sample_pool_size,
            )
            if sampled is not None:
                return sampled, "structure_db", []
        except Exception:
            pass

        try:
            sampled = self._sample_from_slots(
                target_db,
                literals,
                sample_pool_size=sample_pool_size,
            )
            if sampled is not None:
                return sampled, "slot_db", []
        except Exception:
            pass

        sampled: dict[int, Any] = {}
        unresolved: list[int] = []
        for idx, literal in enumerate(literals):
            key = _literal_key(literal)
            candidates = (literals_data or {}).get(key, {}).get("existing_values", [])
            if candidates:
                sampled[_slot_index(str(literal.get("slot", "")), idx)] = random.choice(candidates)
            else:
                unresolved.append(idx)

        if unresolved:
            for idx in unresolved[:]:
                literal = literals[idx]
                example_value = literal.get("example_value", literal.get("value"))
                if example_value is not None:
                    sampled[_slot_index(str(literal.get("slot", "")), idx)] = example_value
                    unresolved.remove(idx)

        if unresolved:
            failed = [str(literals[idx].get("slot", idx)) for idx in unresolved]
            return sampled, "unresolved", failed

        method = "literal_cache" if literals_data else "example_value"
        return sampled, method, []
