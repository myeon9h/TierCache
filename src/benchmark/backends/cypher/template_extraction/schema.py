from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CypherSchemaSummary:
    """Lightweight label/property summary collected from literal contexts."""

    node_properties: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    relationship_properties: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_literal(self, literal: dict[str, Any]) -> None:
        contexts = literal.get("contexts") or []
        if not contexts or not isinstance(contexts[0], dict):
            return
        context = contexts[0]
        prop = context.get("property")
        if not prop:
            return
        if context.get("owner_type") == "node" and context.get("label"):
            self.node_properties[str(context["label"])].add(str(prop))
        elif context.get("owner_type") == "relationship" and context.get("relation_type"):
            self.relationship_properties[str(context["relation_type"])].add(str(prop))

    def to_dict(self) -> dict[str, dict[str, list[str]]]:
        return {
            "nodes": {key: sorted(values) for key, values in sorted(self.node_properties.items())},
            "relationships": {
                key: sorted(values)
                for key, values in sorted(self.relationship_properties.items())
            },
        }
