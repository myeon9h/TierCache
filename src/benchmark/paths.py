"""Resolve all public SQStream paths from the TierCache repository root."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src" / "benchmark" / "configs"
POOL_ROOT = REPO_ROOT / "data" / "benchmark" / "structural_pairs_pool"
WORKLOAD_ROOT = REPO_ROOT / "data" / "SQStream"


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path
