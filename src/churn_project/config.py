"""Central configuration loader.

Reads ``config/project_config.yaml`` once and exposes it as a typed-ish object
with resolved absolute paths. Every other module imports ``load_config`` (or the
module-level ``CONFIG``) so there is a single source of truth for the schema,
paths, RFM settings and modelling parameters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List

import yaml

# Project root = two levels up from this file (src/churn_project/config.py).
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "project_config.yaml")


@dataclass
class Config:
    """Resolved project configuration."""

    raw: Dict[str, Any]
    project_root: str

    # convenience accessors ---------------------------------------------------
    def path(self, key: str) -> str:
        """Return an absolute path for a key under the ``paths`` section."""
        rel = self.raw["paths"][key]
        return os.path.join(self.project_root, rel)

    @property
    def paths(self) -> Dict[str, str]:
        return {k: os.path.join(self.project_root, v) for k, v in self.raw["paths"].items()}

    @property
    def schema(self) -> Dict[str, Any]:
        return self.raw["schema"]

    @property
    def id_col(self) -> str:
        return self.schema["id_col"]

    @property
    def target(self) -> str:
        return self.schema["target"]

    @property
    def date_cols(self) -> List[str]:
        return list(self.schema["date_cols"])

    @property
    def numeric_cols(self) -> List[str]:
        return list(self.schema["numeric_cols"])

    @property
    def categorical_cols(self) -> List[str]:
        return list(self.schema["categorical_cols"])

    @property
    def all_expected_cols(self) -> List[str]:
        """The 15 verified columns, in logical order."""
        return (
            [self.id_col]
            + self.date_cols
            + self.numeric_cols
            + self.categorical_cols
            + [self.target]
        )

    @property
    def rfm(self) -> Dict[str, Any]:
        return self.raw["rfm"]

    @property
    def modelling(self) -> Dict[str, Any]:
        return self.raw["modelling"]

    @property
    def leakage_cols(self) -> List[str]:
        return list(self.raw["modelling"]["leakage_cols"])

    @property
    def random_seed(self) -> int:
        return int(self.raw["project"]["random_seed"])

    def ensure_output_dirs(self) -> None:
        """Create all output/data directories if they do not exist."""
        for key in (
            "data_interim",
            "data_processed",
            "figures",
            "tables",
            "metrics",
            "models",
        ):
            os.makedirs(self.path(key), exist_ok=True)


@lru_cache(maxsize=4)
def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Config:
    """Load and cache the project configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, project_root=PROJECT_ROOT)


# Module-level singleton for convenient imports.
CONFIG = load_config()
