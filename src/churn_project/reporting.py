"""Small helpers for persisting tables, figures and metrics consistently."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import pandas as pd

from .config import CONFIG, Config


def save_table(df: pd.DataFrame, name: str, config: Config = CONFIG) -> str:
    """Save a dataframe as CSV under outputs/tables/ and return the path."""
    config.ensure_output_dirs()
    path = os.path.join(config.path("tables"), name)
    df.to_csv(path, index=False)
    return path


def save_json(obj: Dict[str, Any], name: str, subdir: str = "tables", config: Config = CONFIG) -> str:
    """Save a dict as pretty JSON under the given outputs subdir."""
    config.ensure_output_dirs()
    path = os.path.join(config.path(subdir), name)

    def _default(o):
        try:
            import numpy as np

            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
        except Exception:
            pass
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)
    return path


def save_figure(fig, name: str, config: Config = CONFIG, dpi: int = 120) -> str:
    """Save a matplotlib figure under outputs/figures/ and return the path."""
    config.ensure_output_dirs()
    path = os.path.join(config.path("figures"), name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
