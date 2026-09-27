"""
config.py — load config.yaml into a plain dict and resolve paths.

Why a module for this: every script needs the same paths/hyperparameters, and we want
ONE place that turns the (possibly relative) archive path into an absolute one and creates
the output folders. Import `load_config()` everywhere; never hard-code a path in a script.
"""
from __future__ import annotations
import os
from pathlib import Path
import yaml

# The solution/ directory = two levels up from this file (src/rambhalp/config.py).
SOLUTION_DIR = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Read config.yaml, resolve paths relative to solution/, make output dirs.

    Returns the config dict with two extra convenience keys injected:
        cfg["_solution_dir"]  absolute Path to solution/
        cfg["data"]["_archive_abs"]  absolute Path to the RAMBHA archive root
    """
    cfg_path = Path(path) if path else (SOLUTION_DIR / "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["_solution_dir"] = SOLUTION_DIR

    # Resolve the archive root. Absolute stays absolute; relative is joined to solution/.
    root = Path(cfg["data"]["archive_root"])
    if not root.is_absolute():
        root = (SOLUTION_DIR / root).resolve()
    cfg["data"]["_archive_abs"] = root

    # Resolve + create output directories so no script has to worry about it.
    for key in ("outputs", "figures", "tables", "checkpoints"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = (SOLUTION_DIR / p).resolve()
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"]["_" + key + "_abs"] = p

    return cfg


def out(cfg: dict, kind: str, name: str) -> Path:
    """Return an absolute path inside an output subfolder.

    kind in {"figures","tables","checkpoints","outputs"}; name is a filename.
    Example: out(cfg, "tables", "classical_baseline.csv")
    """
    return cfg["paths"]["_" + kind + "_abs"] / name
