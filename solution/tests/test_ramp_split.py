"""Each ops segment is a triangle ramp; both halves must survive as separate sweeps.

The commanded bias runs -12 -> +12 -> -12 inside a single ops segment. Before
preprocess.split_ramps existed, bin_sweep averaged the rising and falling halves into one
curve per segment: half the measurements thrown away, and any up/down hysteresis averaged
into invisibility rather than measured.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rambhalp.config import load_config
from rambhalp import io_pds, preprocess


def _one_file(split: bool):
    cfg = load_config()
    root = cfg["data"]["_archive_abs"]
    if not (root / "data" / "raw").exists():
        pytest.skip("archive not found — set data.archive_root in config.yaml")
    cfg["preprocess"]["split_ramps"] = split
    files = io_pds.discover_files(root, level=cfg["data"]["level"], max_files=1)
    if not files:
        pytest.skip("no science files found at the configured level")
    return cfg, preprocess.segment_file(files[0], cfg)


def test_split_yields_up_and_down_ramps():
    _, sweeps = _one_file(split=True)
    assert sweeps, "no sweeps segmented"
    dirs = [s.direction for s in sweeps]
    assert dirs.count("up") > 0 and dirs.count("down") > 0
    # a triangle contributes exactly one of each
    assert abs(dirs.count("up") - dirs.count("down")) <= 1


def test_keys_stay_unique_after_splitting():
    """(timestamp, idx) is the join key for every downstream merge — it must not collide."""
    _, sweeps = _one_file(split=True)
    keys = {(s.timestamp, s.idx) for s in sweeps}
    assert len(keys) == len(sweeps), "duplicate (timestamp, idx) after ramp splitting"


def test_both_halves_cover_the_sweep_range():
    """Each half must span the commanded range on its own, or it is not a usable sweep."""
    cfg, sweeps = _one_file(split=True)
    lo, hi = cfg["preprocess"]["bias_min"], cfg["preprocess"]["bias_max"]
    for s in sweeps[:8]:
        assert s.V.min() <= lo + 2.0, f"{s.direction} ramp starts at {s.V.min():.1f} V"
        assert s.V.max() >= hi - 2.0, f"{s.direction} ramp ends at {s.V.max():.1f} V"


def test_splitting_roughly_doubles_the_sweep_count():
    _, merged = _one_file(split=False)
    _, split = _one_file(split=True)
    assert len(split) >= 1.8 * len(merged), (
        f"splitting gave {len(split)} sweeps from {len(merged)} merged ones — expected ~2x")
