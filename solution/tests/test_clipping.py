"""A clipped sweep must be flagged, and an ordinary one must not."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rambhalp.config import load_config  # noqa: E402
from rambhalp import infer, physics  # noqa: E402


def test_clipped_sweep_is_flagged_and_normal_sweep_is_not():
    cfg = load_config()
    V = np.arange(-12.0, 12.05, 0.1)                       # binned sweep, 0.1 V steps
    rng = np.random.default_rng(0)
    clean = physics.forward_iv_np(V, 450.0, 0.4, -3.7, cfg)
    normal = clean + rng.normal(0.0, 0.002 * np.ptp(clean), V.size)
    clipped = np.minimum(normal, 0.8 * clean.max())        # the front end saturating near the top
    flags = infer.clipped_sweeps([SimpleNamespace(V=V, I=normal),
                                  SimpleNamespace(V=V, I=clipped)], cfg)
    assert not flags[0]
    assert flags[1]
