"""A clipped sweep must be flagged, and an ordinary one must not."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rambhalp.config import load_config  # noqa: E402
from rambhalp import infer, physics  # noqa: E402
from rambhalp.preprocess import common_grid  # noqa: E402


def test_clipped_curve_is_flagged_and_normal_curve_is_not():
    cfg = load_config()
    grid = common_grid(cfg)
    normal = physics.forward_iv_np(grid, 450.0, 0.4, -3.7, cfg)
    # the instrument saturating at 60% of the peak: a flat top from well below +8 V
    clipped = np.minimum(normal, 0.6 * normal.max())
    flags = infer.clipped_mask(np.stack([normal, clipped]), cfg)
    assert not flags[0]
    assert flags[1]
