"""Physics sanity: the OML forward model must behave like real Langmuir physics, and the
synthetic generator must produce invertible-looking curves."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rambhalp.config import load_config
from rambhalp import physics, synthetic
from rambhalp.preprocess import common_grid


def test_current_monotonic_increasing():
    """Electron current should rise with bias voltage (canonical convention)."""
    cfg = load_config()
    V = common_grid(cfg)
    I = physics.forward_iv_np(V, Ne_cc=500, Te_eV=0.3, Vp_V=0.0, cfg=cfg)
    assert I[-1] > I[0]
    assert np.all(np.diff(I) >= -1e-15)  # non-decreasing


def test_density_scales_amplitude():
    """Doubling Ne should roughly double the electron saturation current."""
    cfg = load_config()
    V = common_grid(cfg)
    a = physics.forward_iv_np(V, 500, 0.3, 0.0, cfg)[-1]
    b = physics.forward_iv_np(V, 1000, 0.3, 0.0, cfg)[-1]
    assert 1.8 < b / a < 2.2


def test_synthetic_shapes():
    cfg = load_config()
    X, Y, p = synthetic.make_dataset(64, cfg, seed=0)
    assert X.shape == (64, cfg["preprocess"]["grid_points"])
    assert Y.shape == (64, 3)
    assert np.isfinite(X).all()
