"""The synthetic training set must cover the real sweeps it will be applied to.

This is the check whose absence let the network mode-collapse: real curves were roughly ten
times fainter than anything in the training set, so the network extrapolated below everything
it had ever seen and returned near-identical parameters for all 14,931 sweeps (Te spread of
0.037 eV across 4,559 "recovered" sweeps). No loss curve or hold-out score shows that -- the
model was accurate on synthetic data throughout. Only comparing the two input distributions
does.

Skips cleanly when the archive or the sweep cache is absent.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rambhalp.config import load_config
from rambhalp import synthetic, infer

MAX_REAL_SWEEPS = 2000
N_SYNTHETIC = 4000


def _real_and_synthetic_peaks():
    cfg = load_config()
    cache = cfg["paths"]["_outputs_abs"] / "cache" / "sweeps.pkl"
    if not cache.exists():
        pytest.skip("outputs/cache/sweeps.pkl not built yet — run scripts/01 first")
    with open(cache, "rb") as f:
        sweeps = pickle.load(f)
    if not sweeps:
        pytest.skip("no sweeps in cache")
    step = max(1, len(sweeps) // MAX_REAL_SWEEPS)
    X_real, _ = infer.prepare_input(sweeps[::step], cfg)
    X_syn, _, _ = synthetic.make_dataset(N_SYNTHETIC, cfg, seed=4242)
    return X_real, X_syn, X_real.max(axis=1), X_syn.max(axis=1)


def test_real_amplitudes_lie_inside_the_training_distribution():
    """At least 90% of real curves must have a peak inside the synthetic 5th-95th percentile.

    Under the pre-2026-09-03 config this was 62%, with the real median sitting at the
    synthetic 10th percentile.
    """
    _, _, pk_real, pk_syn = _real_and_synthetic_peaks()
    lo, hi = np.percentile(pk_syn, 5), np.percentile(pk_syn, 95)
    inside = float(np.mean((pk_real >= lo) & (pk_real <= hi)))
    assert inside >= 0.90, (
        f"only {inside:.1%} of real sweeps fall inside the synthetic p5-p95 amplitude band "
        f"[{lo:.3g}, {hi:.3g}]; real median peak is {np.median(pk_real):.3g}. Widen "
        f"synthetic.ne_cc or check probe.sheath_exponent.")


def test_real_median_is_not_at_the_edge_of_the_training_distribution():
    """The real median must not sit in the outer 10% of the synthetic distribution.

    Sitting at the edge means the network is extrapolating for half its real inputs even if
    the ranges nominally overlap.
    """
    _, _, pk_real, pk_syn = _real_and_synthetic_peaks()
    pct = float(np.mean(pk_syn < np.median(pk_real)))
    assert 0.10 <= pct <= 0.90, (
        f"the real median curve amplitude sits at the {pct:.0%} percentile of the synthetic "
        f"training distribution — the training set is centred on the wrong density range.")


def test_curve_shapes_agree():
    """Peak-normalised median curves must agree to better than 0.15 RMSE across the sweep.

    Amplitude coverage alone is not enough: if the synthetic curve has a different SHAPE the
    network still cannot invert real data. Before the sheath_exponent fix this RMSE was 0.268;
    after it, 0.078.
    """
    X_real, X_syn, _, _ = _real_and_synthetic_peaks()
    norm = lambda A: np.median(A / (A.max(axis=1, keepdims=True) + 1e-12), axis=0)
    rmse = float(np.sqrt(np.mean((norm(X_syn) - norm(X_real)) ** 2)))
    assert rmse < 0.15, (
        f"synthetic and real median curve shapes differ by RMSE {rmse:.3f}; the forward model "
        f"in physics.py does not describe the real sweeps. Check probe.sheath_exponent and "
        f"synthetic.vp_V.")
