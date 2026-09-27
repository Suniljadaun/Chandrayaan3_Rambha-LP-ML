"""The five-parameter network must actually resolve the second population.

Predicting five numbers is easy; predicting them *correctly* is the claim. These tests train a
small network on a small synthetic set and check that it recovers the hot component it was
shown, and — the control that matters — that it reports little or no hot component when there
is none.

Skipped without torch.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rambhalp.config import load_config
from rambhalp import synthetic, train
from rambhalp.model import n_targets, predict

torch = pytest.importorskip("torch")


def _cfg(two=True, epochs=60):
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["synthetic"]["two_population"] = two
    cfg["train"]["epochs"] = epochs
    cfg["train"]["batch_size"] = 256
    return cfg


def test_output_width_follows_the_config():
    assert n_targets(_cfg(two=False)) == 3
    assert n_targets(_cfg(two=True)) == 5


def test_network_recovers_the_hot_population():
    cfg = _cfg()
    X, Y, _ = synthetic.make_dataset(10000, cfg, seed=21)
    Xv, Yv, pv = synthetic.make_dataset(1500, cfg, seed=22)
    model, _ = train.train_model(X, Y, cfg, verbose=False)
    p = predict(model, Xv, cfg, mc_samples=1)

    ne_err = float(np.median(np.abs(np.log10(p["Ne_cc"]) - Yv[:, 0])))
    tc_err = float(np.median(np.abs(p["Te_eV"] - Yv[:, 1])))
    assert ne_err < 0.25, f"cold density error {ne_err:.3f} dex"

    # Judge the cold temperature against the do-nothing baseline rather than a fixed number
    # of eV. A network trained on a small set for a few dozen epochs will not reach the
    # accuracy of the full 40k-sample, 250-epoch run, so an absolute threshold here tests the
    # training budget and not the method. What must hold at ANY budget is that the network
    # extracts real information: predicting the training median for every sweep is the
    # baseline to beat, and beating it by half is a claim about the model, not the schedule.
    baseline = float(np.median(np.abs(Yv[:, 1] - np.median(Y[:, 1]))))
    assert tc_err < 0.5 * baseline, (
        f"cold temperature error {tc_err:.3f} eV vs {baseline:.3f} eV for predicting the "
        f"median -- the network is barely beating a constant")

    # The hot component must track the truth, not sit at a constant. Correlation over the
    # validation set is the test that distinguishes a real prediction from a learned average.
    true_frac = pv["Ne_h"] / (pv["Ne"] + pv["Ne_h"])
    pred_frac = p["Nh_cc"] / np.maximum(p["Ne_cc"] + p["Nh_cc"], 1e-9)
    m = np.isfinite(true_frac) & np.isfinite(pred_frac)
    r = float(np.corrcoef(true_frac[m], pred_frac[m])[0, 1])
    assert r > 0.5, f"predicted hot fraction correlates with truth at only {r:.2f}"


def test_network_does_not_invent_a_hot_population():
    """Train with two populations available, then evaluate on curves that have none.

    A network that reports a hot component on single-population curves would make the
    archive result meaningless, since every sweep would appear to have two populations.
    """
    cfg = _cfg()
    X, Y, _ = synthetic.make_dataset(10000, cfg, seed=31)
    model, _ = train.train_model(X, Y, cfg, verbose=False)

    clean = copy.deepcopy(cfg)
    clean["synthetic"]["hot_fraction"] = [0.0, 0.0]     # genuinely single-population curves
    clean["synthetic"]["hot_absent_frac"] = 1.0
    Xc, _Yc, _ = synthetic.make_dataset(1000, clean, seed=32)
    p = predict(model, Xc, cfg, mc_samples=1)
    frac = p["Nh_cc"] / np.maximum(p["Ne_cc"] + p["Nh_cc"], 1e-9)
    assert float(np.median(frac)) < 0.06, (
        f"reported a hot population at {np.median(frac):.1%} of the density on curves that "
        f"have only one")
