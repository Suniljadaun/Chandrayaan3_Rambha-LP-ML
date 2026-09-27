"""Controls for the two-population decomposition.

The decomposition is only evidence about the plasma if it does two things: recover a second
population that really is there, and decline to invent one that is not. Both are tested here on
synthetic curves where the answer is known by construction.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rambhalp.config import load_config
from rambhalp import physics, twopop
from rambhalp.preprocess import Sweep, common_grid

NOISE_FRAC = 0.002          # the measured level on two thirds of the archive
ION_FRAC = 0.023            # physical ion floor, ~sqrt(m_e/m_H) of I_e0


def _base(cfg, Vp, Vf, Te, Ie0):
    """A ClassicalResult carrying the TRUE potentials, so the decomposition can be tested in
    isolation from the classical stage's acceptance gate.

    That gate is not incidental here: a strongly two-population curve fails the classical
    log-linear R^2 test outright (measured: R^2 = 0.04, Te = 6.6 eV on a synthetic cold+hot
    plasma). Testing the decomposition through the gate would only re-test the gate.
    """
    from rambhalp.classical import ClassicalResult
    return ClassicalResult(date="s", timestamp="t", idx=0, adc_channel=-1, t_start="",
                           V_float=Vf, Te_eV=Te, Ne_cc=1.0, r2=1.0, failed=False,
                           fail_reason="", Vp_V=Vp, Ie0_A=Ie0)


def _sweep(pops, cfg, rng, noise_frac=NOISE_FRAC):
    """Build a curve from one or more (Ne_cc, Te_eV) populations sharing a plasma potential."""
    g = common_grid(cfg)
    Vp = -4.0
    I = np.zeros_like(g)
    for Ne, Te in pops:
        I = I + physics.forward_iv_np(g, Ne, Te, Vp, cfg)
    Ie0 = physics.electron_sat_current(pops[0][0] * 1e6, pops[0][1], cfg)
    I = I + rng.normal(0.0, noise_frac * np.ptp(I), g.size) - ION_FRAC * Ie0
    return Sweep(date="synthetic", level="rawB", timestamp="t", idx=0, V=g.copy(), I=I)


def test_recovers_a_known_two_population_plasma():
    """A cold majority plus a hot minority, at the proportions the archive suggests."""
    cfg = load_config()
    rng = np.random.default_rng(3)
    Nc, Tc, Nh, Th = 420.0, 0.25, 25.0, 1.40
    sw = _sweep([(Nc, Tc), (Nh, Th)], cfg, rng)
    Ie0 = physics.electron_sat_current(Nc * 1e6, Tc, cfg)
    out = twopop.fit_two_populations(sw, cfg, base=_base(cfg, -4.0, -5.2, Tc, Ie0))
    assert out is not None, "the decomposition declined a curve that genuinely has two populations"
    assert 0.6 < out["Tc_eV"] / Tc < 1.6, f"cold Te {out['Tc_eV']:.3f} vs true {Tc}"
    assert 0.5 < out["Th_eV"] / Th < 2.0, f"hot Te {out['Th_eV']:.3f} vs true {Th}"
    assert out["Th_eV"] > 2.0 * out["Tc_eV"], "the two populations were not separated"


def test_does_not_invent_a_second_population():
    """On a single Maxwellian the decomposition must not report a substantial hot component.

    This is the control that makes the real-archive result meaningful. A method that splits
    every curve into two populations tells you nothing about whether two are present.
    """
    cfg = load_config()
    rng = np.random.default_rng(11)
    Tc, Nc = 0.28, 450.0
    sw = _sweep([(Nc, Tc)], cfg, rng)
    Ie0 = physics.electron_sat_current(Nc * 1e6, Tc, cfg)
    out = twopop.fit_two_populations(sw, cfg, base=_base(cfg, -4.0, -5.2, Tc, Ie0))
    if out is None:
        return                      # declining a single-population curve is a correct outcome
    assert out["hot_fraction"] < 0.25, (
        f"invented a hot population carrying {out['hot_fraction']:.1%} of the density on a "
        f"curve that has only one")


def test_two_populations_beat_one_where_two_exist():
    """On a genuine two-population curve the second component must reduce the retarding-region
    residual — otherwise it is decoration, not an improvement."""
    cfg = load_config()
    rng = np.random.default_rng(5)
    Tc, Nc = 0.25, 420.0
    sw = _sweep([(Nc, Tc), (25.0, 1.40)], cfg, rng)
    Ie0 = physics.electron_sat_current(Nc * 1e6, Tc, cfg)
    out = twopop.fit_two_populations(sw, cfg, base=_base(cfg, -4.0, -5.2, Tc, Ie0))
    if out is None:
        pytest.skip("decomposition declined")
    if not np.isfinite(out["resid_one_pop"]) or not np.isfinite(out["resid_two_pop"]):
        pytest.skip("residuals unavailable")
    assert out["resid_two_pop"] < out["resid_one_pop"], (
        f"two-population residual {out['resid_two_pop']:.4f} did not improve on the "
        f"single-population {out['resid_one_pop']:.4f}")
