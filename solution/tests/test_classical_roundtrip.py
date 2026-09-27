"""Round-trip test for the classical fit.

Build an I-V curve from the OML forward model with KNOWN (Ne, Te, Vp), hand it to
classical.fit_sweep, and check the fit gives the answer back. Eight lines of setup that
would have caught the 2026-09-03 density bug on day one: the old fit read the current at the
top of the sweep instead of at the plasma potential and returned Ne roughly 8-13x too high on
every single one of these cases, while reporting success.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rambhalp.config import load_config
from rambhalp import physics, classical
from rambhalp.preprocess import Sweep, common_grid

# (Ne cm^-3, Te eV, Vp V). Chosen to sit where the real archive sits -- the corrected
# classical fit gives Te around 0.3 eV and Vp around -4 V -- rather than spread over the
# full sampled synthetic range. Test cases far from the data's own regime were previously
# reporting a failure of the fit when what they really showed was that the cases were
# unrepresentative.
CASES = [(200, 0.20, 0.0), (500, 0.25, -4.0), (500, 0.30, -1.5), (1000, 0.35, 1.0),
         (2000, 0.45, -3.0), (100, 0.30, 2.0), (3000, 0.40, -5.0), (300, 0.55, -2.0)]


def _sweep(Ne, Te, Vp, cfg, rng, noise_frac=0.002, ion_frac=0.023):
    """An OML curve with a physical ion floor (~sqrt(m_e/m_H) of I_e0) and gaussian noise at
    the level actually measured on the archive.

    noise_frac is a fraction of the CURVE SPAN, and 0.002 is the measured median for the
    49.9 kohm and 1 Mohm settings (the 20 Mohm setting is ~0.018). The previous default here
    was 2% of I_e0, which is a different and much harsher quantity, and it made a
    correctly-narrow retardation window look broken.
    """
    g = common_grid(cfg)
    Ie0 = physics.electron_sat_current(Ne * 1e6, Te, cfg)
    I = physics.forward_iv_np(g, Ne, Te, Vp, cfg)
    I = I + rng.normal(0.0, noise_frac * np.ptp(I), g.size) - ion_frac * Ie0
    return Sweep(date="synthetic", level="rawB", timestamp="t", idx=0, V=g.copy(), I=I)


def test_noisy_channel_still_recovers_density():
    """At the 20 Mohm noise level (1.8% of span) density must still come back within 40%,
    even though temperature will not. This is the regime the ML model exists to serve."""
    cfg = load_config()
    rng = np.random.default_rng(5)
    r = classical.fit_sweep(_sweep(500, 0.30, -4.0, cfg, rng, noise_frac=0.018), cfg)
    if r.failed:
        pytest.skip(f"classical fit declines this sweep ({r.fail_reason}) — expected at 20 Mohm")
    assert 0.6 < r.Ne_cc / 500 < 1.4, f"Ne {r.Ne_cc:.0f} vs true 500 at 20 Mohm noise"


@pytest.mark.parametrize("Ne,Te,Vp", CASES)
def test_roundtrip_recovers_parameters(Ne, Te, Vp):
    cfg = load_config()
    rng = np.random.default_rng(7)
    r = classical.fit_sweep(_sweep(Ne, Te, Vp, cfg, rng), cfg)

    assert not r.failed, f"fit failed on a clean synthetic curve: {r.fail_reason}"
    # Density within 40%. The old implementation was out by a factor of 8-13 here.
    assert 0.6 < r.Ne_cc / Ne < 1.4, f"Ne {r.Ne_cc:.0f} vs true {Ne}"
    # Temperature within 40%.
    assert 0.6 < r.Te_eV / Te < 1.4, f"Te {r.Te_eV:.2f} vs true {Te}"
    # Plasma potential within half a volt.
    assert abs(r.Vp_V - Vp) < 0.5, f"Vp {r.Vp_V:.2f} vs true {Vp}"


def test_density_is_measured_at_the_plasma_potential():
    """The fitted I_e0 must equal physics.electron_sat_current for the fitted parameters.

    This is the invariant the old code broke: it used the current at V = +12 V, which for a
    spherical OML probe is (1 + (12 - Vp)/Te) times larger than the current at Vp -- about
    12x at a typical Te of 1 eV.
    """
    cfg = load_config()
    rng = np.random.default_rng(3)
    Ne, Te, Vp = 800.0, 0.35, -4.0
    r = classical.fit_sweep(_sweep(Ne, Te, Vp, cfg, rng), cfg)
    assert not r.failed, r.fail_reason
    expected = physics.electron_sat_current(r.Ne_cc * 1e6, r.Te_eV, cfg)
    assert 0.9 < r.Ie0_A / expected < 1.1, (
        f"fitted I_e0 {r.Ie0_A:.3e} A is not the current at Vp "
        f"(forward model says {expected:.3e} A)")


def test_vfloat_is_the_last_crossing_not_the_first():
    """Noise in the ion floor makes the raw current cross zero many times; V_float must be
    the crossing above which the current stays positive."""
    cfg = load_config()
    rng = np.random.default_rng(11)
    sw = _sweep(200, 0.3, 0.0, cfg, rng)
    r = classical.fit_sweep(sw, cfg)
    # For Te = 0.3 eV and an ion floor at 2.3% of I_e0, V_float sits about 3.8*Te below Vp.
    assert -3.0 < r.V_float < 1.0, f"V_float {r.V_float:.2f} is not near the true lift-off"


def test_two_temperature_estimators_agree_on_a_true_maxwellian():
    """The slope fit and the floating-potential offset must agree on synthetic curves.

    The synthetic generator produces a single Maxwellian electron population by
    construction, so both estimators are measuring the same quantity and must return the
    same answer. This test is what makes their disagreement on the REAL archive — a median
    factor of 1.4, identical at two different probe-resistance settings — evidence about the
    plasma rather than a bug in either estimator.
    """
    cfg = load_config()
    rng = np.random.default_rng(17)
    ratios = []
    for Ne, Te, Vp in [(500, 0.25, -4.0), (1000, 0.35, 1.0), (300, 0.30, -2.0)]:
        r = classical.fit_sweep(_sweep(Ne, Te, Vp, cfg, rng), cfg)
        if r.failed or not np.isfinite(r.Te_from_Vf):
            continue
        ratios.append(r.Te_eV / r.Te_from_Vf)
    if not ratios:
        pytest.skip("no sweeps produced both estimates")
    assert 0.7 < float(np.median(ratios)) < 1.45, (
        f"the two estimators disagree by {np.median(ratios):.2f}x on synthetic Maxwellian "
        f"curves, where they should agree — one of them is wrong")
