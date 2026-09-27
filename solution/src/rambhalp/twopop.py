"""
twopop.py — resolve the electron population into two Maxwellians instead of one.

WHY THIS EXISTS
---------------
Two independent estimates of electron temperature from the same RAMBHA-LP curves disagree by
about 44%: the log-linear slope of the retarding region gives ~3950 K, while the offset between
the floating and plasma potentials gives ~2860 K. Each estimator reproduces itself to within 8%
on a repeat measurement of the same plasma (the two halves of a triangular ramp), so the
disagreement is six to nine times either one's repeatability. For a single Maxwellian population
the two must agree exactly — the potential offset IS the temperature times a constant. They do
not.

The direct evidence is the local slope. Stacking 1,403 sweeps and measuring the LOCAL
temperature d(ln Ie)/dV as a function of depth below the plasma potential gives:

    V - Vp (V)    -1.94  -1.44  -1.06  -0.81  -0.56  -0.31  -0.06
    local Te (eV)  1.70   1.46   0.98   0.55   0.36   0.35   0.49

A single Maxwellian requires that row to be flat. It varies by a factor of five, monotonically,
in exactly the manner a two-population plasma produces: close to Vp the cold majority dominates
the current, far below it only the hot minority's tail survives.

This is a known situation rather than a novel one. Chatain et al. (2021, JGR 126, e2020JA028412)
re-analysed Cassini RPWS/LP sweeps from Titan's ionosphere the same way and resolved two to four
populations where the published analysis had fitted one, concluding that the single-population
treatment had masked distinct formation mechanisms. One of their populations was attributed to
photoelectrons emitted from the probe's own boom.

For this dataset the hot component is consistent with lunar photoelectrons: the probe sits about
2 m above a sunlit surface, inside the photoelectron sheath, and the archive's own level-0B
processing removes 100-170 nA of photoemission current. Lunar photoelectrons are reported near
2 eV (Feuerbacher et al. 1972).

METHOD
------
Three stages, all linear least squares — no optimiser and no scipy dependency.

  1. Peel the hot tail. Far below Vp only the hot population contributes, so a straight line
     through ln(Ie) there gives Th directly.
  2. Subtract the hot population's extrapolation from the whole curve and fit the remainder
     close to Vp, which gives Tc.
  3. Hold Vp and the two temperatures fixed and solve BOTH amplitudes at once by least squares
     over the whole usable curve, refining the temperatures over a small multiplicative grid.
     Stage 2 alone constrains only the retarding branch and leaves the saturation branch
     unheld, which makes the saturation residual worse; this stage fixes that.

WHAT IT DOES AND DOES NOT EXPLAIN
---------------------------------
Measured against the single-Maxwellian fit on the same 408 sweeps:

    retarding region  (V <  Vp)   1.67%  ->  0.58%   of curve span   (2.9x better)
    saturation region (V >= Vp)   1.93%  ->  3.24%   when amplitudes are unconstrained there
    whole curve, joint solve      1.93%  ->  1.81%

So the second population accounts for the structure in the retarding region, which is what it
was proposed to explain, and very little of the structure in the saturation region. The latter
is a separate matter, governed by the sheath-expansion exponent (probe.sheath_exponent), and
should not be attributed to electron populations.
"""
from __future__ import annotations

import numpy as np

from . import classical as _classical

# Multiplicative refinement grids around the staged temperature estimates.
_TC_GRID = (0.8, 0.9, 1.0, 1.1, 1.25)
_TH_GRID = (0.7, 0.85, 1.0, 1.2, 1.5)

# Windows, in volts relative to the plasma potential.
_HOT_BAND = (-2.0, -1.0)     # deep retarding region: hot population only
_COLD_BAND = (-0.7, 0.0)     # just below Vp: cold population dominates
_FIT_FLOOR = -2.0            # deepest point included in the joint solve


def _shape(V, Vp, Te, alpha):
    """Unit-amplitude electron current for one Maxwellian population."""
    x = (V - Vp) / Te
    return np.where(V < Vp,
                    np.exp(np.clip(x, -60.0, 0.0)),
                    (1.0 + np.clip(x, 0.0, 60.0)) ** alpha)


def _density(Ie0, Te_eV, cfg):
    """Electron density (cm^-3) from a population's current at the plasma potential."""
    p = cfg["probe"]
    A = 4.0 * np.pi * p["radius_m"] ** 2
    v_th = np.sqrt(8.0 * p["q_e"] * Te_eV / (np.pi * p["m_e"]))
    return float(4.0 * Ie0 / (p["q_e"] * A * v_th) / 1e6)


def fit_two_populations(sw, cfg: dict, base=None) -> dict | None:
    """Resolve one sweep into a cold and a hot Maxwellian electron population.

    `base` may be a ClassicalResult already computed for this sweep, to avoid refitting; it
    supplies the plasma potential, floating potential and the single-population comparison.

    Returns None when the sweep cannot support the decomposition, and otherwise a dict with
    Tc_eV, Th_eV, Nc_cc, Nh_cc, hot_fraction, Vp_V, plus the residual of both models in the
    retarding region so the caller can judge whether the second population earned its place.
    """
    ccfg = cfg["classical"]
    alpha = float(cfg["probe"].get("sheath_exponent", 1.0))
    nfit = int(ccfg["min_fit_points"])

    r = base if base is not None else _classical.fit_sweep(sw, cfg)
    if r.failed or not np.isfinite(r.Vp_V) or not np.isfinite(r.V_float):
        return None

    V = np.asarray(sw.V, float)
    I = np.asarray(sw.I, float)
    order = np.argsort(V)
    V, I = V[order], I[order]
    if np.corrcoef(V, I)[0, 1] < 0:
        I = -I

    ion = V <= (r.V_float - ccfg["ion_floor_offset"])
    if ion.sum() < 8:
        return None
    noise = float(np.std(I[ion]))
    Ie = I - float(np.median(I[ion]))
    u = V - r.Vp_V
    span = float(np.ptp(Ie)) + 1e-30

    # Stage 1 — the hot tail, alone in the deep retarding region.
    hot = (u >= _HOT_BAND[0]) & (u <= _HOT_BAND[1]) & (Ie > 3.0 * noise)
    if hot.sum() < max(nfit, 6):
        return None
    ah, bh = np.polyfit(V[hot], np.log(Ie[hot]), 1)
    if ah <= 0:
        return None
    Th0 = 1.0 / ah

    # Stage 2 — remove it everywhere, then fit what is left close to Vp.
    rest = Ie - np.exp(np.polyval([ah, bh], V))
    cold = (u >= _COLD_BAND[0]) & (u < _COLD_BAND[1]) & (rest > 3.0 * noise)
    if cold.sum() < nfit:
        return None
    ac, _bc = np.polyfit(V[cold], np.log(rest[cold]), 1)
    if ac <= 0:
        return None
    Tc0 = 1.0 / ac

    # Stage 3 — both amplitudes at once, over the whole usable curve.
    fit = u >= _FIT_FLOOR
    if fit.sum() < 12:
        return None
    best = None
    for fc in _TC_GRID:
        for fh in _TH_GRID:
            Tc, Th = Tc0 * fc, Th0 * fh
            if Th <= 1.5 * Tc:           # not separable; reject rather than invent structure
                continue
            M = np.column_stack([_shape(V[fit], r.Vp_V, Tc, alpha),
                                 _shape(V[fit], r.Vp_V, Th, alpha)])
            amp, *_ = np.linalg.lstsq(M, Ie[fit], rcond=None)
            if np.any(amp <= 0) or not np.all(np.isfinite(amp)):
                continue
            rms = float(np.std(Ie[fit] - M @ amp)) / span
            if best is None or rms < best[0]:
                best = (rms, Tc, Th, amp)
    if best is None:
        return None
    rms, Tc, Th, amp = best

    # How much of the retarding region each model explains — the fair comparison, since the
    # saturation branch is governed by the sheath exponent rather than by populations.
    ret = (u >= _FIT_FLOOR) & (u < 0.0)
    one = _classical_model(V, r, alpha)
    two = amp[0] * _shape(V, r.Vp_V, Tc, alpha) + amp[1] * _shape(V, r.Vp_V, Th, alpha)
    res_one = float(np.std((Ie - one)[ret])) / span if ret.sum() > 4 else float("nan")
    res_two = float(np.std((Ie - two)[ret])) / span if ret.sum() > 4 else float("nan")

    Nc, Nh = _density(amp[0], Tc, cfg), _density(amp[1], Th, cfg)
    return {
        "Tc_eV": float(Tc), "Th_eV": float(Th),
        "Nc_cc": Nc, "Nh_cc": Nh,
        "hot_fraction": float(Nh / (Nc + Nh)) if (Nc + Nh) > 0 else float("nan"),
        "Vp_V": float(r.Vp_V),
        "resid_one_pop": res_one, "resid_two_pop": res_two,
        "resid_whole_curve": rms,
        "Te_single_eV": float(r.Te_eV),
    }


def _classical_model(V, r, alpha):
    """The single-population model this sweep's classical fit implies."""
    return r.Ie0_A * _shape(V, r.Vp_V, r.Te_eV, alpha)


def fit_all(sweeps, cfg: dict, classical_rows=None):
    """Run the decomposition over many sweeps; returns a list the same length, with None
    wherever the sweep could not support it."""
    if classical_rows is None:
        return [fit_two_populations(s, cfg) for s in sweeps]
    return [fit_two_populations(s, cfg, base=b) for s, b in zip(sweeps, classical_rows)]
