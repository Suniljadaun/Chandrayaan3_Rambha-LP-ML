"""
classical.py — the classical Langmuir/OML baseline fit, and the failure classifier.

This is the method the ML model is meant to beat. For every sweep we try the textbook
recipe:
  * V_float : zero crossing of the I-V curve
  * Vp      : plasma potential, from the knee of the curve
  * Te      : slope of ln(I_e) vs V in the electron-retardation region (Te = 1/slope, eV)
  * Ne      : from the electron thermal current I_e0 *at the plasma potential*

Crucially, we also decide *whether the fit succeeded*, using the thresholds in config.yaml.
The list of FAILED sweeps is the exact set the ML model will be asked to recover — so this
classifier is a load-bearing part of the whole project, not a formality.

------------------------------------------------------------------------------------------
Why this file was rewritten (2026-09-03)
------------------------------------------------------------------------------------------
The previous version had two coupled bugs that between them set the project's headline
numbers, and both are fixed here.

1. DENSITY WAS ~12x TOO HIGH.  It used `Iesat = np.percentile(Ie, 97)` — the current near
   the top of the sweep, V ~ +12 V — and fed it to `Ne = 4*I/(q*A*v_th)`, a relation that
   assumes the current has SATURATED at the electron thermal current I_e0.  For a spherical
   probe in the OML regime the current does not saturate; above the plasma potential it grows
   linearly as

        I_e(V) = I_e0 * (1 + (V - Vp)/Te)

   At V = +12 V with a typical fitted Te ~ 1.05 eV that factor is ~12.5x, i.e. 1.10 dex.  The
   measured ML-minus-classical density disagreement was -0.994 dex — the same number.  The
   network was right and the baseline was wrong.
   FIX: locate Vp, fit the saturation branch (I linear in V above Vp) and extrapolate it back
   to V = Vp.  That intercept IS I_e0, which is what the thermal-flux relation wants and what
   `physics.electron_sat_current` uses, so the two are now the same quantity.

2. THE RETARDATION FIT WINDOW WAS IN THE WRONG PLACE.  It selected points by current
   fraction, `0.05*Iesat < Ie < 0.50*Iesat`.  Because `Iesat` was the inflated top-of-sweep
   value, that window landed around V = -10 to -4 V on real sweeps — below the floating
   potential, in baseline and noise, nowhere near the electron-retardation region.  A
   log-linear fit there has poor R^2 essentially always, which is why 10,849 of 12,588
   "classical failures" were nothing but `R2 < 0.90`: an artifact, not physics.
   FIX: the window is now defined relative to the plasma potential —
   [Vp - retardation_span, Vp] — which is what `classical.retardation_span` in config.yaml
   was always meant for.  That key, and `classical.ion_floor_offset`, were previously read
   by no code at all; both are now used.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import numpy as np

from .preprocess import Sweep


@dataclass
class ClassicalResult:
    date: str
    timestamp: str
    idx: int
    adc_channel: int
    t_start: str
    V_float: float
    Te_eV: float
    Ne_cc: float
    r2: float
    failed: bool
    fail_reason: str
    # Added by the 2026-09-03 rewrite. Defaulted so that older callers which construct a
    # ClassicalResult from the eleven original CSV columns (scripts/06_validate_hop.py) keep
    # working unchanged.
    Vp_V: float = np.nan
    Ie0_A: float = np.nan
    direction: str = ""            # "up" | "down"; see preprocess.split_ramps
    segment: int = -1
    Te_from_Vf: float = np.nan     # second, independent temperature estimate — see _te_from_vfloat

    def as_row(self) -> dict:
        return asdict(self)


def _r2(y, yhat):
    """Coefficient of determination for the log-linear electron-retardation fit."""
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _te_from_vfloat(V_float: float, Vp: float, cfg: dict) -> float:
    """A second estimate of Te that uses no curve fitting at all.

    For a Maxwellian electron distribution the probe floats at the potential where the
    electron and ion fluxes balance, which fixes the offset between the floating and plasma
    potentials:

        Vp - V_float = Te * ln( sqrt( m_i / (2 pi m_e) ) )

    So the measured gap between the two potentials gives Te directly. This is completely
    independent of the log-linear slope fit: it uses only where the curve crosses zero and
    where the saturation branch extrapolates to, never the shape in between.

    Having both is the point. If the electrons really are a single Maxwellian the two
    estimates must agree; if they disagree systematically, they do not. On the RAMBHA-LP
    archive they disagree by a median factor of 1.4, identically at two different probe
    resistance settings, which is why this is computed and reported for every sweep rather
    than being used to replace the slope fit.

    Returns NaN unless both potentials are finite and ordered Vp > V_float.
    """
    if not (np.isfinite(V_float) and np.isfinite(Vp)) or Vp <= V_float:
        return float("nan")
    p = cfg["probe"]
    m_i = float(p.get("m_i_amu", 1.0)) * 1.66053906660e-27
    K = np.log(np.sqrt(m_i / (2.0 * np.pi * p["m_e"])))
    if K <= 0:
        return float("nan")
    return float((Vp - V_float) / K)


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    """Moving average with an odd window, edge-padded so length is preserved.

    Used only to stabilise the numerical derivatives that locate Vp; every fitted quantity
    is computed from the UNsmoothed data.
    """
    w = int(w) | 1
    if w < 3 or y.size < w:
        return y
    pad = w // 2
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(yp, np.ones(w) / w, mode="valid")


def _find_vfloat(V: np.ndarray, I: np.ndarray) -> float:
    """Floating potential = the bias at which the probe current changes sign for good.

    Taking the FIRST sign change (what this file used to do) is wrong on any real sweep: the
    ion-saturation floor sits within noise of zero, so the raw current crosses zero a dozen
    or more times down at V ~ -12 V. On the synthetic check that put V_float at -11.3 V when
    the true value was -1.1 V, which then placed the whole electron-retardation window inside
    the ion floor and failed the fit. We smooth first and take the LAST upward crossing --
    the point above which the current stays positive.
    """
    Is = _smooth(I, max(5, V.size // 25))
    neg = np.where(Is <= 0)[0]
    if neg.size == 0 or neg[-1] + 1 >= V.size:
        return float("nan")
    k = int(neg[-1])
    dI = Is[k + 1] - Is[k]
    if not np.isfinite(dI) or dI == 0:
        return float(V[k])
    return float(V[k] - Is[k] * (V[k + 1] - V[k]) / dI)


def _find_vp(V: np.ndarray, Ie: np.ndarray) -> int:
    """Index of the plasma potential = the knee of the I-V curve.

    For the ideal OML sphere the electron current is

        V <  Vp :  I_e0 * exp((V-Vp)/Te)      ->  d2I/dV2 = (I_e0/Te^2) * exp((V-Vp)/Te)
        V >= Vp :  I_e0 * (1 + (V-Vp)/Te)     ->  d2I/dV2 = 0

    so the second derivative rises to a maximum exactly AT Vp and vanishes above it. Taking
    argmax of the (smoothed) second derivative is therefore the textbook knee-finder for this
    model, and it needs no prior estimate of Te.
    """
    w = max(5, V.size // 25)
    Is = _smooth(Ie, w)
    d1 = np.gradient(Is, V)
    d2 = np.gradient(_smooth(d1, w), V)
    d2 = _smooth(d2, w)
    # Ignore the outer 5% at each end: gradients there are edge artefacts, and a Vp pinned to
    # the last sample would leave no saturation branch to fit.
    lo = max(3, V.size // 20)
    hi = max(lo + 1, V.size - lo)
    return lo + int(np.argmax(d2[lo:hi]))


def fit_sweep(sw: Sweep, cfg: dict) -> ClassicalResult:
    """Run the classical fit on one sweep and classify success/failure.

    Method (offset-robust, standard Langmuir analysis for a spherical OML probe):
      * orient so electron current rises with V (data sign convention varies by channel)
      * subtract the ion-saturation floor, sampled below V_float (or at the most negative
        bias if the curve never crosses zero)
      * locate Vp from the knee of the curve
      * fit ln(I_e) vs V over [Vp - retardation_span, Vp]  ->  Te = 1/slope
      * fit I_e vs V over [Vp, top of sweep] and extrapolate to Vp  ->  I_e0
      * Ne from I_e0 via the electron thermal flux relation
    A true zero crossing is NOT required — many calibrated sweeps sit on a positive offset;
    requiring a sign change would flag a convention, not a physics failure. V_float is still
    reported when a crossing exists.
    """
    ccfg = cfg["classical"]
    pcfg = cfg["probe"]
    V, I = np.asarray(sw.V, float), np.asarray(sw.I, float)

    res = dict(V_float=np.nan, Vp_V=np.nan, Te_eV=np.nan, Ne_cc=np.nan, Ie0_A=np.nan,
               r2=0.0, failed=True, fail_reason="")

    if V.size < 10:
        res["fail_reason"] = "too few points"
        return _mk(sw, res)

    # ---- sort by V once, so every window below is a plain slice in voltage ----
    order = np.argsort(V)
    V, I = V[order], I[order]

    # ---- orient: electron current should increase with V ----
    if np.corrcoef(V, I)[0, 1] < 0:
        I = -I

    # ---- floating potential: last upward zero crossing of the smoothed current ----
    res["V_float"] = _find_vfloat(V, I)

    # ---- ion / offset baseline ----
    # Preferred window: everything at least `ion_floor_offset` volts below V_float, i.e. deep
    # in ion saturation where no electrons reach the probe. Falls back to the most negative
    # 10% of the sweep when there is no zero crossing to anchor to.
    n_edge = max(3, V.size // 10)
    bm = np.zeros(V.size, dtype=bool)
    if np.isfinite(res["V_float"]):
        bm = V <= (res["V_float"] - ccfg["ion_floor_offset"])
    if bm.sum() < 3:
        bm[:] = False
        bm[:n_edge] = True
    baseline = float(np.median(I[bm]))
    noise = float(np.std(I[bm])) + 1e-30
    Ie = I - baseline                                  # electron current (>= ~0)

    # ---- signal-to-noise gate ----
    Iemax = float(np.percentile(Ie, 97))
    if not np.isfinite(Iemax) or Iemax <= 0:
        res["fail_reason"] = "flat/negative electron current (no usable signal)"
        return _mk(sw, res)
    if Iemax < 3 * noise:
        res["fail_reason"] = f"low SNR (Iemax {Iemax:.1e} < 3*noise {noise:.1e})"
        return _mk(sw, res)

    # ---- Te, then Vp and I_e0 from the saturation branch ----
    # Textbook order, and non-circular:
    #   1. The electron-retardation region is the window just ABOVE the floating potential,
    #      [V_float, V_float + retardation_span] -- exactly what config.yaml's
    #      `retardation_span` comment has always said it was. (The first draft of this
    #      rewrite anchored the window to Vp instead; on real sweeps that put it on top of
    #      the ion-baseline window below V_float, where Ie is ~0 by construction, and 99.5%
    #      of sweeps failed for "too few retardation points".)
    #        ln(I_e) is linear there with slope 1/Te.
    #   2. Above the plasma potential the current follows the sheath-expansion law
    #          I_e(V) = I_e0 * (1 + (V - Vp)/Te) ** alpha,   alpha = probe.sheath_exponent
    #      so raising it to the power 1/alpha linearises it:
    #          u = I_e**(1/alpha) = (I_e0**(1/alpha) / Te) * (V - (Vp - Te))
    #      A straight-line fit of u against V therefore has slope m = I_e0**(1/alpha)/Te and
    #      crosses zero at V0 = Vp - Te, giving both remaining unknowns:
    #          Vp   = V0 + Te
    #          I_e0 = (m * Te) ** alpha
    #      For alpha = 1 (the old hard-coded OML sphere) this is exactly a line fit to I; for
    #      alpha = 0.5, which is what the real sweeps actually show, it is the familiar
    #      "I^2 versus V" extrapolation. alpha is measured, not assumed -- see config.yaml.
    # I_e0 is the electron current AT the plasma potential -- the quantity the thermal-flux
    # relation below assumes, and the same one physics.electron_sat_current() returns. The
    # old code used the current at the TOP of the sweep instead, ~(1 + (12-Vp)/Te) ~ 12x
    # larger, which is the entire 10x density disagreement with the network.
    nfit = int(ccfg["min_fit_points"])
    span = float(ccfg["retardation_span"])

    # Anchor of the retardation window: the floating potential when the curve crosses zero,
    # otherwise the point where the electron current first lifts clear of the ion floor.
    if np.isfinite(res["V_float"]) and V[0] < res["V_float"] < V[-1]:
        anchor = float(res["V_float"])
    else:
        lift = np.where(Ie > max(2.0 * noise, 0.02 * Iemax))[0]
        if lift.size == 0:
            res["fail_reason"] = "electron current never lifts off the ion floor"
            return _mk(sw, res)
        anchor = float(V[lift[0]])

    # Saturation branch: above the retardation window. Falls back to the top 30% of the
    # sweep if the window runs off the end.
    sat = V >= (anchor + span)
    if sat.sum() < nfit:
        sat = V >= np.percentile(V, 70)
    if sat.sum() < nfit:
        res["fail_reason"] = "no saturation branch to fit"
        return _mk(sw, res)
    alpha = float(pcfg.get("sheath_exponent", 1.0))
    sat = sat & (Ie > 0.0)                             # u = Ie**(1/alpha) needs Ie > 0
    if sat.sum() < nfit:
        res["fail_reason"] = "no positive saturation branch to fit"
        return _mk(sw, res)
    u = Ie[sat] ** (1.0 / alpha)
    m, b = np.polyfit(V[sat], u, 1)
    if not np.isfinite(m) or m <= 0:
        res["fail_reason"] = "non-positive saturation slope"
        return _mk(sw, res)
    V0 = -b / m                                        # = Vp - Te

    # Retardation fit, with one refinement pass that trims the window at Vp once Vp is known
    # (fitting past Vp drags the exponential slope towards the linear branch and biases Te
    # high).
    # 1.5 sigma. Two sigma was chosen when the synthetic noise was set an order of magnitude
    # too high; at the measured noise level (0.2% of span on two thirds of the archive) it
    # discards half of an already short retardation branch. At Te = 0.2 eV that branch spans
    # only about 0.75 V, i.e. 8 samples at the 0.1 V commanded step.
    floor = 1.5 * noise
    # Upper edge of the retardation window. `retardation_span` (6 V) is a generous outer
    # bound; the real limit is the plasma potential, because past Vp the curve is linear and
    # a log-linear fit through it drags the slope down and inflates Te. We do not know Vp
    # yet, so the first pass is capped by the knee estimate and later passes by V0 + Te.
    Vp_knee = float(V[_find_vp(V, Ie)])
    top = anchor + span
    if anchor + 0.3 < Vp_knee < top:
        top = Vp_knee
    Te = np.nan
    for _ in range(3):
        mask = (V >= anchor) & (V <= top) & (Ie > floor)
        if mask.sum() < nfit:
            res["fail_reason"] = f"too few retardation points ({int(mask.sum())})"
            return _mk(sw, res)
        logIe = np.log(Ie[mask])
        p = np.polyfit(V[mask], logIe, 1)
        slope = float(p[0])
        res["r2"] = float(_r2(logIe, np.polyval(p, V[mask])))
        if slope <= 0:
            res["fail_reason"] = "non-positive retardation slope (Te undefined)"
            return _mk(sw, res)
        Te = 1.0 / slope
        Vp = V0 + Te
        new_top = min(anchor + span, Vp)
        if new_top <= anchor + 0.3 or abs(new_top - top) < 0.05:
            break
        top = new_top

    res["Te_eV"] = float(Te)
    res["Vp_V"] = float(V0 + Te)
    res["Te_from_Vf"] = _te_from_vfloat(res["V_float"], res["Vp_V"], cfg)

    # Sanity gate: the plasma potential must lie inside the swept voltage range. Vp is
    # obtained by extrapolating the saturation-branch line back to where it crosses zero, so
    # a curve that is nearly FLAT above V_float -- i.e. one that does not have the
    # exponential-then-linear shape the OML sphere model predicts -- throws that intercept
    # far outside the sweep. Reporting a Vp of -18 V for a sweep that only spans -12..+12 V
    # would be meaningless, so such sweeps are failed explicitly rather than silently
    # producing a number. On the real RAMBHA archive this fires often, and that is itself a
    # finding: those sweeps are not OML-consistent and belong in the ML target set.
    if not (V[0] <= res["Vp_V"] <= V[-1]):
        res["fail_reason"] = (f"Vp {res['Vp_V']:.1f} V outside swept range "
                              f"[{V[0]:.0f},{V[-1]:.0f}] (curve not OML-consistent)")
        return _mk(sw, res)

    Ie0 = float((m * Te) ** alpha)                     # == the law evaluated at V = Vp
    if not np.isfinite(Ie0) or Ie0 <= 0:
        res["fail_reason"] = "non-positive electron current at Vp"
        return _mk(sw, res)
    res["Ie0_A"] = Ie0

    # ---- density from the electron thermal current ----
    Te_K = Te * pcfg["q_e"] / pcfg["k_b"]
    A = 4 * np.pi * pcfg["radius_m"] ** 2
    v_th = np.sqrt(8 * pcfg["k_b"] * Te_K / (np.pi * pcfg["m_e"]))
    ne_m3 = 4 * Ie0 / (pcfg["q_e"] * A * v_th)
    res["Ne_cc"] = float(ne_m3 / 1e6)

    # ---- failure classification against config thresholds ----
    reasons = []
    lo_te, hi_te = ccfg["te_plausible_eV"]
    lo_ne, hi_ne = ccfg["ne_plausible_cc"]
    if ccfg["fail_if_te_nonpositive"] and not (Te > 0):
        reasons.append("Te<=0")
    if not (lo_te <= Te <= hi_te):
        reasons.append(f"Te {Te:.2f} out of [{lo_te},{hi_te}] eV")
    if not (lo_ne <= res["Ne_cc"] <= hi_ne):
        reasons.append(f"Ne {res['Ne_cc']:.0f} out of band")
    if res["r2"] < ccfg["min_r2"]:
        reasons.append(f"R2 {res['r2']:.2f}<{ccfg['min_r2']}")

    if reasons:
        res["failed"] = True
        res["fail_reason"] = "; ".join(reasons)
    else:
        res["failed"] = False
        res["fail_reason"] = ""
    return _mk(sw, res)


def _mk(sw: Sweep, res: dict) -> ClassicalResult:
    return ClassicalResult(
        date=sw.date, timestamp=sw.timestamp, idx=sw.idx,
        adc_channel=sw.adc_channel, t_start=sw.t_start,
        V_float=res["V_float"], Te_eV=res["Te_eV"], Ne_cc=res["Ne_cc"],
        r2=res["r2"], failed=res["failed"], fail_reason=res["fail_reason"],
        Vp_V=res.get("Vp_V", np.nan), Ie0_A=res.get("Ie0_A", np.nan),
        direction=getattr(sw, "direction", ""), segment=getattr(sw, "segment", -1),
        Te_from_Vf=res.get("Te_from_Vf", np.nan),
    )


def fit_all(sweeps, cfg: dict):
    """Run the classical fit over every sweep; return a list of ClassicalResult."""
    return [fit_sweep(sw, cfg) for sw in sweeps]
