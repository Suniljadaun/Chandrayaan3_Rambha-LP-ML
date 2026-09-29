#!/usr/bin/env python3
"""
Step 10 — Is the second electron population real, or a baseline artefact?

THE QUESTION
------------
The evidence that this archive is not a single Maxwellian is the local temperature
d(ln Ie)/dV varying five-fold across the retarding region (1.70 eV at 1.9 V below Vp down to
0.36 eV near Vp, stacked over 1403 sweeps). Phase 8 read that as a second, hotter electron
population. But three other things flatten a deep retarding slope in exactly the same direction:

  * a constant current offset left behind by imperfect ion-floor subtraction;
  * a residual LINEAR tilt in the baseline (already modelled in synthetic.py);
  * photoemission from the probe surface, which for V < Vp is bias-independent -- i.e. a
    constant, and therefore the same thing as the first item.

That last point corrects an earlier plan. Adding a photoemission term to the forward model would
NOT have settled this: in the retarding region probe photoemission is a constant, and the
pipeline's ion-floor subtraction removes constants. Probe photoemission cannot produce a graded
local-Te profile. (Photoelectrons from the regolith and lander are a different matter -- those
surround the probe and are collected like any ambient electrons, so they behave as a genuine
second Maxwellian. The I-V curve cannot distinguish locally-produced photoelectrons from
solar-wind electrons; only their dependence on illumination can. That is a question about the
population's ORIGIN, not about whether two populations exist.)

WHAT ACTUALLY DISCRIMINATES
---------------------------
The SHAPE of the local-Te profile, not its size. A constant offset diverges sharply at the
deepest point and collapses to Te almost immediately; a second Maxwellian rises gradually.
Numerically, with a true Te of 0.35 eV:

    model                                  u=-1.9   u=-1.0   ratio
    constant offset, 3% of Ie0               2.74     0.53     5.2
    constant offset, 10% of Ie0              8.30     0.96     8.6
    hot population, 5% at 1.5 eV             0.84     0.46     1.8
    REAL ARCHIVE                             1.70     1.17     1.45

The real profile is gradual. An offset tuned to match the deepest point undershoots the middle
of the range badly. That is a shape test, independent of amplitude, and it is what this script
measures properly rather than by eye.

METHOD
------
Stack real sweeps aligned on u = V - Vp, then fit three models to the retarding region by grid
search over the temperatures with linear least squares for the amplitudes (no scipy needed):

  NULL  single Maxwellian                      A1*exp(u/Tc)                      2 params
  ART   single Maxwellian + offset + tilt      A1*exp(u/Tc) + c0 + c1*u          4 params
  TWO   two Maxwellians                        A1*exp(u/Tc) + A2*exp(u/Th)       4 params

ART and TWO have the SAME parameter count, so comparing their residuals is fair with no
information criterion needed. Each model's predicted local-Te profile is also compared against
the measured one, because a model can fit the current well and still get the slope structure
wrong -- and the slope structure is the actual claim.

Outputs:
  outputs_2pop/tables/model_comparison.json
"""
import _bootstrap as B
import argparse
import json
import numpy as np
import pandas as pd
from rambhalp.config import load_config, out
from rambhalp import classical as classical_mod

UG = np.arange(-8.0, 2.0001, 0.05)
_TC_GRID = np.arange(0.15, 1.21, 0.025)
_TH_GRID = np.arange(0.6, 6.01, 0.10)


def _solve(M, y):
    amp, *_ = np.linalg.lstsq(M, y, rcond=None)
    return amp, float(np.sqrt(np.mean((y - M @ amp) ** 2)))


def fit_models(u, Ie):
    """Fit NULL, ART and TWO to one retarding-region curve. Returns a dict per model."""
    outp = {}

    best = None
    for Tc in _TC_GRID:
        M = np.exp(u / Tc)[:, None]
        amp, rms = _solve(M, Ie)
        if amp[0] > 0 and (best is None or rms < best[0]):
            best = (rms, Tc, amp)
    if best:
        rms, Tc, amp = best
        outp["NULL"] = {"rms": rms, "Tc_eV": float(Tc), "A1": float(amp[0]), "n_params": 2,
                        "pred": amp[0] * np.exp(u / Tc)}

    best = None
    for Tc in _TC_GRID:
        M = np.column_stack([np.exp(u / Tc), np.ones_like(u), u])
        amp, rms = _solve(M, Ie)
        if amp[0] > 0 and (best is None or rms < best[0]):
            best = (rms, Tc, amp)
    if best:
        rms, Tc, amp = best
        outp["ART"] = {"rms": rms, "Tc_eV": float(Tc), "A1": float(amp[0]),
                       "offset": float(amp[1]), "tilt": float(amp[2]), "n_params": 4,
                       "pred": amp[0] * np.exp(u / Tc) + amp[1] + amp[2] * u}

    best = None
    for Tc in _TC_GRID:
        for Th in _TH_GRID:
            if Th <= 1.7 * Tc:          # Yip et al. 2020 separability gate
                continue
            M = np.column_stack([np.exp(u / Tc), np.exp(u / Th)])
            amp, rms = _solve(M, Ie)
            if np.all(amp > 0) and (best is None or rms < best[0]):
                best = (rms, Tc, Th, amp)
    if best:
        rms, Tc, Th, amp = best
        outp["TWO"] = {"rms": rms, "Tc_eV": float(Tc), "Th_eV": float(Th),
                       "A1": float(amp[0]), "A2": float(amp[1]), "n_params": 4,
                       "hot_current_frac": float(amp[1] / (amp[0] + amp[1])),
                       "pred": amp[0] * np.exp(u / Tc) + amp[1] * np.exp(u / Th)}
    return outp


def local_te(u, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.gradient(np.log(np.maximum(y, 1e-300)), u)
    return 1.0 / g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u-lo", type=float, default=-2.2)
    ap.add_argument("--u-hi", type=float, default=-0.05)
    ap.add_argument("--min-stack", type=int, default=300)
    args = ap.parse_args()

    cfg = load_config()
    cfg["synthetic"]["two_population"] = True
    from pathlib import Path as _P
    bp = _P(cfg["_solution_dir"]) / "outputs_2pop"
    for k, sub in (("outputs", ""), ("figures", "figures"),
                   ("tables", "tables"), ("checkpoints", "checkpoints")):
        q = bp / sub if sub else bp
        q.mkdir(parents=True, exist_ok=True)
        cfg["paths"]["_" + k + "_abs"] = q

    print("=" * 74)
    print("STEP 10: second population, or baseline artefact?")
    print("=" * 74)

    sweeps = B.load_pickle(B.cache_dir(cfg) / "sweeps.pkl")
    cdf = pd.read_csv(out(cfg, "tables", "classical_baseline.csv"))
    import dataclasses
    fields = {f.name for f in dataclasses.fields(classical_mod.ClassicalResult)}
    usable = [c for c in cdf.columns if c in fields]
    ok = cdf[~cdf["failed"].astype(bool)]
    cmap = {(r["timestamp"], r["idx"]): classical_mod.ClassicalResult(**{k: r[k] for k in usable})
            for _, r in ok.iterrows()}
    pr = cfg["probe"]
    m_i = float(pr.get("m_i_amu", 1.0)) * 1.66053906660e-27
    K = float(np.log(np.sqrt(m_i / (2.0 * np.pi * pr["m_e"]))))
    for r in cmap.values():
        if not np.isfinite(r.V_float) and np.isfinite(r.Vp_V) and np.isfinite(r.Te_eV):
            r.V_float = float(r.Vp_V - r.Te_eV * K)

    groups = {"ALL": []}
    for sw in sweeps:
        r = cmap.get((sw.timestamp, sw.idx))
        if r is None or not np.isfinite(r.Vp_V) or not np.isfinite(r.V_float):
            continue
        V = np.asarray(sw.V, float); I = np.asarray(sw.I, float)
        o = np.argsort(V); V, I = V[o], I[o]
        if np.corrcoef(V, I)[0, 1] < 0:
            I = -I
        ion = V <= (r.V_float - cfg["classical"]["ion_floor_offset"])
        if ion.sum() < 8:
            continue
        Ie = I - float(np.median(I[ion]))
        u = V - r.Vp_V
        if u.min() > args.u_lo - 0.2 or u.max() < 0.5:
            continue
        curve = np.interp(UG, u, Ie, left=np.nan, right=np.nan)
        groups["ALL"].append(curve)
        groups.setdefault(f"ch{int(sw.adc_channel)}", []).append(curve)
        groups.setdefault(sw.date, []).append(curve)

    report = {}
    sel = (UG >= args.u_lo) & (UG <= args.u_hi)
    print(f"\nfitting window: u in [{args.u_lo}, {args.u_hi}] V, {sel.sum()} grid points")

    for name in ["ALL"] + sorted(k for k in groups if k != "ALL"):
        stack = groups[name]
        if len(stack) < args.min_stack:
            continue
        M = np.vstack(stack)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(M, axis=0)
        u_s, Ie_s = UG[sel], mean[sel]
        if not np.all(np.isfinite(Ie_s)) or np.any(Ie_s <= 0):
            m2 = np.isfinite(Ie_s) & (Ie_s > 0)
            if m2.sum() < 20:
                continue
            u_s, Ie_s = u_s[m2], Ie_s[m2]

        fits = fit_models(u_s, Ie_s)
        if "ART" not in fits or "TWO" not in fits:
            continue
        span = float(np.ptp(Ie_s))
        lt_meas = local_te(u_s, Ie_s)

        entry = {"n_stacked": len(stack)}
        for tag in ("NULL", "ART", "TWO"):
            if tag not in fits:
                continue
            f = fits[tag]
            lt_pred = local_te(u_s, f["pred"])
            good = np.isfinite(lt_meas) & np.isfinite(lt_pred)
            entry[tag] = {k: v for k, v in f.items() if k != "pred"}
            entry[tag]["rms_frac_of_span"] = f["rms"] / span
            entry[tag]["localTe_rms_eV"] = float(
                np.sqrt(np.mean((lt_pred[good] - lt_meas[good]) ** 2)))
        entry["ART_over_TWO_current_rms"] = fits["ART"]["rms"] / fits["TWO"]["rms"]
        entry["ART_over_TWO_localTe_rms"] = (entry["ART"]["localTe_rms_eV"]
                                             / max(entry["TWO"]["localTe_rms_eV"], 1e-12))
        report[name] = entry

        print(f"\n--- {name}  (n = {len(stack)}) " + "-" * (44 - len(name)))
        print(f"   {'model':>5} {'params':>6} {'current rms':>12} {'localTe rms':>12} "
              f"{'Tc (K)':>8} {'Th (K)':>8}")
        for tag in ("NULL", "ART", "TWO"):
            if tag not in entry:
                continue
            e = entry[tag]
            th = f"{e['Th_eV']*11604:>8.0f}" if "Th_eV" in e else f"{'-':>8}"
            print(f"   {tag:>5} {e['n_params']:>6} {e['rms_frac_of_span']:>11.4%} "
                  f"{e['localTe_rms_eV']:>11.3f}  {e['Tc_eV']*11604:>8.0f} {th}")
        print(f"   ART/TWO  current rms x{entry['ART_over_TWO_current_rms']:.2f}   "
              f"localTe rms x{entry['ART_over_TWO_localTe_rms']:.2f}")

    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print("ART and TWO have the same number of free parameters, so whichever has the smaller")
    print("residual wins outright -- no penalty term required.")
    print()
    print("localTe rms is the decisive column. It asks whether a model reproduces the measured")
    print("five-fold variation of the local temperature, which is the actual evidence for a")
    print("second population. A model can track the current well and still get that structure")
    print("wrong, because the current is dominated by the region near Vp where both models agree.")
    print()
    print("ART/TWO localTe rms >> 1  -> the artefact explanation cannot reproduce the slope")
    print("                            structure; the second population is real.")
    print("ART/TWO localTe rms ~ 1   -> the two explanations are indistinguishable on this data,")
    print("                            and the second population should NOT be claimed.")
    print("ART/TWO localTe rms << 1  -> a baseline offset plus tilt explains it better, and the")
    print("                            hot population was an artefact of imperfect subtraction.")
    print()
    print("Agreement across the per-channel and per-day stacks matters: the 20 Mohm channel is")
    print("~9x noisier, so if the verdict flips with channel it is instrumental, not lunar.")

    path = out(cfg, "tables", "model_comparison.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
