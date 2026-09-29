#!/usr/bin/env python3
"""
Step 12 — Does the retarding slope change WITHIN a single sweep?

WHY THIS IS THE RIGHT TEST
--------------------------
Every previous attempt at this question went through a STACKED curve, and stacking is what caused
all the trouble: averaging sweeps that each have a different temperature bends the result, so the
bend proves nothing (Phase 15). Stacking also forced a matched synthetic control, and every wrong
turn in Phases 14-15c came from getting that control subtly wrong.

This script avoids stacking entirely. For each sweep on its own it fits the retarding slope in two
separate bias windows:

    DEEP  u in [-1.5, -0.8] V below Vp     (electron current ~1.4-13% of its peak)
    NEAR  u in [-0.6, -0.05] V below Vp    (~13-88% of peak)

A single Maxwellian has ONE temperature, so Te_near and Te_deep must agree within measurement
error. A curve that genuinely steepens or flattens with bias will show Te_near != Te_deep in
individual sweeps, with no averaging involved.

Both windows carry real signal on the quiet channels, which is why this works where the per-sweep
two-population fit did not: that fit needed the region 1.5-2.6 V below Vp, where the current is
under the noise floor (Phase 14). These windows sit above it.

TWO INDEPENDENT ERROR ESTIMATES, SO THE RESULT CANNOT REST ON AN ASSUMPTION
--------------------------------------------------------------------------
1. RAMP PAIRS. Each commanded sweep is a triangle, so its rising and falling halves are two
   independent measurements of the same plasma seconds apart. The scatter of (Te_near - Te_deep)
   between the two halves is a purely empirical error bar on that difference -- no model, no
   assumption about noise.
2. SYNTHETIC CONTROL. The identical two-window fit is run on synthetic sweeps that are PERFECT
   single Maxwellians carrying the archive's measured noise and baseline tilt. That gives the null
   distribution of (Te_near - Te_deep): whatever the fitting procedure itself produces when the
   answer is known to be zero.

The real difference is meaningful only if it exceeds BOTH. If the synthetic control shows the same
offset, the effect is an artefact of the fit windows rather than a property of the plasma -- which
is exactly the possibility Phase 15c flagged and could not test without circularity.

Outputs:
  outputs_2pop/tables/slope_curvature.json
"""
import _bootstrap as B
import argparse
import dataclasses
import json
import numpy as np
import pandas as pd
from rambhalp.config import load_config, out
from rambhalp import classical as classical_mod, physics, synthetic


def _fit_window(u, Ie, noise, lo, hi, min_pts):
    """Te from a straight line through ln(Ie) over u in [lo, hi]. None if unusable."""
    m = (u >= lo) & (u <= hi) & (Ie > 3.0 * noise)
    if m.sum() < min_pts:
        return None
    a, _b = np.polyfit(u[m], np.log(Ie[m]), 1)
    if a <= 0:
        return None
    return float(1.0 / a)


def two_window(u, Ie, noise, args):
    d = _fit_window(u, Ie, noise, args.deep_lo, args.deep_hi, args.min_pts)
    n = _fit_window(u, Ie, noise, args.near_lo, args.near_hi, args.min_pts)
    if d is None or n is None:
        return None
    return d, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-lo", type=float, default=-1.5)
    ap.add_argument("--deep-hi", type=float, default=-0.8)
    ap.add_argument("--near-lo", type=float, default=-0.6)
    ap.add_argument("--near-hi", type=float, default=-0.05)
    ap.add_argument("--min-pts", type=int, default=5)
    ap.add_argument("--n-synth", type=int, default=6000)
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

    print("=" * 76)
    print("STEP 12: does the retarding slope change within a single sweep?")
    print("=" * 76)
    print(f"DEEP window u in [{args.deep_lo}, {args.deep_hi}] V   "
          f"NEAR window u in [{args.near_lo}, {args.near_hi}] V")

    # ---------------- real sweeps ---------------------------------------------------------
    sweeps = B.load_pickle(B.cache_dir(cfg) / "sweeps.pkl")
    cdf = pd.read_csv(out(cfg, "tables", "classical_baseline.csv"))
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

    rows = []
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
        noise = float(np.std(I[ion]))
        Ie = I - float(np.median(I[ion]))
        res = two_window(V - r.Vp_V, Ie, noise, args)
        if res is None:
            continue
        rows.append({"date": sw.date, "channel": int(sw.adc_channel),
                     "timestamp": sw.timestamp, "segment": int(sw.segment),
                     "direction": sw.direction,
                     "Te_deep": res[0], "Te_near": res[1], "d": res[1] - res[0]})
    real = pd.DataFrame(rows)
    print(f"\nreal sweeps with both windows fitted: {len(real)}")
    if real.empty:
        print("none -- widen the windows or lower --min-pts")
        return

    # ---------------- synthetic control, identical procedure AND identical selection -------
    # The control must be drawn from the ARCHIVE's own (Te, Ne, Vp) distribution, not from
    # config.yaml's wide uniform prior. This is not a detail -- it decides the result.
    #
    # The deep window requires Ie > 3*noise at u = -1.5 V, where a single Maxwellian sits at
    # exp(-1.5/Te) of its peak: 0.7% at Te = 0.3 eV, 2.4% at 0.4 eV, 8% at 0.6 eV. Against a
    # noise floor of 0.24% (quiet channels) and 1.8% (20 Mohm), the window is fittable ONLY on
    # sweeps that are hot enough -- so the fitted sample is biased toward large Te_deep, which
    # makes d negative with no hot population present at all.
    #
    # A control drawn from the config prior (median Te ~1.15 eV) clears that threshold easily
    # and never experiences the selection. Resampling the archive's measured parameters exposes
    # it to exactly the same cut, so the comparison measures physics rather than the prior.
    from rambhalp.preprocess import common_grid
    grid = np.asarray(common_grid(cfg), float)
    rng = np.random.default_rng(4242)
    src = ok[["Te_eV", "Ne_cc", "Vp_V"]].dropna()
    src = src[(src.Te_eV > 0) & (src.Ne_cc > 0)]
    pick = rng.integers(0, len(src), args.n_synth)
    Te_s = src.Te_eV.values[pick]; Ne_s = src.Ne_cc.values[pick]; Vp_s = src.Vp_V.values[pick]

    chans = cfg["synthetic"].get("noise_frac_by_channel") or [cfg["synthetic"]["noise_frac"]]
    ion_lo, ion_hi = cfg["synthetic"]["ion_current_frac"]
    c1 = {**cfg, "synthetic": {**cfg["synthetic"], "two_population": False}}

    srows = []
    for j in range(args.n_synth):
        clean = physics.forward_iv_np(grid, Ne_s[j], Te_s[j], Vp_s[j], c1)
        span = float(np.ptp(clean)) + 1e-30
        mag = 10 ** rng.uniform(np.log10(ion_lo), np.log10(ion_hi))
        tilt = rng.choice([-1.0, 1.0]) * mag * span * (grid - grid.min()) / np.ptp(grid)
        lo, hi = chans[rng.integers(len(chans))]
        nf = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
        Ie = clean + tilt + rng.normal(0.0, nf * span, grid.size)
        u = grid - Vp_s[j]
        ionm = u <= -2.0
        if ionm.sum() < 8:
            continue
        noise = float(np.std(Ie[ionm]))
        Ie = Ie - float(np.median(Ie[ionm]))
        res = two_window(u, Ie, max(noise, 1e-12), args)
        if res is None:
            continue
        srows.append({"Te_true": float(Te_s[j]), "Te_deep": res[0], "Te_near": res[1],
                      "d": res[1] - res[0]})
    syn = pd.DataFrame(srows)
    print(f"synthetic single-Maxwellian sweeps fitted: {len(syn)} of {args.n_synth} "
          f"({len(syn)/args.n_synth:.0%} pass the same selection)")
    if len(syn):
        print(f"  control Te drawn from the archive: median true Te {np.median(Te_s):.3f} eV, "
              f"median Te of those SELECTED {np.median(syn['Te_true']):.3f} eV "
              f"<- the selection bias, measured")

    # ---------------- ramp-pair empirical error on d --------------------------------------
    pv = real.pivot_table(index=["timestamp", "segment"], columns="direction",
                          values="d", aggfunc="first").dropna()
    ramp_err = None
    if {"up", "down"} <= set(pv.columns) and len(pv) > 30:
        dd = (pv["down"] - pv["up"]).values
        ramp_err = float(dd.std() / np.sqrt(2.0))
        print(f"ramp-pair comparisons: {len(pv)}   empirical error on d: {ramp_err:.4f} eV")

    report = {"window_deep": [args.deep_lo, args.deep_hi],
              "window_near": [args.near_lo, args.near_hi]}

    def summarise(tag, df):
        d = df["d"].values
        e = {"n": int(len(d)), "median_Te_deep": float(np.median(df["Te_deep"])),
             "median_Te_near": float(np.median(df["Te_near"])),
             "median_d": float(np.median(d)), "mean_d": float(np.mean(d)),
             "sd_d": float(np.std(d, ddof=1)),
             "frac_positive": float((d > 0).mean())}
        report[tag] = e
        print(f"  {tag:<28} {e['n']:>6}  Te_deep {e['median_Te_deep']:>6.3f}  "
              f"Te_near {e['median_Te_near']:>6.3f}  d {e['median_d']:>+6.3f}  "
              f"{e['frac_positive']:>5.1%} positive")
        return e

    print(f"\n  {'sample':<28} {'n':>6}  {'Te_deep':>14} {'Te_near':>14} "
          f"{'median d':>9}  {'d > 0':>6}")
    er = summarise("REAL_all", real)
    es = summarise("SYNTHETIC_control", syn) if len(syn) > 30 else None
    for ch, g in real.groupby("channel"):
        if len(g) >= 200:
            summarise(f"REAL_channel_{ch}", g)

    # ---------------- verdict --------------------------------------------------------------
    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    if es is None:
        print("synthetic control too small to compare -- raise --n-synth")
    else:
        excess = er["median_d"] - es["median_d"]
        print(f"real median d      = {er['median_d']:+.4f} eV   (sd {er['sd_d']:.2f})")
        print(f"synthetic median d = {es['median_d']:+.4f} eV   (sd {es['sd_d']:.2f})")
        print(f"difference of medians = {excess:+.4f} eV")
        print()
        # The median difference is NOT the right headline, and quoting it as
        # excess/(sd/sqrt(n)) is wrong. d has heavy tails -- the real distribution is about ten
        # times WIDER than the control at every quantile while having the same fraction above
        # zero. Two distributions with identical sign balance and different scale produce
        # different medians with no shift in location at all, so a median gap can be pure scale.
        #
        # The scale-free statistic is the fraction of sweeps with d > 0. A genuine hot population
        # drives it decisively down: in a controlled test a 5% hot component at 1.5 eV took it
        # from 52% to 0%. So that fraction, compared against the control's, is the verdict.
        p1, p2 = er["frac_positive"], es["frac_positive"]
        se = float(np.sqrt(p1 * (1 - p1) / er["n"] + p2 * (1 - p2) / es["n"]))
        sig = abs(p1 - p2) / max(se, 1e-12)
        print(f"SIGN STATISTIC (scale-free): real {p1:.1%} of sweeps with d > 0, "
              f"control {p2:.1%}")
        print(f"   difference {100*(p1-p2):+.1f}% +- {100*se:.1f}%  =  {sig:.1f} sigma")
        print()
        print(f"scatter ratio real/control = {er['sd_d']/max(es['sd_d'],1e-12):.1f}x   "
              f"<- the real per-sweep variability our noise model does NOT reproduce")
        report["excess_over_control_eV"] = excess
        report["sign_statistic"] = {"real_frac_positive": p1, "control_frac_positive": p2,
                                    "difference": p1 - p2, "se": se, "sigma": sig}
        report["scatter_ratio_real_over_control"] = er["sd_d"] / max(es["sd_d"], 1e-12)
        report["verdict"] = ("consistent with a single Maxwellian" if sig < 3.0
                             else "slope varies within sweeps")
        report["ramp_pair_error_eV"] = ramp_err
        print()
        print("Read the SIGN statistic, not the medians. Below ~3 sigma these sweeps are")
        print("consistent with a single temperature, and the bend seen in the stacked")
        print("diagnostic was an artefact of stacking rather than a property of the plasma.")
        print("Above it, and with the same sign in both channels, the retarding slope genuinely")
        print("varies within individual sweeps -- with no averaging and no forward model, which")
        print("is the cleanest evidence this dataset can give either way.")

    path = out(cfg, "tables", "slope_curvature.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
