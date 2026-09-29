#!/usr/bin/env python3
"""
Step 9 — Is the second electron population real ambient plasma, or probe/lander photoelectrons?

WHY THIS MATTERS
----------------
The recovered hot component sits at 1.2-1.9 eV. Dove et al. (2012) measure a laboratory
photoelectron layer at 1.4 +- 0.3 eV, and the RAMBHA-LP probe sits ~2 m above sunlit regolith.
So the hot component may not be lunar plasma at all -- it may be electrons photo-emitted from the
probe and lander surfaces. Those are two completely different claims about the Moon, and the
report must not pick one by assertion.

THE DISCRIMINATING PHYSICS
--------------------------
Photoemission current is set by solar flux and the emitting surface area. It does NOT scale with
the surrounding plasma density. An ambient hot population does: both populations are drawn from
the same plasma, so their densities rise and fall together.

That gives a clean, quantitative separation. Fit

    log10(Nh) = a * log10(Nc) + b

    slope a ~ 1  -> hot density tracks cold density   -> AMBIENT hot plasma
    slope a ~ 0  -> hot density is independent of it  -> PHOTOELECTRONS (a fixed current)

Equivalently: if the hot component is photoemission, the hot FRACTION must fall as the plasma
gets denser, because a fixed emitted current is being divided by a larger plasma density.

Four tests are run. Test A is the decisive one; B, C and D are independent corroboration so the
verdict does not rest on a single statistic.

  A. slope of log10(Nh) against log10(Nc), and the sign of corr(hot fraction, Nc)
  B. the published Th/Tc >= 1.7 separability gate (Yip et al. 2020) against our 1.5
  C. does Nh track the CONSTANT offset of the ion branch? Photoemission enters a sweep as a
     bias-independent additive current (Manju et al.; Johansson et al. 2017), so a sweep with
     more photoemission has a more negative ion-region floor. If the hot component is
     photoelectrons, Nh should follow that floor. If it is plasma, it should not.
  D. per-day hot fraction against per-day density. Solar elevation at the landing site rose then
     fell over the mission while density rose monotonically, so the two hypotheses predict
     different day-level shapes.

Test A is run on BOTH the network output and the independent classical decomposition, but they
do NOT carry equal weight, and it matters why.

**The network is trained on a prior that already assumes the ambient answer.** synthetic.py draws
a hot FRACTION and sets Ne_h = Ne_cold * frac / (1 - frac), so in every training example the hot
density is proportional to the cold one -- a built-in slope of exactly 1. Verified numerically:
that construction yields slope +1.015 and corr(log Nc, fraction) = +0.013, while a fixed
photoemitted density yields slope -0.007 and corr = -0.919. So the network is biased toward
reporting "ambient" no matter what the sweeps say.

That makes the classical decomposition the PRIMARY evidence here -- it has no such prior. The
network's slope is still informative, but only in one direction: a slope clearly BELOW 1 is
meaningful, because it means the real curves are pushing the network away from its own prior. A
slope near 1 from the network proves nothing at all.

A degeneracy in either method alone could also manufacture a correlation, so agreement between
the two -- which share no code -- is what makes a result physical.

Run AFTER a completed --two-population pipeline run.

Outputs:
  outputs_2pop/tables/photoelectron_test.json
"""
import _bootstrap as B
import argparse
import json
import numpy as np
import pandas as pd
from rambhalp.config import load_config, out
from rambhalp import twopop, classical as classical_mod


def _fit_slope(x, y):
    """Least-squares slope, intercept and Pearson r of y on x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return None
    x, y = x[m], y[m]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, icept = np.linalg.lstsq(A, y, rcond=None)[0]
    r = float(np.corrcoef(x, y)[0, 1])
    # bootstrap the slope so the verdict carries an error bar
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(400):
        k = rng.integers(0, x.size, x.size)
        Ab = np.vstack([x[k], np.ones_like(k, float)]).T
        boots.append(np.linalg.lstsq(Ab, y[k], rcond=None)[0][0])
    return {"slope": float(slope), "slope_sd": float(np.std(boots)),
            "intercept": float(icept), "pearson_r": r, "n": int(x.size)}


def _ion_floor_fraction(sw):
    """The constant part of the ion branch, as a fraction of the curve's span.

    Photoemission is bias-independent, so it shows up as a flat offset in the most negative
    part of the sweep. Taking the MEDIAN of the lowest-bias tenth (not the minimum) keeps this
    robust to noise spikes.
    """
    V, I = np.asarray(sw.V, float), np.asarray(sw.I, float)
    if V.size < 20:
        return np.nan
    order = np.argsort(V)
    V, I = V[order], I[order]
    k = max(3, V.size // 10)
    floor = float(np.median(I[:k]))
    span = float(np.ptp(I))
    if not np.isfinite(span) or span <= 0:
        return np.nan
    return floor / span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=10,
                    help="number of stacks for the density and ion-floor tests")
    ap.add_argument("--min-stack", type=int, default=200,
                    help="minimum sweeps per stack; averaging N cuts the noise by sqrt(N)")
    args = ap.parse_args()

    cfg = load_config()
    cfg["synthetic"]["two_population"] = True
    from pathlib import Path as _P
    base = _P(cfg["_solution_dir"]) / "outputs_2pop"
    for key, sub in (("outputs", ""), ("figures", "figures"),
                     ("tables", "tables"), ("checkpoints", "checkpoints")):
        q = base / sub if sub else base
        q.mkdir(parents=True, exist_ok=True)
        cfg["paths"]["_" + key + "_abs"] = q

    print("=" * 70)
    print("STEP 9: is the hot population plasma, or photoelectrons?")
    print("=" * 70)

    ml = pd.read_csv(out(cfg, "tables", "ml_predictions.csv"))
    ml = ml[ml["recovered"].astype(bool)].copy()
    ml = ml[(ml["Ne_cc"] > 0) & (ml["Nh_cc"] > 0)]
    ml["frac"] = ml["Nh_cc"] / (ml["Ne_cc"] + ml["Nh_cc"])
    print(f"network rows with a hot component: {len(ml)}")

    report = {}

    # ---------------- TEST A (network) ----------------------------------------------------
    a_net = _fit_slope(np.log10(ml["Ne_cc"]), np.log10(ml["Nh_cc"]))
    r_frac_net = float(np.corrcoef(np.log10(ml["Ne_cc"]), ml["frac"])[0, 1])
    report["A_network"] = {**(a_net or {}), "corr_logNc_vs_hotfraction": r_frac_net}
    print(f"\nTEST A (network)   slope d log Nh / d log Nc = "
          f"{a_net['slope']:+.3f} +- {a_net['slope_sd']:.3f}   r = {a_net['pearson_r']:+.3f}")
    print(f"                   corr(log Nc, hot fraction) = {r_frac_net:+.3f}")

    # ---------------- classical decomposition, on STACKED curves -------------------------
    # Why stacked and not per-sweep: the hot tail lives 1.5-2.6 V below Vp, where the electron
    # current is exp(-d/Te) of its peak -- 2.4e-2 at 1.5 V and 6.7e-3 at 2.0 V for Te = 0.4 eV.
    # The measured noise floor is 0.2% of span on the quiet channels and 1.8% on the 20 Mohm
    # setting. So in a SINGLE sweep the hot tail is at or under the noise: a gate trace over
    # 1200 sweeps found 679 with ZERO points in the hot band above 3 sigma, and zero successful
    # per-sweep decompositions.
    #
    # That is a genuine result, not an obstacle to route around: the second population is a
    # POPULATION-LEVEL feature of this archive, not a per-sweep measurement. It is also exactly
    # how it was found in the first place -- the local-Te diagnostic stacked 1403 sweeps.
    #
    # So the independent test is run on stacked curves. Sweeps are grouped by classical density,
    # aligned on u = V - Vp, and averaged; averaging N sweeps cuts the noise by sqrt(N) and
    # brings the tail above the floor. Each stack then gets one decomposition, and the slope of
    # log Nh against log Nc is measured ACROSS the density bins.
    sweeps = B.load_pickle(B.cache_dir(cfg) / "sweeps.pkl")
    cdf = pd.read_csv(out(cfg, "tables", "classical_baseline.csv"))

    import dataclasses
    fields = {f.name for f in dataclasses.fields(classical_mod.ClassicalResult)}
    usable = [c for c in cdf.columns if c in fields]
    ok = cdf[~cdf["failed"].astype(bool)]
    cmap = {(r["timestamp"], r["idx"]): classical_mod.ClassicalResult(**{k: r[k] for k in usable})
            for _, r in ok.iterrows()}

    # V_float is nan for a third of good fits (the single-population fit does not need it) and
    # fit_two_populations requires it. Recover it from the flux-balance relation already used
    # elsewhere in the project: Vp - V_float = Te * ln( sqrt( m_i / (2 pi m_e) ) ).
    pr = cfg["probe"]
    m_i = float(pr.get("m_i_amu", 1.0)) * 1.66053906660e-27
    K = float(np.log(np.sqrt(m_i / (2.0 * np.pi * pr["m_e"]))))
    n_est = 0
    for r in cmap.values():
        if not np.isfinite(r.V_float) and np.isfinite(r.Vp_V) and np.isfinite(r.Te_eV):
            r.V_float = float(r.Vp_V - r.Te_eV * K)
            n_est += 1
    print(f"V_float recovered from flux balance for {n_est} of {len(cmap)} classical fits")

    # Collect aligned curves
    # The grid must reach well below V_float - ion_floor_offset, or fit_two_populations finds
    # no ion region and declines. V_float sits about Te*ln(sqrt(m_i/2*pi*m_e)) ~ 1.1 V below Vp
    # and the offset is another 2 V, so a grid starting at u = -3.0 misses it by a hair and
    # every stack silently returns None. Real sweeps reach u ~ -8, so use that.
    UG = np.arange(-8.0, 2.0001, 0.05)          # u = V - Vp grid
    recs = []
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
        if u.min() > -2.2 or u.max() < 0.5:
            continue
        recs.append({"date": sw.date, "Ne": r.Ne_cc, "Te": r.Te_eV, "Vp": r.Vp_V,
                     "floor": _ion_floor_fraction(sw),
                     "Ie": np.interp(UG, u, Ie, left=np.nan, right=np.nan)})
    print(f"sweeps aligned on u = V - Vp: {len(recs)}")

    def decompose_stack(group, label):
        """Average a group's aligned curves and decompose the result."""
        if len(group) < args.min_stack:
            return None
        M = np.vstack([g["Ie"] for g in group])
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(M, axis=0)
        good = np.isfinite(mean)
        if good.sum() < 40:
            return None
        Vp = float(np.median([g["Vp"] for g in group]))
        Te = float(np.median([g["Te"] for g in group]))
        Vs, Is = UG[good] + Vp, mean[good]

        class _S:                                  # minimal stand-in for a Sweep
            pass
        st = _S(); st.V, st.I = Vs, Is
        st.date = label; st.timestamp = label; st.idx = 0

        base_r = classical_mod.ClassicalResult(
            date=label, timestamp=label, idx=0, adc_channel=-1, t_start="",
            V_float=float(Vp - Te * K), Te_eV=Te, Ne_cc=float(np.median([g["Ne"] for g in group])),
            r2=1.0, failed=False, fail_reason="")
        if "Vp_V" in fields:
            base_r.Vp_V = Vp
        for band, minpts in ((None, None), ((-2.6, -0.8), 5)):
            try:
                tp = twopop.fit_two_populations(st, cfg, base=base_r,
                                               hot_band=band, min_hot_pts=minpts)
            except Exception:
                tp = None
            if tp and tp["Nc_cc"] > 0 and tp["Nh_cc"] > 0:
                tp["_n_stacked"] = len(group)
                tp["_band"] = "default" if band is None else "relaxed"
                return tp
        return None

    # ---------------- TEST A (classical, stacked by density bin) --------------------------
    recs_sorted = sorted([r for r in recs if np.isfinite(r["Ne"]) and r["Ne"] > 0],
                         key=lambda r: r["Ne"])
    nb = args.bins
    edges = np.linspace(0, len(recs_sorted), nb + 1).astype(int)
    stacks = []
    for i in range(nb):
        grp = recs_sorted[edges[i]:edges[i + 1]]
        if not grp:
            continue
        tp = decompose_stack(grp, f"nebin{i}")
        if tp:
            tp["_Ne_bin_median"] = float(np.median([g["Ne"] for g in grp]))
            stacks.append(tp)
    print(f"\ndensity-binned stacks decomposed: {len(stacks)} of {nb}")
    if stacks:
        print(f"   {'Nc (cc)':>9} {'Nh (cc)':>9} {'Tc (K)':>8} {'Th (K)':>8} "
              f"{'hotfrac':>8} {'n':>6} {'band':>8}")
        for t in stacks:
            print(f"   {t['Nc_cc']:>9.1f} {t['Nh_cc']:>9.2f} {t['Tc_eV']*11604:>8.0f} "
                  f"{t['Th_eV']*11604:>8.0f} {t['hot_fraction']:>8.3f} "
                  f"{t['_n_stacked']:>6d} {t['_band']:>8}")
    if len(stacks) >= 4:
        Nc = np.array([t["Nc_cc"] for t in stacks]); Nh = np.array([t["Nh_cc"] for t in stacks])
        fr = np.array([t["hot_fraction"] for t in stacks])
        a_cl = _fit_slope(np.log10(Nc), np.log10(Nh))
        r_fr = float(np.corrcoef(np.log10(Nc), fr)[0, 1])
        report["A_classical_stacked"] = {**(a_cl or {}), "corr_logNc_vs_hotfraction": r_fr,
                                        "n_bins": len(stacks)}
        print(f"\nTEST A (classical, stacked) slope d log Nh / d log Nc = "
              f"{a_cl['slope']:+.3f} +- {a_cl['slope_sd']:.3f}   r = {a_cl['pearson_r']:+.3f}")
        print(f"                            corr(log Nc, hot fraction) = {r_fr:+.3f}")

        # ---------------- TEST B: the published separability gate -------------------------
        ratio = np.array([t["Th_eV"] / max(t["Tc_eV"], 1e-9) for t in stacks])
        report["B_separability"] = {
            "n_stacks": int(len(ratio)), "median_Th_over_Tc": float(np.median(ratio)),
            "frac_passing_1.5_ours": float((ratio >= 1.5).mean()),
            "frac_passing_1.7_Yip2020": float((ratio >= 1.7).mean())}
        print(f"\nTEST B  median Th/Tc = {np.median(ratio):.2f}   "
              f"pass 1.5 (ours) {(ratio>=1.5).mean():.0%}   "
              f"pass 1.7 (Yip 2020) {(ratio>=1.7).mean():.0%}")

    # ---------------- TEST C: stacks binned by ion-branch offset -------------------------
    # Photoemission is bias-independent, so it appears as a flat offset in the ion branch.
    # If the hot component is photoelectrons, Nh should rise with that offset while Nc, the
    # control, should not.
    fl = sorted([r for r in recs if np.isfinite(r["floor"])], key=lambda r: r["floor"])
    fstacks = []
    fedges = np.linspace(0, len(fl), args.bins + 1).astype(int)
    for i in range(args.bins):
        grp = fl[fedges[i]:fedges[i + 1]]
        if not grp:
            continue
        tp = decompose_stack(grp, f"floorbin{i}")
        if tp:
            tp["_floor_median"] = float(np.median([g["floor"] for g in grp]))
            fstacks.append(tp)
    if len(fstacks) >= 4:
        fo = np.array([t["_floor_median"] for t in fstacks])
        print(f"\nTEST C  ion-floor-binned stacks: {len(fstacks)}")
        print(f"   {'floor':>8} {'Nc (cc)':>9} {'Nh (cc)':>9} {'hotfrac':>8}")
        for t in fstacks:
            print(f"   {t['_floor_median']:>8.4f} {t['Nc_cc']:>9.1f} {t['Nh_cc']:>9.2f} "
                  f"{t['hot_fraction']:>8.3f}")
        c_h = _fit_slope(fo, np.log10([t["Nh_cc"] for t in fstacks]))
        c_c = _fit_slope(fo, np.log10([t["Nc_cc"] for t in fstacks]))
        report["C_ion_floor"] = {"Nh_vs_floor": c_h, "Nc_vs_floor_control": c_c}
        if c_h and c_c:
            print(f"   corr(floor, log Nh) = {c_h['pearson_r']:+.3f}   "
                  f"[control, log Nc = {c_c['pearson_r']:+.3f}]")

    # ---------------- TEST D: per-day stacks ---------------------------------------------
    dstacks = []
    for d in sorted({r["date"] for r in recs}):
        grp = [r for r in recs if r["date"] == d]
        tp = decompose_stack(grp, d)
        if tp:
            tp["_date"] = d
            dstacks.append(tp)
    if dstacks:
        print(f"\nTEST D  per-day stacks: {len(dstacks)}")
        print(f"   {'date':>10} {'Nc (cc)':>9} {'Nh (cc)':>9} {'hotfrac':>8} {'n':>6}")
        for t in dstacks:
            print(f"   {t['_date']:>10} {t['Nc_cc']:>9.1f} {t['Nh_cc']:>9.2f} "
                  f"{t['hot_fraction']:>8.3f} {t['_n_stacked']:>6d}")
        report["D_by_day"] = [{"date": t["_date"], "Nc_cc": t["Nc_cc"], "Nh_cc": t["Nh_cc"],
                               "hot_fraction": t["hot_fraction"], "n": t["_n_stacked"]}
                              for t in dstacks]

    # ---------------- verdict --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HOW TO READ THIS")
    print("=" * 70)
    print("TEST A slope near 1 -> hot density tracks the plasma      -> AMBIENT population.")
    print("TEST A slope near 0 -> hot density is a fixed current     -> PHOTOELECTRONS.")
    print("A negative corr(log Nc, hot fraction) supports photoelectrons: a constant emitted")
    print("current is a smaller share of a denser plasma.")
    print()
    print("WEIGHTING: the STACKED CLASSICAL slope is the primary evidence. The network was")
    print("trained on synthetic data built with Nh proportional to Nc -- a hard-wired slope of")
    print("1 -- so a network slope near 1 may just be its prior. A network slope clearly BELOW")
    print("1 is meaningful: the real sweeps are then overriding that prior. The classical fit")
    print("shares no code and carries no such prior.")
    print()
    print("TEST C is the most specific: photoemission is a bias-independent offset, so if the")
    print("hot component is photoelectrons, Nh should track the ion-branch floor while Nc")
    print("(the control) should not.")

    path = out(cfg, "tables", "photoelectron_test.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
