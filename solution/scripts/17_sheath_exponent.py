#!/usr/bin/env python3
"""
Step 17 — Choose the sheath-expansion exponent from the data alone, and check the fit window.

WHY THIS SCRIPT EXISTS
----------------------
probe.sheath_exponent = 0.40 sets how the electron current grows above the plasma potential,
and every classical and network result depends on it. It was first chosen during development,
partly because it made the classical fit reproduce the values published for 2 Sep 2023. A value
chosen by matching the published numbers cannot then be used to validate against them, so this
script chooses it again using nothing but the sweeps.

1. EXPONENT. For each exponent on a grid, run the classical fit on every sweep and score how well
   the fitted saturation law I_e0 * (1 + (V - Vp)/Te)**alpha describes the measured current above
   Vp (R^2 on the linear current). Compared on the sweeps that the classical fit passes at EVERY
   exponent, so each exponent is judged on the same curves. Reported overall and per probe
   resistor setting, since an instrumental cause would differ between the three resistor chains.

2. WINDOW. The retardation fit window is 1.0 V above V_float (classical.retardation_span). Re-run
   the classical fit at 0.8, 1.0 and 1.5 V and compare density and temperature on the sweeps that
   pass at all three, to show the reported values do not depend on that choice.

Uses only the cached sweeps from step 1; no network and no published values.

Outputs:
  outputs/tables/sheath_exponent.json
"""
import _bootstrap as B
import copy
import json
import numpy as np
from rambhalp.config import load_config, out
from rambhalp import classical

ALPHAS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
WINDOWS = [0.8, 1.0, 1.5]


def electron_current(sw, fit, cfg):
    """Sorted V and the baseline-subtracted electron current, exactly as fit_sweep builds them."""
    V, I = np.asarray(sw.V, float), np.asarray(sw.I, float)
    o = np.argsort(V)
    V, I = V[o], I[o]
    if np.corrcoef(V, I)[0, 1] < 0:
        I = -I
    vf = fit.V_float
    bm = np.zeros(V.size, dtype=bool)
    if np.isfinite(vf):
        bm = V <= vf - cfg["classical"]["ion_floor_offset"]
    if bm.sum() < 3:
        bm[:] = False
        bm[:max(3, V.size // 10)] = True
    return V, I - float(np.median(I[bm]))


def saturation_r2(sw, cfg, alpha):
    """R^2 of the fitted saturation law against the measured current above Vp; NaN if the fit fails."""
    c = copy.deepcopy(cfg)
    c["probe"]["sheath_exponent"] = alpha
    fit = classical.fit_sweep(sw, c)
    if fit.failed or not np.isfinite(fit.Vp_V):
        return np.nan
    V, Ie = electron_current(sw, fit, c)
    m = V >= fit.Vp_V
    if m.sum() < 10:
        return np.nan
    model = fit.Ie0_A * (1.0 + (V[m] - fit.Vp_V) / fit.Te_eV) ** alpha
    ss = np.sum((Ie[m] - Ie[m].mean()) ** 2)
    return float(1.0 - np.sum((Ie[m] - model) ** 2) / ss) if ss > 0 else np.nan


def exponent_scan(sweeps, cfg):
    R = np.array([[saturation_r2(s, cfg, a) for a in ALPHAS] for s in sweeps])
    common = np.all(np.isfinite(R), axis=1)
    Rc = R[common]
    best = np.array(ALPHAS)[np.argmax(Rc, axis=1)]
    res = np.array([s.probe_res for s in sweeps])[common]

    def summary(mask):
        b = best[mask]
        return {
            "n": int(mask.sum()),
            "median_r2_by_alpha": {str(a): round(float(np.median(Rc[mask, i])), 4)
                                   for i, a in enumerate(ALPHAS)},
            "alpha_with_highest_median_r2": float(ALPHAS[int(np.argmax(np.median(Rc[mask], axis=0)))]),
            "best_alpha_per_sweep_median": float(np.median(b)),
            "best_alpha_per_sweep_iqr": [float(x) for x in np.percentile(b, [25, 75])],
            "best_alpha_counts": {str(a): int((b == a).sum()) for a in ALPHAS},
        }

    out_ = {"n_sweeps_scanned": len(sweeps), "all": summary(np.ones(len(Rc), dtype=bool)),
            "by_probe_resistor_ohm": {}}
    for r in sorted(set(res)):
        out_["by_probe_resistor_ohm"][f"{r:.0f}"] = summary(res == r)
    return out_


def window_scan(sweeps, cfg):
    fits = {}
    for w in WINDOWS:
        c = copy.deepcopy(cfg)
        c["classical"]["retardation_span"] = w
        fits[w] = [classical.fit_sweep(s, c) for s in sweeps]
    passed = {w: np.array([not f.failed for f in fits[w]]) for w in WINDOWS}
    common = np.logical_and.reduce([passed[w] for w in WINDOWS])
    ref = 1.0
    ne_ref = np.array([f.Ne_cc for f in fits[ref]])[common]
    te_ref = np.array([f.Te_eV for f in fits[ref]])[common]
    rows = {}
    for w in WINDOWS:
        ne = np.array([f.Ne_cc for f in fits[w]])[common]
        te = np.array([f.Te_eV for f in fits[w]])[common]
        rows[str(w)] = {
            "n_pass": int(passed[w].sum()),
            "median_Ne_cc_common": round(float(np.median(ne)), 1),
            "median_Te_eV_common": round(float(np.median(te)), 3),
            "median_Ne_ratio_to_1V": round(float(np.median(ne / ne_ref)), 4),
            "median_Te_diff_to_1V_eV": round(float(np.median(te - te_ref)), 4),
        }
    return {"n_common": int(common.sum()), "windows_V": rows}


def main():
    cfg = load_config()
    print("=" * 70)
    print("STEP 17: sheath exponent from the data alone; fit-window sensitivity")
    print("=" * 70)
    sweeps = B.load_pickle(B.cache_dir(cfg) / "sweeps.pkl")
    report = {"exponent": exponent_scan(sweeps, cfg), "window": window_scan(sweeps, cfg),
              "configured_sheath_exponent": cfg["probe"]["sheath_exponent"],
              "configured_retardation_span_V": cfg["classical"]["retardation_span"]}
    path = out(cfg, "tables", "sheath_exponent.json")
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
