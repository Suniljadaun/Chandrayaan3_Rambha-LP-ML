#!/usr/bin/env python3
"""
Step 14 — Does the Te refinement head help on REAL sweeps, not just synthetic ones?

WHY THIS SCRIPT EXISTS AND WHY IT IS NOT OPTIONAL
-------------------------------------------------
scripts/13_vp_decoupling.py established, on synthetic hold-out data at full training budget over
three seeds, that aligning the curve on the plasma potential and normalising by the density
halves the temperature error: 22.8% +- 1.5% -> 11.8% +- 0.9%, a factor of 1.93 at 10.7 sigma.

That is a synthetic result, and this project has already been burned by exactly that distinction.
The physics-loss ablation (scripts/08) found a term that looks actively harmful on synthetic
hold-out — removing it improved synthetic Te from 0.078 to 0.055 eV — while buying ten percentage
points of real-archive recovery and a third of the ramp-pair systematic. Judged on synthetic
accuracy alone, the correct decision reversed.

So the refinement is not adopted until the real-data checks agree. Two of them bear on Te and
neither needs ground truth:

  1. RAMP-PAIR SCATTER IN Te. Each commanded sweep is a triangle, so its rising and falling
     halves are two independent measurements of the same plasma seconds apart. Their
     disagreement is a repeatability measure derived from the instrument, not from a model. If
     the refined Te is genuinely better it must be more repeatable here. This is the decisive
     test.
  2. AGREEMENT WITH THE CLASSICAL METHOD. On sweeps the classical fit handles, the refined Te
     should not drift away from an independent estimator that shares no code with it.

A refinement that improves synthetic accuracy while worsening ramp scatter is overfitting the
simulator, and should be rejected however good the synthetic number looks.

Outputs:
  outputs/tables/te_refine_eval.json
"""
import _bootstrap as B
import argparse
import dataclasses
import json
import numpy as np
import pandas as pd
from rambhalp.config import load_config, out
from rambhalp import train, infer, synthetic, te_refine, classical as classical_mod
from rambhalp.preprocess import common_grid
from rambhalp.model import predict


def ramp_scatter(rows, key):
    """Median |difference| in `key` between the two halves of the same commanded sweep."""
    df = pd.DataFrame(rows)
    if "direction" not in df or "segment" not in df:
        return None
    p = df.pivot_table(index=["timestamp", "segment"], columns="direction",
                       values=key, aggfunc="first").dropna()
    if not {"up", "down"} <= set(p.columns) or len(p) < 30:
        return None
    d = (p["down"] - p["up"]).values
    return {"n_pairs": int(len(d)), "median_abs_diff": float(np.median(np.abs(d))),
            "median_signed": float(np.median(d)), "sd": float(np.std(d, ddof=1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-synth", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--u-window", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="alignment window in volts relative to Vp. The default (-2.0 0.6) "
                         "spans the region where this archive is known to depart from the "
                         "forward model; try e.g. --u-window -2.6 -0.8 to read Te from below it.")
    args = ap.parse_args()

    cfg = load_config()
    cfg["synthetic"]["two_population"] = False
    V = np.asarray(common_grid(cfg), float)
    if args.u_window:
        te_refine.set_window(*args.u_window)
    print(f"alignment window: u in [{te_refine.U_LO}, {te_refine.U_HI}] V, "
          f"{te_refine.U_N} points")

    print("=" * 76)
    print("STEP 14: does the Te refinement head survive contact with real sweeps?")
    print("=" * 76)

    ckpt = out(cfg, "checkpoints", "pinn.pt")
    model = train.load_checkpoint(cfg, ckpt)
    print(f"main model: {ckpt.name}")

    # ---- train the refinement head on Vp predicted by the main model --------------------
    Xtr, Ytr, _ = synthetic.make_dataset(args.n_synth, cfg, seed=4321)
    vp_tr = predict(model, Xtr, cfg, mc_samples=1)["Vp_V"]
    head, _ = te_refine.train_head(Xtr, Ytr, vp_tr, V, cfg, epochs=args.epochs)

    report = {}

    # ---- 1. synthetic hold-out (the number scripts/13 already established) ---------------
    Xv, Yv, _ = synthetic.make_dataset(4000, cfg, seed=8765)
    pv = predict(model, Xv, cfg, mc_samples=1)
    te_ref, _ = te_refine.predict_te(head, Xv, V, pv["Vp_V"])
    band = (Yv[:, 1] >= 0.30) & (Yv[:, 1] < 0.50)
    rel = lambda p: 100 * float(np.median(np.abs(p[band] - Yv[band, 1]) / Yv[band, 1]))
    report["synthetic"] = {
        "baseline_abs_eV": float(np.median(np.abs(pv["Te_eV"] - Yv[:, 1]))),
        "refined_abs_eV": float(np.median(np.abs(te_ref - Yv[:, 1]))),
        "baseline_pct_in_band": rel(pv["Te_eV"]), "refined_pct_in_band": rel(te_ref),
        "n_in_band": int(band.sum()),
    }
    s = report["synthetic"]
    print(f"\nsynthetic hold-out   baseline {s['baseline_abs_eV']:.4f} eV "
          f"({s['baseline_pct_in_band']:.1f}% in 0.3-0.5 band)")
    print(f"                     refined  {s['refined_abs_eV']:.4f} eV "
          f"({s['refined_pct_in_band']:.1f}% in band)")

    # ---- 2. real sweeps -------------------------------------------------------------------
    sweeps = B.load_pickle(B.cache_dir(cfg) / "sweeps.pkl")
    Xr, _elec = infer.prepare_input(sweeps, cfg)
    pr = predict(model, Xr, cfg, mc_samples=1)
    te_ref_r, _ = te_refine.predict_te(head, Xr, V, pr["Vp_V"])

    rows = [{"timestamp": sw.timestamp, "idx": sw.idx, "segment": int(sw.segment),
             "direction": sw.direction, "Te_base": float(pr["Te_eV"][i]),
             "Te_ref": float(te_ref_r[i])} for i, sw in enumerate(sweeps)]

    rb, rr = ramp_scatter(rows, "Te_base"), ramp_scatter(rows, "Te_ref")
    report["ramp_consistency"] = {"baseline": rb, "refined": rr}
    if rb and rr:
        print(f"\nRAMP-PAIR SCATTER IN Te  (the decisive real-data test, {rb['n_pairs']} pairs)")
        print(f"   baseline  median |Te_down - Te_up| = {rb['median_abs_diff']:.4f} eV")
        print(f"   refined   median |Te_down - Te_up| = {rr['median_abs_diff']:.4f} eV")
        ratio = rr["median_abs_diff"] / max(rb["median_abs_diff"], 1e-12)
        report["ramp_ratio_refined_over_baseline"] = ratio
        print(f"   ratio refined/baseline = {ratio:.2f}   "
              f"({'BETTER' if ratio < 0.95 else 'WORSE' if ratio > 1.05 else 'no change'})")

    # ---- 3. agreement with the classical method -------------------------------------------
    cdf = pd.read_csv(out(cfg, "tables", "classical_baseline.csv"))
    fields = {f.name for f in dataclasses.fields(classical_mod.ClassicalResult)}
    usable = [c for c in cdf.columns if c in fields]
    ok = cdf[~cdf["failed"].astype(bool)]
    cmap = {(r["timestamp"], r["idx"]): float(r["Te_eV"]) for _, r in ok.iterrows()
            if np.isfinite(r["Te_eV"])}
    db, dr = [], []
    for i, sw in enumerate(sweeps):
        t = cmap.get((sw.timestamp, sw.idx))
        if t is None:
            continue
        db.append(pr["Te_eV"][i] - t); dr.append(te_ref_r[i] - t)
    if len(db) > 100:
        report["agreement_with_classical"] = {
            "n": len(db),
            "baseline_median_diff_eV": float(np.median(db)),
            "refined_median_diff_eV": float(np.median(dr)),
            "baseline_median_abs_eV": float(np.median(np.abs(db))),
            "refined_median_abs_eV": float(np.median(np.abs(dr))),
        }
        a = report["agreement_with_classical"]
        print(f"\nAGREEMENT WITH CLASSICAL Te  ({a['n']} overlapping sweeps)")
        print(f"   baseline  median diff {a['baseline_median_diff_eV']:+.4f} eV   "
              f"median |diff| {a['baseline_median_abs_eV']:.4f} eV")
        print(f"   refined   median diff {a['refined_median_diff_eV']:+.4f} eV   "
              f"median |diff| {a['refined_median_abs_eV']:.4f} eV")

    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    print("Adopt ONLY if the ramp-pair scatter improves or holds. A refinement that sharpens")
    print("synthetic accuracy while worsening ramp repeatability has learned the simulator, not")
    print("the instrument -- which is precisely the trap the physics-loss ablation set earlier")
    print("in this project, and in that case the synthetic number should be ignored.")

    path = out(cfg, "tables", "te_refine_eval.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
