#!/usr/bin/env python3
"""
Step 8 — Ablate the physics-informed loss term.

WHY THIS EXISTS
---------------
The network's loss is  w_sup * supervised + w_phys * physics_reconstruction, with w_phys = 0.5.
Controlled experiments on synthetic data (20000 sweeps, 120 epochs, lr 3e-3, identical seeds)
found that the physics term makes accuracy WORSE, in both modes and on every parameter:

    three-parameter   Te  0.144 -> 0.066 eV   Vp 0.248 -> 0.133 V   (w_phys 0.5 -> 0.0)
    five-parameter    Tc  0.239 -> 0.109 eV   Vp 0.359 -> 0.169 V   (w_phys 0.5 -> 0.0)

That measurement is on SYNTHETIC hold-out only, where the labels are exact by construction and
a reconstruction term can only add bias. It cannot settle the question, because the physics
term's real justification was never synthetic accuracy -- it was sim-to-real robustness, which
a synthetic hold-out is structurally incapable of measuring.

This script settles it on the archive. It retrains at several w_phys values and scores each on
the REAL-data checks alongside the synthetic one, so the trade (if there is one) is visible:

    synthetic hold-out     - best-case accuracy, favours w_phys = 0
    agreement w/ classical - does the network still match an independent method on real sweeps?
    ramp consistency       - do the two halves of one real triangle sweep still agree?
    recovery fraction      - does it still fit the sweeps the classical method cannot?

If the real-data checks hold up as w_phys falls, the physics term is costing accuracy and
buying nothing, and should go. If they degrade, it is buying robustness and should stay
despite the synthetic cost. Either result is a finding worth reporting.

Requires a completed run (cached sweeps + classical baseline + synthetic set) in the same
output tree. Run scripts/run_all.py first, then this.

Outputs:
  <outputs>/tables/physics_ablation.json
"""
import _bootstrap as B
import argparse
import copy
import json
import numpy as np
import pandas as pd
from rambhalp.config import load_config, out
from rambhalp import train, infer, validate, classical as classical_mod


WEIGHTS = [0.5, 0.25, 0.1, 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-population", action="store_true",
                    help="ablate the five-parameter network instead of the three-parameter one")
    ap.add_argument("--weights", type=float, nargs="+", default=WEIGHTS)
    ap.add_argument("--repeats", type=int, default=1,
                    help="retrain each weight this many times with different seeds and report "
                         "the spread. A single run cannot distinguish a real 2-point difference "
                         "in recovery from seed noise; 3 repeats can.")
    args = ap.parse_args()

    cfg = load_config()
    if args.two_population:
        # Same redirection run_all.py uses, so this reads the tree that run produced.
        # Rebuilding the paths by string-replacing "outputs" would also rewrite any earlier
        # occurrence of that word in the absolute path.
        from pathlib import Path as _P
        cfg["synthetic"]["two_population"] = True
        base = _P(cfg["_solution_dir"]) / "outputs_2pop"
        for key, sub in (("outputs", ""), ("figures", "figures"),
                         ("tables", "tables"), ("checkpoints", "checkpoints")):
            q = base / sub if sub else base
            q.mkdir(parents=True, exist_ok=True)
            cfg["paths"]["_" + key + "_abs"] = q

    print("=" * 70)
    print("STEP 8: physics-loss ablation")
    print(f"mode: {'five-parameter' if args.two_population else 'three-parameter'}")
    print(f"weights: {args.weights}")
    print("=" * 70)

    cd = B.cache_dir(cfg)
    tr = np.load(cd / "synth_train.npz")
    X, Y = tr["X"], tr["Y"]
    sweeps = B.load_pickle(cd / "sweeps.pkl")

    cdf = pd.read_csv(out(cfg, "tables", "classical_baseline.csv"))
    classical_rows = [classical_mod.ClassicalResult(**{
        k: row[k] for k in ["date", "timestamp", "idx", "adc_channel", "t_start",
                            "V_float", "Te_eV", "Ne_cc", "r2", "failed", "fail_reason"]
    }) for _, row in cdf.iterrows()]
    fail_keys = set(zip(cdf.loc[cdf["failed"], "timestamp"], cdf.loc[cdf["failed"], "idx"]))

    results = {}
    for w in args.weights:
        print(f"\n--- w_physics = {w} " + "-" * 46)
        c = copy.deepcopy(cfg)
        c["model"]["w_physics"] = float(w)

        runs = []
        for rep in range(args.repeats):
            # Vary the seed THROUGH THE CONFIG, not by calling torch.manual_seed here.
            # train_model() runs torch.manual_seed(cfg["train"]["seed"]) itself as its first
            # act, so any seeding done outside it is overwritten and every repeat comes back
            # bit-identical -- which is exactly what the first --repeats run produced (sd
            # 0.0000 on every metric). A zero spread reported that way looks like a
            # reassuringly stable model and is actually a broken experiment, so it is worth
            # being explicit about the cause here.
            c["train"]["seed"] = int(cfg["train"]["seed"]) + rep
            model, hist = train.train_model(X, Y, c, verbose=False)
            ml_rows = infer.apply(model, sweeps, c)
            n_rec = sum(1 for m in ml_rows
                        if m.get("recovered") and (m["timestamp"], m["idx"]) in fail_keys)
            runs.append({
                "synthetic_holdout": validate.synthetic_holdout_error(model, c),
                "agreement_with_classical": validate.agreement_with_classical(
                    ml_rows, classical_rows),
                "ramp_consistency": validate.ramp_consistency(ml_rows),
                "recovered": int(n_rec),
            })
            if args.repeats > 1:
                print(f"    rep {rep} (seed {c['train']['seed']}): "
                      f"Te {runs[-1]['synthetic_holdout']['Te_median_abs_err_eV']:.3f}  "
                      f"recovered {n_rec} ({n_rec/max(len(fail_keys),1):.1%})", flush=True)

        if args.repeats > 1:
            recs = [r["recovered"] for r in runs]
            tes = [r["synthetic_holdout"]["Te_median_abs_err_eV"] for r in runs]
            rmp = [r["ramp_consistency"]["Ne_dex_median_abs_difference"] for r in runs]
            print(f"  spread over {args.repeats} seeds: recovered {min(recs)}-{max(recs)} "
                  f"(sd {np.std(recs):.0f})  Te sd {np.std(tes):.4f}  ramp sd {np.std(rmp):.5f}")

        # Headline row is the LAST repeat; the full set is kept under "repeats" when
        # --repeats > 1, and the summary table below reports mean +- sd across repeats
        # rather than this single row -- quoting one arbitrary seed as "the" result is how
        # the 0.25-vs-0.5 comparison looked decisive when it is not.
        last = runs[-1]
        entry = {
            "synthetic_holdout": last["synthetic_holdout"],
            "agreement_with_classical": last["agreement_with_classical"],
            "ramp_consistency": last["ramp_consistency"],
            "recovered_from_classical_failures": int(n_rec),
            "recovery_fraction_of_failed": float(n_rec / max(len(fail_keys), 1)),
            "final_val_loss": (float(hist["val"][-1]) if hist.get("val") else None),
            "n_repeats": args.repeats,
        }
        if args.repeats > 1:
            entry["repeats"] = runs
            entry["recovered_sd"] = float(np.std([r["recovered"] for r in runs]))
        results[str(w)] = entry

        sh, ag, rc = (entry["synthetic_holdout"], entry["agreement_with_classical"],
                      entry["ramp_consistency"])
        print(f"  synthetic : Ne {sh['Ne_dex_median_abs_err']:.3f} dex   "
              f"Te {sh['Te_median_abs_err_eV']:.3f} eV   Vp {sh['Vp_median_abs_err_V']:.3f} V")
        print(f"  real      : classical offset Ne {ag['Ne_dex_median_diff']:+.4f} dex  "
              f"Te {ag['Te_median_diff_eV']:+.4f} eV")
        print(f"              ramp |dNe| {rc['Ne_dex_median_abs_difference']:.4f} dex  "
              f"r {rc['correlation_log10_Ne']:.4f}")
        print(f"              recovered {n_rec} of {len(fail_keys)} "
              f"({n_rec/max(len(fail_keys),1):.1%})")

    path = out(cfg, "tables", "physics_ablation.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    def ms(w, getter):
        """mean and sd of a metric across this weight's repeats."""
        rs = results[w].get("repeats") or [results[w]]
        vals = [getter(r) for r in rs]
        return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    print("\n" + "=" * 78)
    print(f"{'w_phys':>7} | {'synth Te (eV)':>17} | {'ramp |dNe| (dex)':>19} | {'recovery':>17}")
    print("-" * 78)
    for w in results:
        te_m, te_s = ms(w, lambda r: r["synthetic_holdout"]["Te_median_abs_err_eV"])
        rp_m, rp_s = ms(w, lambda r: r["ramp_consistency"]["Ne_dex_median_abs_difference"])
        rc_m, rc_s = ms(w, lambda r: r["recovered"] if "recovered" in r
                        else r["recovered_from_classical_failures"])
        n_fail = max(len(fail_keys), 1)
        print(f"{w:>7} | {te_m:>8.3f} +- {te_s:<6.3f} | {rp_m:>9.4f} +- {rp_s:<6.4f} | "
              f"{rc_m/n_fail:>7.1%} +- {rc_s/n_fail:<6.1%}")
    print("=" * 78)

    if args.repeats > 1:
        ws = list(results)
        if len(ws) == 2:
            a_m, a_s = ms(ws[0], lambda r: r.get("recovered",
                          r.get("recovered_from_classical_failures")))
            b_m, b_s = ms(ws[1], lambda r: r.get("recovered",
                          r.get("recovered_from_classical_failures")))
            se = float(np.sqrt(a_s ** 2 / args.repeats + b_s ** 2 / args.repeats))
            d = a_m - b_m
            print(f"\nrecovery difference {d:+.0f} sweeps, standard error {se:.0f} "
                  f"-> {abs(d)/se if se else float('inf'):.1f} sigma")
            print(f"smallest difference resolvable at {args.repeats} seeds: "
                  f"~{2*se:.0f} sweeps ({200*se/max(len(fail_keys),1):.1f} points)")
            if abs(d) < 2 * se:
                print("=> these two weights are NOT distinguishable on this evidence; "
                      "keep the validated default.")

    print(f"\nwrote {path}")
    print("\nRead it this way: the physics term is meant to trade synthetic accuracy for")
    print("real-data robustness, so a gain in the synthetic column is NOT evidence for a")
    print("lower weight. Only the real columns -- ramp consistency and recovery -- can")
    print("settle it, and only by more than the resolution floor printed above.")


if __name__ == "__main__":
    main()
