#!/usr/bin/env python3
"""
Step 5 — Apply the trained model to the REAL sweeps and count recoveries.

A sweep the instrument clipped is never counted as recovered (config.yaml, recovery).

Prefers the MAVEN-transfer checkpoint if it exists, else the plain PINN.

Outputs:
  outputs/tables/ml_predictions.csv     model params + uncertainty + recon R^2 per sweep
  outputs/tables/recovery_summary.json  the headline recovery numbers
"""
import _bootstrap as B
import json
import numpy as np
import pandas as pd
from rambhalp.config import load_config, out
from rambhalp import train, infer


def main():
    cfg = load_config()
    print("=" * 70)
    print("STEP 5: apply model to real sweeps, count recoveries")
    print("=" * 70)

    ckpt_transfer = out(cfg, "checkpoints", "maven_transfer.pt")
    ckpt_pinn = out(cfg, "checkpoints", "pinn.pt")
    # Only prefer the MAVEN-transfer checkpoint if THIS run actually trained one. File
    # existence alone is not enough: a stale maven_transfer.pt left on disk from an earlier
    # full run must not be silently picked up by a --no-maven run, which never regenerates
    # it. cfg["maven"]["enabled"] reflects whether step 04 ran in *this* invocation
    # (run_all.py sets it False under --no-maven).
    ckpt = ckpt_transfer if (cfg["maven"]["enabled"] and ckpt_transfer.exists()) else ckpt_pinn
    print(f"using checkpoint: {ckpt.name}")
    model = train.load_checkpoint(cfg, ckpt)

    sweeps = B.load_pickle(B.cache_dir(cfg) / "sweeps.pkl")
    classical_df = pd.read_csv(out(cfg, "tables", "classical_baseline.csv"))

    # Fixed seed so the Monte-Carlo dropout draws, and every error bar, are reproducible.
    import torch
    torch.manual_seed(cfg["train"]["seed"])
    ml_rows = infer.apply(model, sweeps, cfg)
    ml_df = pd.DataFrame(ml_rows)
    ml_df.to_csv(out(cfg, "tables", "ml_predictions.csv"), index=False)

    # Match ML rows to classical status by (timestamp, idx) to count recoveries of FAILURES.
    fail_keys = set(zip(classical_df.loc[classical_df["failed"], "timestamp"],
                        classical_df.loc[classical_df["failed"], "idx"]))
    n_total = len(classical_df)
    n_classical_ok = int((~classical_df["failed"]).sum())
    recovered = [r for r in ml_rows
                 if (r["timestamp"], r["idx"]) in fail_keys and r["recovered"]]
    n_recovered = len(recovered)

    # Clipped sweeps are unusable for BOTH methods: the classical fit's saturation branch on them
    # is almost entirely the clipped plateau, just as the network's input is. So a classical
    # pass on a clipped sweep is not counted as usable either.
    ok_keys = set(zip(classical_df.loc[~classical_df["failed"], "timestamp"],
                      classical_df.loc[~classical_df["failed"], "idx"]))
    n_classical_ok_unclipped = sum(1 for r in ml_rows
                                   if (r["timestamp"], r["idx"]) in ok_keys and not r["clipped"])
    n_usable = n_classical_ok_unclipped + n_recovered
    summary = {
        "n_total_sweeps": n_total,
        "n_classical_ok": n_classical_ok,
        "n_classical_ok_unclipped": n_classical_ok_unclipped,
        "n_unclipped_total": int((~ml_df["clipped"]).sum()),
        "n_classical_failed": n_total - n_classical_ok,
        "n_ml_recovered_from_failed": n_recovered,
        "recovery_fraction_of_failed":
            n_recovered / max(1, n_total - n_classical_ok),
        "combined_yield": n_usable,
        "combined_yield_fraction": n_usable / max(1, n_total),
        # Clipped sweeps (config.yaml, recovery) are never counted as recovered.
        "n_clipped_total": int(ml_df["clipped"].sum()),
        "n_classical_failed_clipped": int(sum(1 for r in ml_rows
                                          if (r["timestamp"], r["idx"]) in fail_keys and r["clipped"])),
        "n_classical_ok_clipped": int(sum(1 for r in ml_rows
                                      if (r["timestamp"], r["idx"]) not in fail_keys and r["clipped"])),
        "checkpoint": ckpt.name,
    }
    with open(out(cfg, "tables", "recovery_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 70)
    for k, v in summary.items():
        print(f"{k:32s}: {v}")
    print(f"\nwrote {out(cfg,'tables','ml_predictions.csv')} and recovery_summary.json")


if __name__ == "__main__":
    main()
