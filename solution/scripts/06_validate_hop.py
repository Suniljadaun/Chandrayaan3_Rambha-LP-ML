#!/usr/bin/env python3
"""
Step 6 — Validate honestly: synthetic hold-out, agreement with classical, density trend,
ramp consistency.

Outputs:
  outputs/tables/validation.json
"""
import _bootstrap as B
import json
import pandas as pd
from rambhalp.config import load_config, out
from rambhalp import train, validate, classical as classical_mod


def main():
    cfg = load_config()
    print("=" * 70)
    print("STEP 6: validation (no ground truth — three independent checks)")
    print("=" * 70)

    ckpt_transfer = out(cfg, "checkpoints", "maven_transfer.pt")
    # See 05_apply_and_recover.py for why existence alone isn't sufficient here.
    ckpt = (ckpt_transfer if (cfg["maven"]["enabled"] and ckpt_transfer.exists())
            else out(cfg, "checkpoints", "pinn.pt"))
    model = train.load_checkpoint(cfg, ckpt)

    ml_df = pd.read_csv(out(cfg, "tables", "ml_predictions.csv"))
    ml_rows = ml_df.to_dict("records")

    # Rebuild classical results objects from the CSV for the agreement check.
    cdf = pd.read_csv(out(cfg, "tables", "classical_baseline.csv"))
    classical_rows = [classical_mod.ClassicalResult(**{
        k: row[k] for k in ["date", "timestamp", "idx", "adc_channel", "t_start",
                            "V_float", "Te_eV", "Ne_cc", "r2", "failed", "fail_reason"]
    }) for _, row in cdf.iterrows()]

    report = {
        "synthetic_holdout": validate.synthetic_holdout_error(model, cfg),
        "agreement_with_classical": validate.agreement_with_classical(ml_rows, classical_rows),
        "density_trend": validate.density_trend(ml_rows, cfg),
        "ramp_consistency": validate.ramp_consistency(ml_rows),
        "checkpoint": ckpt.name,
    }
    with open(out(cfg, "tables", "validation.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nwrote {out(cfg,'tables','validation.json')}")


if __name__ == "__main__":
    main()
