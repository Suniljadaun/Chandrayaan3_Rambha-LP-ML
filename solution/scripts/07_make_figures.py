#!/usr/bin/env python3
"""
Step 7 — Render every report figure from the saved tables/checkpoints.

Outputs: outputs/figures/F1..F6 .png
"""
import _bootstrap as B
import json
import pandas as pd
from rambhalp.config import load_config, out
from rambhalp import train, figures, classical as classical_mod


def main():
    cfg = load_config()
    print("=" * 70)
    print("STEP 7: figures")
    print("=" * 70)

    sweeps = B.load_pickle(B.cache_dir(cfg) / "sweeps.pkl")
    cdf = pd.read_csv(out(cfg, "tables", "classical_baseline.csv"))
    classical_rows = classical_mod.fit_all(sweeps, cfg)  # cheap; keeps sweep<->row aligned

    figures.fig_example_sweep(sweeps, classical_rows, cfg)
    figures.fig_failure_map(cdf, cfg)

    with open(out(cfg, "tables", "train_history.json")) as f:
        hist = json.load(f)
    figures.fig_training_curves(hist, cfg)

    ckpt = out(cfg, "checkpoints", "maven_transfer.pt")
    # Same fix as 05_apply_and_recover.py / 06_validate_hop.py: existence alone isn't enough,
    # a stale maven_transfer.pt from an earlier full run must not be picked up by a
    # --no-maven run, or F4's scatter plot silently shows the wrong model's predictions.
    ckpt = ckpt if (cfg["maven"]["enabled"] and ckpt.exists()) else out(cfg, "checkpoints", "pinn.pt")
    model = train.load_checkpoint(cfg, ckpt)
    figures.fig_synthetic_scatter(model, cfg)

    with open(out(cfg, "tables", "recovery_summary.json")) as f:
        rec = json.load(f)
    figures.fig_recovery(rec["n_total_sweeps"], rec["n_classical_ok"],
                         rec["n_ml_recovered_from_failed"], cfg)

    ml_df = pd.read_csv(out(cfg, "tables", "ml_predictions.csv"))
    figures.fig_density_trend(ml_df, cfg)
    print("all figures written to", cfg["paths"]["_figures_abs"])


if __name__ == "__main__":
    main()
