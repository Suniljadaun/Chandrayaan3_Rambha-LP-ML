#!/usr/bin/env python3
"""
Step 15 — Are the MC-dropout error bars honest?

Two checks, one with known truth and one without:

  1. Synthetic hold-out (4,000 sweeps, truth known). For each parameter, z = error / sigma.
     A calibrated sigma puts 68.3% of |z| under 1 and 95.4% under 2. The factor that would
     make it so (median |z| / 0.674) is the amount sigma should be scaled by.
  2. Real archive (no truth). The up and down halves of one commanded sweep measure the same
     plasma seconds apart, so their difference in log10(Ne) should be explained by the two
     sigmas combined. Same coverage test on z = diff / sqrt(s_up^2 + s_down^2).

Outputs:
  outputs/tables/uncertainty_calibration.json
"""
import _bootstrap as B
import json
import numpy as np
import pandas as pd
from rambhalp.config import load_config, out
from rambhalp import train
from rambhalp.model import predict

IDEAL = {"within_1sigma": 0.683, "within_2sigma": 0.954}


def coverage(err, sigma):
    """Coverage of |err| by 1 and 2 sigma, and the scale factor sigma would need."""
    ok = np.isfinite(err) & np.isfinite(sigma) & (sigma > 0)
    z = np.abs(err[ok]) / sigma[ok]
    return {
        "n": int(ok.sum()),
        "within_1sigma": round(float(np.mean(z < 1)), 3),
        "within_2sigma": round(float(np.mean(z < 2)), 3),
        "median_abs_z": round(float(np.median(z)), 3),
        # |z| of a unit normal has median 0.674; sigma x this factor would be calibrated
        "sigma_scale_needed": round(float(np.median(z) / 0.674), 2),
    }


def main():
    cfg = load_config()
    print("=" * 70)
    print("STEP 15: uncertainty calibration (synthetic truth + real up/down pairs)")
    print("=" * 70)
    model = train.load_checkpoint(cfg, out(cfg, "checkpoints", "pinn.pt"))
    mc = cfg["validate"]["mc_samples"]

    import torch
    torch.manual_seed(cfg["train"]["seed"])       # reproducible dropout draws
    val = np.load(B.cache_dir(cfg) / "synth_val.npz")
    X, Y = val["X"], val["Y"]                       # Y = [log10 Ne, Te, Vp]
    p = predict(model, X, cfg, mc_samples=mc)
    synthetic = {
        "Ne_dex": coverage(np.log10(p["Ne_cc"]) - Y[:, 0], p["Ne_dex_std"]),
        "Te_eV": coverage(p["Te_eV"] - Y[:, 1], p["Te_eV_std"]),
        "Vp_V": coverage(p["Vp_V"] - Y[:, 2], p["Vp_V_std"]),
    }

    ml = pd.read_csv(out(cfg, "tables", "ml_predictions.csv"))
    ml = ml[ml["recovered"]]
    up = ml[ml["direction"] == "up"].set_index(["timestamp", "segment"])
    dn = ml[ml["direction"] == "down"].set_index(["timestamp", "segment"])
    pair = up.join(dn, lsuffix="_up", rsuffix="_dn", how="inner")
    diff = np.log10(pair["Ne_cc_up"].to_numpy()) - np.log10(pair["Ne_cc_dn"].to_numpy())
    sig = np.hypot(pair["Ne_dex_std_up"].to_numpy(), pair["Ne_dex_std_dn"].to_numpy())
    real = {"Ne_dex_up_vs_down": coverage(diff, sig)}

    report = {"ideal": IDEAL, "mc_samples": mc, "synthetic_holdout": synthetic,
              "real_ramp_pairs": real}
    path = out(cfg, "tables", "uncertainty_calibration.json")
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
