"""
infer.py — apply the trained network to REAL RAMBHA sweeps and count recoveries.

Two jobs:
  1. prepare_input(): turn a real Sweep into the exact fixed-length, fixed-scale vector the
     network was trained on. Real sweeps can arrive with either current sign convention, so we
     ORIENT each curve to the canonical 'electron current increases with V' form (the same form
     the synthetic generator used). Amplitude is preserved (scaled by the fixed I_SCALE), so
     density stays recoverable.
  2. apply(): run the model (with MC-dropout uncertainty), reconstruct each sweep through the
     OML forward model, score the reconstruction, and decide which sweeps are RECOVERED.

'Recovered' is defined conservatively (config-driven): plausible Ne and Te, a good physics
reconstruction, and bounded uncertainty. A sweep the classical method failed on, that the
network fits self-consistently with tight error bars, is a genuine recovery — and we only ever
report the number the reconstruction score actually supports.
"""
from __future__ import annotations
import numpy as np

from . import physics
from .model import predict
from .preprocess import resample_to_grid, common_grid


def prepare_input(sweeps, cfg):
    """Return (X, electron_curves) for a list of Sweeps, in the canonical form the network
    was trained on: oriented so electron current rises with V, and with the ion/offset
    baseline subtracted so the curve starts at ~0.

    X               : (N, G) network input (scaled electron current on the common grid)
    electron_curves : (N, G) the baseline-subtracted electron current in amps (for scoring)
    """
    grid = common_grid(cfg)
    X = np.empty((len(sweeps), grid.size), dtype=np.float32)
    elec = np.empty((len(sweeps), grid.size), dtype=np.float32)
    for i, sw in enumerate(sweeps):
        I = resample_to_grid(sw, cfg)               # amps on common grid
        # Orient: electron current should rise with V. If it falls, flip sign.
        if np.corrcoef(grid, I)[0, 1] < 0:
            I = -I
        # Subtract the ion/offset baseline (median of the most-negative-bias samples).
        n_edge = max(3, grid.size // 10)
        baseline = np.median(np.sort(I)[:n_edge])
        Ie = I - baseline
        elec[i] = Ie
        X[i] = physics.to_input_np(Ie).astype(np.float32)
    return X, elec


def _recon_r2(curve_amps, Ne, Te, Vp, cfg, Nh=None, Th=None):
    """R^2 of the electron reconstruction against an observed (baseline-subtracted) curve.

    When the model predicts a second population, both are reconstructed — scoring a
    two-population prediction against a one-population model would penalise the network for
    the very structure it was built to resolve.
    """
    grid = common_grid(cfg)
    if Nh is not None and Th is not None:
        recon = physics.forward_iv_two_np(grid, Ne, Te, Nh, Th, Vp, cfg)
    else:
        recon = physics.forward_iv_np(grid, Ne, Te, Vp, cfg)
    ss_res = np.sum((curve_amps - recon) ** 2)
    ss_tot = np.sum((curve_amps - curve_amps.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def apply(model, sweeps, cfg, mc_samples=None):
    """Predict parameters for every sweep and score reconstruction + recovery.

    Returns a list of dicts (one per sweep) with predictions, uncertainty, reconstruction R^2,
    and a boolean 'recovered'.
    """
    mc = mc_samples or cfg["validate"]["mc_samples"]
    X, elec = prepare_input(sweeps, cfg)
    pred = predict(model, X, cfg, mc_samples=mc)

    ccfg = cfg["classical"]
    lo_te, hi_te = ccfg["te_plausible_eV"]
    lo_ne, hi_ne = ccfg["ne_plausible_cc"]

    two_pop = "Nh_cc" in pred
    rows = []
    for i, sw in enumerate(sweeps):
        Ne, Te, Vp = pred["Ne_cc"][i], pred["Te_eV"][i], pred["Vp_V"][i]
        Nh = pred["Nh_cc"][i] if two_pop else None
        Th = pred["Th_eV"][i] if two_pop else None
        r2 = _recon_r2(elec[i], Ne, Te, Vp, cfg, Nh=Nh, Th=Th)
        recovered = bool(
            (lo_ne <= Ne <= hi_ne) and (lo_te <= Te <= hi_te)
            and (r2 >= ccfg["min_r2"]) and (pred["Ne_dex_std"][i] < 0.5)
        )
        rows.append(dict(
            date=sw.date, timestamp=sw.timestamp, idx=sw.idx, t_start=sw.t_start,
            direction=getattr(sw, "direction", ""), segment=getattr(sw, "segment", -1),
            Ne_cc=float(Ne), Te_eV=float(Te), Vp_V=float(Vp),
            Ne_dex_std=float(pred["Ne_dex_std"][i]),
            Te_eV_std=float(pred["Te_eV_std"][i]),
            Vp_V_std=float(pred["Vp_V_std"][i]),
            recon_r2=float(r2), recovered=recovered,
            **({"Nh_cc": float(Nh), "Th_eV": float(Th),
                "Nh_cc_std": float(pred["Nh_cc_std"][i]),
                "Th_eV_std": float(pred["Th_eV_std"][i])} if two_pop else {}),
        ))
    return rows
