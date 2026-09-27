"""
model.py — the inversion network that maps an I-V curve to plasma parameters.

Architecture: a straightforward MLP. The input is the fixed-length scaled current vector
(grid_points long); the output is 3 numbers [log10(Ne), Te, Vp]. Dropout is kept ACTIVE at
inference time on purpose — running many stochastic forward passes (MC-dropout) gives us an
uncertainty estimate, which is essential because we have no ground truth to calibrate against.

Why an MLP and not a CNN: the curve is short (~240 points), smooth, and already on a common
grid, so a fully-connected net trains fast on CPU and is trivial to defend. The design note in
GUIDE.md §6 explains the trade-off (CNN over V would add translation-equivariance we do not
need because Vp is a target, not a nuisance).
"""
from __future__ import annotations
import numpy as np


def build_model(cfg: dict):
    """Construct the torch MLP from config. Imported lazily so non-torch steps stay light."""
    import torch
    import torch.nn as nn

    grid_points = cfg["preprocess"]["grid_points"]
    hidden = cfg["model"]["hidden"]
    p_drop = cfg["model"]["dropout"]

    layers = []
    prev = grid_points
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p_drop)]
        prev = h
    layers += [nn.Linear(prev, n_targets(cfg))]
    return nn.Sequential(*layers)


def n_targets(cfg: dict) -> int:
    """3 for [log10 Ne, Te, Vp]; 5 for [log10 Nc, Tc, log10 Nh, Th, Vp]."""
    return 5 if cfg["synthetic"].get("two_population", False) else 3


def predict(model, X, cfg, mc_samples: int = 1):
    """Predict parameters for a batch of curves X (numpy, (N, G)).

    If mc_samples > 1, keep dropout ON and average `mc_samples` stochastic passes; the
    per-sample std is returned as an uncertainty. Returns dict of numpy arrays:
       Ne_cc, Te_eV, Vp_V  (means) and Ne_cc_std, Te_eV_std, Vp_V_std (uncertainties).
    """
    import torch
    device = next(model.parameters()).device
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)

    if mc_samples > 1:
        model.train()          # dropout active
    else:
        model.eval()

    preds = []
    with torch.no_grad():
        for _ in range(max(1, mc_samples)):
            preds.append(model(Xt).cpu().numpy())
    preds = np.stack(preds, axis=0)               # (S, N, 3) RAW network outputs

    # Decode EACH stochastic sample through the SAME transform train.py's
    # _params_from_output applies before computing the loss (clamp -> softplus+1e-3 for
    # Te, clamp for logNe/Vp), THEN aggregate mean/std. This matters for two reasons:
    #  1. Correctness: the network was never trained to output Te_eV directly on channel
    #     1 -- it was trained so that softplus(channel_1)+1e-3 equals Te. Using the raw
    #     channel as Te_eV (as this function used to) fed wrong-scale, sometimes-negative
    #     values into every downstream physics reconstruction (sqrt(negative) -> NaN, the
    #     RuntimeWarning from electron_sat_current) and into every reported accuracy
    #     number that calls predict(), including synthetic_holdout_error.
    #  2. MC-dropout uncertainty must propagate through the same nonlinearity the model
    #     actually predicts through -- transforming per-sample-then-averaging is the
    #     statistically correct order; averaging raw outputs then transforming is not.
    sp = lambda z: np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)   # stable softplus

    logNe = np.clip(preds[:, :, 0], -2.0, 6.0)
    te_eV = sp(preds[:, :, 1]) + 1e-3
    out = {}
    if preds.shape[2] == 5:
        from .synthetic import _HOT_FLOOR
        logNh = np.clip(preds[:, :, 2], -2.0, 6.0)
        Nh = np.maximum(10 ** logNh - _HOT_FLOOR, 0.0)
        # hot temperature is parameterised as an INCREMENT above the cold one, matching the
        # decoder in train.py and the label ordering in synthetic.py
        th_eV = te_eV * 1.5 + 0.05 + sp(preds[:, :, 3])
        vp = np.clip(preds[:, :, 4], -30.0, 30.0)
        out.update({"Nh_cc": Nh.mean(axis=0), "Th_eV": th_eV.mean(axis=0),
                    "Nh_cc_std": Nh.std(axis=0), "Th_eV_std": th_eV.std(axis=0)})
    else:
        vp = np.clip(preds[:, :, 2], -30.0, 30.0)

    out.update({
        "Ne_cc": 10 ** logNe.mean(axis=0),      # geometric mean in log space, as before
        "Te_eV": te_eV.mean(axis=0),
        "Vp_V": vp.mean(axis=0),
        # convert log-density std into a multiplicative (dex) uncertainty on Ne
        "Ne_dex_std": logNe.std(axis=0),
        "Te_eV_std": te_eV.std(axis=0),
        "Vp_V_std": vp.std(axis=0),
    })
    return out
