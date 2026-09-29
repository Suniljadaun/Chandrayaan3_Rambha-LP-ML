"""
te_refine.py — a second-stage head that recovers the electron temperature from the
plasma-potential-aligned, density-normalised sweep.

WHY
---
Te was by a wide margin the weakest of the three recovered parameters: 0.078 eV on synthetic
hold-out (~20% at the 0.4 eV this archive occupies) against 0.021 dex for density and 0.14 V for
the plasma potential.

The cause is a degeneracy, not noise. A Fisher-information analysis of this measurement gives
corr(Te, Vp) = +0.93 — the network cannot separate a slightly hotter plasma from a slightly
shifted plasma potential — and that single correlation inflates the achievable Te error by 3.3x
over the Te-only bound. Two supporting diagnostics:

  * the error is FLAT in absolute terms across the training prior (0.053 eV at Te = 0.1 eV,
    0.084 eV at Te = 1.6 eV; ratio of low to high bins 1.08), i.e. the network returns one number
    regardless of the answer;
  * fixing that by narrowing the prior and training on log10(Te) moves the error by 2.1 points
    at 1.5 sigma — not significant. The limit is structural.

THE TRANSFORMATION
------------------
Remove both quantities Te is degenerate with before the network sees the curve:

    u = V - Vp_hat                 shift the plasma potential to the origin
    y = I(u) / I(u = 0)            divide out the density
    input = log(y) on a fixed u grid

After this the slope of the input IS 1/Te, independent of density and of where Vp sits. The
network is then reading a quantity that no longer competes with anything else.

MEASURED RESULT (scripts/13_vp_decoupling.py, 20000 training sweeps, 90 epochs, 3 seeds,
scored inside the 0.30-0.50 eV band where this archive actually sits):

    baseline, raw curve          24.2  21.2  23.0  -> 22.8% +- 1.5%
    aligned on a PREDICTED Vp    12.9  11.1  11.5  -> 11.8% +- 0.9%

    11.0 points, 10.7 sigma, a factor of 1.93. In absolute terms 0.091 -> 0.047 eV at 0.4 eV.

An oracle run aligned on the TRUE Vp reaches 11.3%, so with the full-budget Vp estimate
(0.127-0.149 V here, matching the pipeline) the alignment error costs essentially nothing: the
realistic configuration has already reached the ceiling.

DESIGN
------
This is a REFINEMENT HEAD, not a rewrite. Stage one is the existing, validated three-parameter
network, unchanged — it supplies Vp. Stage two is a small separate network that sees only the
aligned curve and returns Te. Nothing about the density or plasma-potential path is touched, so
enabling this cannot alter those results.

IMPORTANT CAVEAT, and it is the reason this defaults to off. Every number above is SYNTHETIC
hold-out. This project has already been caught once by that distinction: the physics-loss
ablation showed a term that looks harmful on synthetic data while buying ten percentage points of
real-archive recovery. A synthetic gain is not a real-data gain until the real-data checks say so
— agreement with the classical method, ramp-pair consistency, and recovery fraction. Run
scripts/14_te_refine_eval.py before turning this on.
"""
from __future__ import annotations

import numpy as np

# Bias grid relative to Vp. Reaches deep enough to see the retarding slope and just past Vp so
# the normalisation point is interior rather than at an edge.
U_LO, U_HI, U_N = -2.0, 0.6, 120
U_GRID = np.linspace(U_LO, U_HI, U_N)


def set_window(lo, hi, n=None):
    """Move the alignment window. Exists because the default spans the bias region where this
    archive is independently known to depart from the forward model (Phase 15b: a positive
    excess over a matched single-Maxwellian control at u = -0.3 and -0.6 V on nine of nine
    observation days, p = 0.002; Phase 16b: that excess reverses sign between probe channels at
    11 sigma, marking it instrumental). A head that reads Te from exactly that window is
    maximally exposed to a known sim-to-real defect, which is the leading explanation for why
    the refinement improves synthetic accuracy and worsens real ramp repeatability."""
    global U_LO, U_HI, U_N, U_GRID
    U_LO, U_HI = float(lo), float(hi)
    U_N = int(n or U_N)
    U_GRID = np.linspace(U_LO, U_HI, U_N)
    return U_GRID


def align_curves(X, V, Vp):
    """Shift each curve onto u = V - Vp, normalise by the current at u = 0, return log.

    X  : (n, G) curves in network input units
    V  : (G,)   the common bias grid
    Vp : (n,)   plasma potential per curve (predicted at inference, true when training an oracle)
    """
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    Vp = np.asarray(Vp, dtype=float)
    out = np.empty((X.shape[0], U_GRID.size), dtype=np.float32)
    for i in range(X.shape[0]):
        u = V - Vp[i]
        y = np.interp(U_GRID, u, X[i], left=np.nan, right=np.nan)
        # Normalise by the current at the plasma potential. Falling back to the curve maximum
        # keeps a degenerate sweep from producing inf/nan rather than merely a poor input.
        ref_at = 0.0 if (U_LO <= 0.0 <= U_HI) else U_HI
        ref = np.interp(ref_at, u, X[i])
        if not np.isfinite(ref) or ref <= 0:
            m = np.nanmax(y) if np.isfinite(y).any() else 0.0
            ref = max(float(m), 1e-12)
        y = np.where(np.isfinite(y), y, 0.0) / ref
        out[i] = np.log(np.clip(y, 1e-6, None))
    return out


def enabled(cfg: dict) -> bool:
    return bool(cfg.get("model", {}).get("te_refine", False))


def build(cfg: dict):
    """Small MLP over the aligned curve. Deliberately smaller than the main network: the input
    is already the quantity of interest, so little representation learning is required."""
    import torch.nn as nn
    p = float(cfg["model"].get("dropout", 0.10))
    return nn.Sequential(
        nn.Linear(U_GRID.size, 256), nn.ReLU(), nn.Dropout(p),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p),
        nn.Linear(128, 1),
    )


def train_head(X, Y, Vp_hat, V, cfg, epochs=None, verbose=True):
    """Train the Te head on curves aligned with the plasma potentials the main model predicts.

    Aligning on the PREDICTED rather than the true Vp during training is deliberate: the head
    then learns on inputs carrying the same alignment error it will meet at inference, instead of
    being trained on a cleaner distribution than it will ever see.
    """
    import torch

    ep = int(epochs or cfg["train"].get("te_refine_epochs", cfg["train"]["epochs"]))
    lr = float(cfg["train"].get("te_refine_lr", 3e-3))
    bs = int(cfg["train"]["batch_size"])
    dev = "cpu"

    A = align_curves(X, V, Vp_hat)
    te = np.asarray(Y, dtype=float)[:, 1].astype(np.float32)
    scale = float(np.ptp(te) / 2.0) or 1.0

    torch.manual_seed(int(cfg["train"]["seed"]))
    model = build(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
    ds = torch.utils.data.TensorDataset(torch.as_tensor(A),
                                        torch.as_tensor(te[:, None]))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)

    hist = []
    for e in range(ep):
        tot = n = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = (((model(xb) - yb) / scale) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * xb.size(0); n += xb.size(0)
        sched.step()
        hist.append(tot / max(n, 1))
        if verbose and (e % 20 == 0 or e == ep - 1):
            print(f"  [te_refine] epoch {e:4d} | loss {hist[-1]:.5f}", flush=True)
    model.eval()
    return model, hist


def predict_te(model, X, V, Vp_hat, mc_samples: int = 1):
    """Te from the aligned curve. mc_samples > 1 keeps dropout on for an MC-dropout spread."""
    import torch
    A = torch.as_tensor(align_curves(X, V, Vp_hat))
    if mc_samples > 1:
        model.train()
        with torch.no_grad():
            s = np.stack([model(A).numpy()[:, 0] for _ in range(mc_samples)])
        model.eval()
        return s.mean(axis=0), s.std(axis=0)
    with torch.no_grad():
        return model(A).numpy()[:, 0], np.zeros(A.shape[0], dtype=np.float32)
