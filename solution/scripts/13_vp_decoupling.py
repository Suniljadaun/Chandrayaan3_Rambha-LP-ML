#!/usr/bin/env python3
"""
Step 13 — Break the temperature / plasma-potential degeneracy.

THE PROBLEM
-----------
Te is the weakest of the three recovered parameters: 0.078 eV on synthetic hold-out, which is
~20% at the 0.4 eV this archive actually sits at, and 0.101 eV repeatability between the two
halves of one commanded sweep.

Two diagnostics locate the cause.

1. The error is FLAT in absolute terms across the whole training prior. Measured on the trained
   three-parameter network, median |Te error| by true Te:

       0.05-0.15 eV -> 0.053 eV (53%)      0.50-0.80 eV -> 0.067 eV (10%)
       0.15-0.30 eV -> 0.092 eV (40%)      0.80-1.20 eV -> 0.077 eV  (8%)
       0.30-0.50 eV -> 0.077 eV (20%)      1.20-2.00 eV -> 0.084 eV  (5%)

   Ratio of the low-Te bins to the high-Te bins: 1.08. The network delivers one flat absolute
   number regardless of the answer, because it minimises squared error in eV over a uniform
   prior. Note also that 41% of the training set sits above 1.2 eV, a regime this archive never
   occupies, while only 10% lands in the 0.3-0.5 eV band that it does.

2. Fixing that parameterisation is NOT enough. Narrowing the prior to 0.20-0.80 eV and training
   on log10(Te), three seeds each, measured inside the 0.30-0.50 eV band:

       current  (0.05-2.0, linear) : 19.9  22.3  24.6  -> mean 22.3%, sd 2.4%
       tight    (0.20-0.80, log)   : 20.4  19.6  20.5  -> mean 20.2%, sd 0.5%

   2.1 points, 1.5 sigma. Not significant. The spread collapses (2.4% -> 0.5%), which is worth
   having for reproducibility, but the accuracy does not move.

So the limit is structural, not a parameterisation artefact. A Fisher-information analysis of
this measurement gives corr(Te, Vp) = +0.93: the network cannot separate a slightly hotter
plasma from a slightly shifted plasma potential. That single correlation inflates the achievable
Te error by 3.3x over the Te-only bound.

THE FIX BEING TESTED
--------------------
Remove the two quantities Te is degenerate with before the network sees the curve:

    u = V - Vp_estimate                     shift out the plasma potential
    y = I(u) / I(u = 0)                     divide out the density
    input = log(y) on a fixed u grid        the slope of this IS 1/Te

Three configurations, three seeds each, all scored inside the 0.30-0.50 eV band:

    A  baseline    Te straight from the raw 240-point curve (the current approach)
    B  ceiling     aligned on the TRUE Vp -- the best the idea could possibly do
    C  realistic   aligned on a PREDICTED Vp -- what is actually available on real sweeps

B measures whether the 3.3x headroom is real. C measures how much survives once the Vp estimate
carries its own error. If C lands near B the transformation is worth building into the pipeline;
if C collapses back to A, the Vp error eats the gain and the idea should be dropped rather than
shipped.

Needs torch. Prints a table; changes nothing in the pipeline.
"""
import _bootstrap as B_
import numpy as np
import torch
import torch.nn as nn
from rambhalp.config import load_config
from rambhalp import synthetic
from rambhalp.preprocess import common_grid

REAL = (0.30, 0.50)
UG = np.linspace(-2.0, 0.6, 120)


def align(X, V, Vp):
    out = np.empty((X.shape[0], UG.size), dtype=np.float32)
    for i in range(X.shape[0]):
        u = V - Vp[i]
        y = np.interp(UG, u, X[i], left=np.nan, right=np.nan)
        ref = np.interp(0.0, u, X[i])
        if not np.isfinite(ref) or ref <= 0:
            ref = max(np.nanmax(y), 1e-12)
        y = np.where(np.isfinite(y), y, 0.0) / ref
        out[i] = np.log(np.clip(y, 1e-6, None))
    return out


def _net(d):
    return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(128, 1))


def fit(Xtr, ttr, Xv, seed, epochs=60, lr=3e-3):
    sc = (np.ptp(ttr) / 2) or 1.0
    torch.manual_seed(seed); np.random.seed(seed)
    m = _net(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.tensor(Xtr), torch.tensor(ttr[:, None].astype(np.float32))),
        batch_size=256, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); (((m(xb) - yb) / sc) ** 2).mean().backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xv)).numpy()[:, 0]


def band(pred, true):
    k = (true >= REAL[0]) & (true < REAL[1])
    return 100 * float(np.median(np.abs(pred[k] - true[k]) / true[k]))


def main():
    cfg = load_config(); cfg["synthetic"]["two_population"] = False
    V = np.asarray(common_grid(cfg), float)
    print("=" * 74)
    print("STEP 13: does decoupling Te from Vp improve the temperature?")
    print("=" * 74)
    rows, vperr = {}, []
    for seed in (0, 1, 2):
        Xtr, Ytr, _ = synthetic.make_dataset(10000, cfg, seed=20 + seed)
        Xv, Yv, _ = synthetic.make_dataset(2500, cfg, seed=700 + seed)
        Te_tr, Te_v = Ytr[:, 1], Yv[:, 1]
        Vp_tr, Vp_v = Ytr[:, 2], Yv[:, 2]

        a = fit(Xtr, Te_tr, Xv, seed)
        b = fit(align(Xtr, V, Vp_tr), Te_tr, align(Xv, V, Vp_v), seed)
        vh_v = fit(Xtr, Vp_tr, Xv, seed)
        vh_tr = fit(Xtr, Vp_tr, Xtr, seed)
        c = fit(align(Xtr, V, vh_tr), Te_tr, align(Xv, V, vh_v), seed)
        vperr.append(float(np.median(np.abs(vh_v - Vp_v))))

        for tag, p in (("A  baseline (raw curve)", a),
                       ("B  aligned on TRUE Vp (ceiling)", b),
                       ("C  aligned on PREDICTED Vp", c)):
            rows.setdefault(tag, []).append(band(p, Te_v))
        print(f"  seed {seed} done   Vp error {vperr[-1]:.3f} V", flush=True)

    print(f"\n  Te error inside the {REAL[0]}-{REAL[1]} eV band, where this archive sits\n")
    for tag, v in rows.items():
        print("  %-34s %s   mean %.1f%%  sd %.1f%%"
              % (tag, " ".join("%.1f%%" % x for x in v), np.mean(v), np.std(v, ddof=1)))
    a_m = np.mean(rows["A  baseline (raw curve)"])
    b_m = np.mean(rows["B  aligned on TRUE Vp (ceiling)"])
    c_m = np.mean(rows["C  aligned on PREDICTED Vp"])
    print(f"\n  median Vp error across seeds: {np.median(vperr):.3f} V")
    print("\n  HOW TO READ THIS")
    print("  B well below A  -> the degeneracy really is the limit; the headroom is real.")
    print("  C close to B    -> the transformation survives a realistic Vp estimate. ADOPT.")
    print("  C close to A    -> Vp error eats the gain. Do not ship it; report the ceiling")
    print("                     as evidence that a better Vp estimate is the way in.")
    print(f"\n  A = {a_m:.1f}%   B = {b_m:.1f}%   C = {c_m:.1f}%")


if __name__ == "__main__":
    main()
