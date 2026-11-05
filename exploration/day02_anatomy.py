"""Anatomy of one real RAMBHA-LP sweep: the three regions of the I-V curve.

Uses the project's own reading and preprocessing (rising half of the first commanded sweep,
26 Aug 2023), the classical fit for the floating potential and the electron-retardation
slope, and the trained network for the plasma potential. Run from the repository root:

    python exploration/day02_anatomy.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution" / "src"))
from rambhalp.config import load_config            # noqa: E402
from rambhalp import io_pds, preprocess, classical  # noqa: E402

TIMESTAMP, IDX = "20230826T145357736", 0

cfg = load_config()
files = io_pds.discover_files(cfg["data"]["_archive_abs"], level="rawB", max_files=None)
sweep = [s for s in preprocess.segment_file(next(f for f in files if f.timestamp == TIMESTAMP), cfg)
         if s.idx == IDX][0]
fit = classical.fit_sweep(sweep, cfg)
ml = pd.read_csv(ROOT / "solution" / "outputs" / "tables" / "ml_predictions.csv")
net = ml[(ml.timestamp.astype(str) == TIMESTAMP) & (ml.idx == IDX)].iloc[0]

V, I = np.asarray(sweep.V, float), np.asarray(sweep.I, float)
vf, vp, te = fit.V_float, float(net.Vp_V), fit.Te_eV
baseline = float(np.median(I[V <= vf - cfg["classical"]["ion_floor_offset"]]))
Ie = I - baseline

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.45, 1]))
regions = [(V.min(), vf, "#dbe8f5", "Ion\nsaturation"),
           (vf, vp, "#fbe3cf", "Electron\nretardation\n(gives Te)"),
           (vp, V.max(), "#dcefdc", "Electron saturation\n(gives Ne)")]
for ax in (a, b):
    for lo, hi, colour, _ in regions:
        ax.axvspan(lo, hi, color=colour, lw=0)
    ax.axvline(vf, color="#1f4e79", ls="--", lw=1.2)
    ax.axvline(vp, color="#2e7d32", ls="--", lw=1.2)
    ax.grid(alpha=0.3)

a.plot(V, I * 1e9, color="black", lw=2)
a.axhline(0, color="k", lw=0.6)
top = I.max() * 1e9 * 1.18
a.set_ylim(I.min() * 1e9 - 12, top)
for lo, hi, _, label in (regions[0], regions[2]):
    a.text((lo + hi) / 2, top * 0.97, label, ha="center", va="top", fontsize=9.5)
a.annotate(regions[1][3].replace("\n", " ", 1), xy=((vf + vp) / 2, 150), xytext=(-11.6, 150),
           fontsize=9.5, va="center", arrowprops=dict(arrowstyle="->", color="#9a4a12"))
a.annotate(f"Floating potential\n$V_f$ = {vf:.2f} V", xy=(vf, 0), xytext=(-11.5, 60),
           fontsize=9.5, color="#1f4e79", arrowprops=dict(arrowstyle="->", color="#1f4e79"))
a.annotate(f"Plasma potential\n$V_p$ = {vp:.2f} V (network)", xy=(vp, np.interp(vp, V, I) * 1e9),
           xytext=(1.0, 40), fontsize=9.5, color="#2e7d32",
           arrowprops=dict(arrowstyle="->", color="#2e7d32"))
a.set_xlabel("Bias voltage (V)")
a.set_ylabel("Probe current (nA)")
a.set_title("(a) The measured curve", loc="left", fontweight="bold")

pos = Ie > 0
b.semilogy(V[pos], Ie[pos] * 1e9, color="black", lw=2, label="Electron current")
win = (V >= vf) & (V <= min(vp, vf + cfg["classical"]["retardation_span"])) & pos
p = np.polyfit(V[win], np.log(Ie[win]), 1)
vv = np.linspace(vf - 0.6, vp + 0.6, 50)
b.semilogy(vv, np.exp(np.polyval(p, vv)) * 1e9, color="#c2410c", lw=1.6, ls="--",
           label=f"Straight-line fit: slope = 1/Te,\nTe = {1 / p[0]:.2f} eV (classical {te:.2f} eV)")
b.set_xlim(vf - 3, vp + 4)
b.set_ylim(Ie[pos].min() * 1e9 * 0.8, Ie.max() * 1e9 * 2)
b.set_xlabel("Bias voltage (V)")
b.set_ylabel("Electron current (nA, log scale)")
b.set_title("(b) Log scale: the rise is exponential", loc="left", fontweight="bold")
b.legend(loc="lower right", fontsize=9, framealpha=0.95)

fig.suptitle("Anatomy of a real RAMBHA-LP sweep, 26 August 2023", fontsize=12.5)
fig.tight_layout()
out = Path(__file__).with_name("day02_anatomy.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out.name} | Vf={vf:.2f} V | Vp={vp:.2f} V (network) | Te={1 / p[0]:.2f} eV")
