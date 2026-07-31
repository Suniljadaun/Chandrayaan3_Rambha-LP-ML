import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "/home/sunil-jadaun/Assignments/IS/Chandrayaan-3/ch3_rambha/LTA_RAMBHA-LP_August2024_final/data/raw/20230826_rawB"
SCI = f"{BASE}/ch3_rlp_nrB_sci_20230826T145357736.csv"
OPS = f"{BASE}/ch3_rlp_nrB_ops_20230826T145357736.csv"

sci = pd.read_csv(SCI, names=["V", "I"], skiprows = 1)
sci["V"] = pd.to_numeric(sci["V"], errors="coerce")
sci["I"] = pd.to_numeric(sci["I"], errors="coerce")

sci = sci.dropna().reset_index(drop=True)

ops = pd.read_csv(OPS, skipinitialspace=True)

ops.columns = [c.strip() for c in ops.columns]

n  = ops["Number_of_samples"].astype(int).tolist()

seg = sci.iloc[0:n[0]]

V, I = seg["V"].to_numpy(), seg["I"].to_numpy()

edges = np.arange(np.floor(V.min()), np.ceil(V.max()) + 0.1, 0.1)

idx = np.digitize(V, edges)

Vb  = np.array([V[idx == b].mean() for b in np.unique(idx) if (idx==b).sum() >= 3])
Ib = np.array([I[idx==b].mean() for b in np.unique(idx) if (idx == b).sum() >= 3])

k = np.where(np.diff(np.sign(Ib)) != 0)[0][0]

vf = Vb[k] - Ib[k] * (Vb[k+1] - Vb[k]) /  (Ib[k+1] - Ib[k]) 


pos = Ib > 0 

d2 = np.gradient(np.gradient(np.log(Ib[pos]), Vb[pos]), Vb[pos])

vp_approx = Vb[pos][np.argmin(d2)]

plt.figure(figsize=(8, 5))
plt.plot(Vb, Ib * 1e9, lw=2, color="black")
plt.axhline(0, color="k", lw=0.5)
# WHAT: shade region 1 (ion saturation): from left edge to Vf.
plt.axvspan(Vb.min(), vf, alpha=0.12, color="tab:blue")
# WHAT: shade region 2 (retardation): Vf to approximate knee.
plt.axvspan(vf, vp_approx, alpha=0.12, color="tab:orange")
# WHAT: shade region 3 (electron saturation): knee to right edge.
plt.axvspan(vp_approx, Vb.max(), alpha=0.12, color="tab:green")
# WHAT: landmark lines + labels.
plt.axvline(vf, ls="--", color="tab:blue", lw=1)
plt.axvline(vp_approx, ls="--", color="tab:green", lw=1)
plt.text(vf, plt.ylim()[1]*0.05, f" V_float={vf:.2f} V", fontsize=8, color="tab:blue")
plt.text(vp_approx, plt.ylim()[1]*0.55, f" V_p≈{vp_approx:.2f} V (approx)", fontsize=8, color="tab:green")
plt.text(Vb.min()+0.3, plt.ylim()[1]*0.85, "1: ion\nsaturation", fontsize=9)
plt.text(vf+0.3, plt.ylim()[1]*0.85, "2: retardation\n(→ Te)", fontsize=9)
plt.text(vp_approx+0.6, plt.ylim()[1]*0.85, "3: e⁻ saturation\n(→ Ne, Vp)", fontsize=9)
plt.xlabel("Bias voltage V (V)"); plt.ylabel("Current (nA)")
plt.title("Anatomy of a real RAMBHA-LP I–V sweep (2023-08-26, sweep #0)")
plt.tight_layout()
plt.savefig("day02_anatomy.png", dpi=130)
print(f"saved day02_anatomy.png | Vf={vf:.2f} V | Vp≈{vp_approx:.2f} V")


