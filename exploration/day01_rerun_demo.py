# WHAT: import numpy under its universal alias np.
# WHEN: top of every scientific Python file.
# WHY: all numeric arrays (voltages, currents) live as numpy arrays — vastly
#      faster than Python lists for math on 835k samples.
import numpy as np

# WHAT: import pandas as pd.
# WHEN: whenever reading/writing tabular files (CSV) or working with tables.
# WHY: our science + ops files are CSVs; pandas reads them in one line and
#      handles messy headers gracefully.
import pandas as pd

# WHAT: import matplotlib's plotting interface.
# WHEN: any script that produces figures.
# WHY: pyplot is the standard figure API; every plot in the final paper
#      will come from it.
import matplotlib.pyplot as plt

# WHAT: base path to the RAMBHA rawB session we verified earlier.
# WHEN: defined once at the top so every path derives from it.
# WHY: one place to edit if you ever move the data. CONFIRM THIS PATH EXISTS
#      before running (batch-generation flag from the header).
BASE = "/home/sunil-jadaun/Assignments/IS/Chandrayaan-3/ch3_rambha/LTA_RAMBHA-LP_August2024_final/data/raw/20230826_rawB"

# WHAT: full paths to the science file (voltage,current samples) and the ops
#       file (per-sweep metadata).
# WHEN: alongside BASE.
# WHY: the science file alone is just 835k anonymous rows; the ops file is the
#      'table of contents' that tells us where each sweep starts and ends.
SCI = f"{BASE}/ch3_rlp_nrB_sci_20230826T145357736.csv"
OPS = f"{BASE}/ch3_rlp_nrB_ops_20230826T145357736.csv"

# WHAT: physical constants in SI units.
# WHEN: defined once; used in the density formula.
# WHY: hard-coding them with names beats magic numbers scattered in formulas;
#      these exact values are CODATA standards.
Q_E = 1.602176634e-19    # electron charge, coulombs
K_B = 1.380649e-23       # Boltzmann constant, J/K
M_E = 9.1093837015e-31   # electron mass, kg

# WHAT: probe radius in metres and its sphere surface area.
# WHEN: once; used when converting measured current -> density.
# WHY: the density formula needs the collecting area. 0.025 m (2.5 cm) is the
#      spec value we used in the demo — VERIFY against the RAMBHA user manual
#      tomorrow (Day 2 reading) and correct if the manual differs. FLAGGED.
PROBE_R = 0.025
A_PROBE = 4 * np.pi * PROBE_R**2

# WHAT: read the science CSV into a table with two named columns.
# WHEN: start of the pipeline.
# WHY: names=['V','I'] labels the columns; skiprows=1 skips the header row so
#      the numbers parse as numbers.
sci = pd.read_csv(SCI, names=["V", "I"], skiprows=1)

# WHAT: force both columns to numeric, turning any stray text into NaN.
# WHEN: immediately after reading any real-world CSV.
# WHY: one corrupt row would otherwise make the whole column 'object' type and
#      silently break the math. errors='coerce' quarantines bad values as NaN.
sci["V"] = pd.to_numeric(sci["V"], errors="coerce")
sci["I"] = pd.to_numeric(sci["I"], errors="coerce")

# WHAT: drop NaN rows and renumber the index 0..N-1.
# WHEN: right after the coercion above.
# WHY: fitting code can't handle NaNs; reset_index keeps positions aligned
#      with our sweep-segmentation arithmetic below.
sci = sci.dropna().reset_index(drop=True)

# WHAT: read the ops (metadata) CSV; strip stray whitespace from headers.
# WHEN: start of pipeline, alongside the science read.
# WHY: PDS-exported CSVs often pad column names with spaces
#      ('Number_of_samples ' != 'Number_of_samples'); stripping avoids
#      KeyErrors that cost an hour of confusion.
ops = pd.read_csv(OPS, skipinitialspace=True)
ops.columns = [c.strip() for c in ops.columns]

# WHAT: pull the per-sweep sample counts as a plain Python list of ints.
# WHEN: before segmentation.
# WHY: this list is the ruler we use to cut the 835k-row stream into sweeps:
#      sweep 0 = first n0 rows, sweep 1 = next n1 rows, and so on.
n_samps = ops["Number_of_samples"].astype(int).tolist()

# WHAT: print a receipt that the files loaded and the counts reconcile.
# WHEN: after loading, before any analysis.
# WHY: receipts-not-claims. If total samples != sum of ops counts, STOP —
#      the segmentation assumption is broken and everything downstream lies.
print(f"science samples: {len(sci):,} | ops sweeps: {len(ops)} | ops total: {sum(n_samps):,}")

# WHAT: cut the sample stream into individual sweeps using the ops counts.
# WHEN: once per session file.
# WHY: each sweep is one -12V→+12V measurement — the atomic unit of this
#      entire project. Everything (classical fit, ML input) operates per-sweep.
sweeps, start = [], 0
for i, n in enumerate(n_samps):
    # WHAT: guard against the ops file claiming more samples than exist.
    # WHY: defensive; a truncated download would otherwise crash mid-loop.
    if start + n > len(sci):
        break
    seg = sci.iloc[start:start + n]
    sweeps.append({
        "idx": i,
        "V": seg["V"].to_numpy(),
        "I": seg["I"].to_numpy(),
        # WHAT: carry probe resistance metadata with each sweep.
        # WHY: THIS column is the whole project — it's how we'll separate the
        #      'easy' sweeps from the high-resistance 'failing' ones.
        "probe_res": ops.iloc[i]["Probe_res(ohm)"],
    })
    start += n
print(f"segmented {len(sweeps)} sweeps")

# WHAT: function to average the many repeated samples at each voltage step
#       into one clean (V, I) point per step.
# WHEN: applied to every sweep before fitting.
# WHY: the instrument dwells ~60 samples per 0.1V step; averaging them crushes
#      random noise by ~sqrt(60) ≈ 8x for free. This is the simplest, most
#      honest denoising there is.
def bin_sweep(V, I, step=0.1):
    edges = np.arange(np.floor(V.min()), np.ceil(V.max()) + step, step)
    idx = np.digitize(V, edges)
    Vb, Ib = [], []
    for b in np.unique(idx):
        m = idx == b
        # WHY >=3: a bin with 1-2 stray samples is noise, not a real step.
        if m.sum() >= 3:
            Vb.append(V[m].mean())
            Ib.append(I[m].mean())
    return np.array(Vb), np.array(Ib)

# WHAT: the classical parameter extraction (simplified Manju-style).
# WHEN: per binned sweep.
# WHY: this is the published physics baseline our ML must match on clean
#      sweeps and beat on distorted ones. Each step maps to Day 4's theory —
#      today, run it; Day 4, we dissect it.
def estimate_params(Vb, Ib):
    out = {"V_float": np.nan, "Te_eV": np.nan, "Ne_cc": np.nan}
    # --- floating potential: where current crosses zero ---
    zc = np.where(np.diff(np.sign(Ib)) != 0)[0]
    if len(zc) == 0:
        return out
    k = zc[0]
    # WHAT: linear interpolation between the two points straddling zero.
    # WHY: more precise than picking the nearest 0.1V grid point.
    out["V_float"] = Vb[k] - Ib[k] * (Vb[k+1] - Vb[k]) / (Ib[k+1] - Ib[k])
    vf = out["V_float"]
    # --- remove the ion contribution: median current well below Vf ---
    # WHY: below Vf the probe repels electrons; what's left is ion current.
    #      Subtracting its level isolates the electron current we fit.
    ion_floor = np.median(Ib[Vb < vf - 2]) if (Vb < vf - 2).any() else 0.0
    Ie = Ib - ion_floor
    # --- Te from the slope of ln(Ie) vs V just above Vf ---
    # WHY: in the retardation region, electron current grows exponentially
    #      with voltage; the growth rate IS the temperature (Day 4 theory).
    mask = (Vb > vf) & (Vb < vf + 6) & (Ie > 0)
    if mask.sum() < 5:
        return out
    slope, _ = np.polyfit(Vb[mask], np.log(Ie[mask]), 1)
    if slope <= 0:
        return out
    out["Te_eV"] = 1.0 / slope
    # --- Ne from the electron current using thermal-flux relation ---
    Te_K = out["Te_eV"] * Q_E / K_B
    v_th = np.sqrt(8 * K_B * Te_K / (np.pi * M_E))       # today's equation!
    Iesat = Ie[Vb > vf].max()
    ne_m3 = 4 * Iesat / (Q_E * A_PROBE * v_th)
    out["Ne_cc"] = ne_m3 / 1e6                            # m^-3 -> cm^-3
    return out

# WHAT: run the extraction over all sweeps and collect a results table.
# WHEN: after segmentation.
# WHY: this table is (a) tonight's receipt, (b) eventually the pseudo-label
#      set the ML model trains on.
rows = []
for s in sweeps:
    Vb, Ib = bin_sweep(s["V"], s["I"])
    if len(Vb) < 20:
        continue
    p = estimate_params(Vb, Ib)
    p.update({"sweep": s["idx"], "probe_res": s["probe_res"]})
    rows.append(p)
res = pd.DataFrame(rows)

# WHAT: save the table and print the summary receipt.
# WHY: the counts below ARE Day 1's deliverable numbers for your notebook.
res.to_csv("day01_sweep_params.csv", index=False)
ok = res["Te_eV"].notna().sum()
print(f"valid fits: {ok}/{len(res)} | median Te={res['Te_eV'].median():.2f} eV | median Ne={res['Ne_cc'].median():.0f}/cc | median Vf={res['V_float'].median():.2f} V")

# WHAT: reproduce figure 1 — one representative sweep.
# WHY: seeing the S-curve on YOUR screen from YOUR machine is the day's
#      psychological milestone; it also verifies matplotlib works.
good = res.dropna(subset=["Te_eV"])
pick = int(good.iloc[0]["sweep"])
s = next(x for x in sweeps if x["idx"] == pick)
Vb, Ib = bin_sweep(s["V"], s["I"])
plt.figure(figsize=(7, 4.5))
plt.scatter(s["V"], s["I"] * 1e9, s=2, alpha=0.15, label="raw samples")
plt.plot(Vb, Ib * 1e9, lw=1.8, color="crimson", label="binned curve")
plt.axhline(0, color="k", lw=0.6)
plt.xlabel("Bias voltage V (V)"); plt.ylabel("Current I (nA)")
plt.title(f"RAMBHA-LP sweep #{pick} — 2023-08-26 (my machine, Day 1)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("day01_fig_sweep.png", dpi=130)
print("saved day01_fig_sweep.png")