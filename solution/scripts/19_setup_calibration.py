#!/usr/bin/env python3
"""
Step 19 — Calibration between the six channel/resistor setups.

WHY THIS SCRIPT EXISTS
----------------------
Each ops file cycles through six setups (ADC channel 1 or 2, probe resistor 49.9 kOhm, 1 MOhm or
20 MOhm). Consecutive segments therefore sample the same plasma a few seconds apart through
different setups, so two setups that are both correctly calibrated must give the same density
for neighbouring segments. This script uses that to measure each setup's density offset.

REFERENCE. The three low-resistance setups (1/49.9k, 2/49.9k, 1/1M). Their unclipped classical
passes agree with each other to about 1 %, so they are taken as the calibration reference.

RATIO. For every sweep, its density divided by the median classical density of the unclipped
classical passes on the reference setups within +-6 segments in the same file (excluding the
sweep's own setup). Reported per setup and per date, for the classical fit and for the network.

ALSO REPORTED
  * how many sweeps inside the headline yield sit on a setup with an offset (the flag);
  * the share of the archive usable on the reference setups alone (no correction needed);
  * the network's own density error on the reference setups, where no offset is mixed in;
  * the clipped 1/20M sweeps: where their flat top begins and whether their values match the
    unclipped sweeps on that setup (a sensitivity figure for the headline).

Uses outputs of steps 1, 5 (classical_baseline.csv, ml_predictions.csv). No retraining.

Outputs:
  outputs/tables/setup_calibration.json
  outputs/tables/setup_flags.csv      one row per sweep: setup, usable, offset flag
"""
import _bootstrap as B
import json
import numpy as np
import pandas as pd
from rambhalp.config import load_config, out

RES_NAME = {49900.0: "49.9k", 1e6: "1M", 2e7: "20M"}
REFERENCE = ["1/49.9k", "2/49.9k", "1/1M"]
SEG_WINDOW = 6


def plateau_start_V(sw, cfg):
    """Voltage where the flat top of a binned sweep begins (first point within the clip tolerance
    of the maximum, counting back from the top end)."""
    rc = cfg.get("recovery", {})
    tol = float(rc.get("clip_flat_tolerance", 0.001))
    V, I = np.asarray(sw.V, float), np.asarray(sw.I, float)
    o = np.argsort(V)
    V, I = V[o], I[o]
    if np.corrcoef(V, I)[0, 1] < 0:
        I = -I
    flat = I.max() - I <= tol * (np.ptp(I) + 1e-30)
    k = len(I) - 1
    while k > 0 and flat[k - 1]:
        k -= 1
    return float(V[k])


def neighbour_median(rows, ref_by_file, col):
    outv = np.full(len(rows), np.nan)
    for j, r in enumerate(rows.itertuples()):
        g = ref_by_file.get(r.timestamp)
        if g is None:
            continue
        m = (np.abs(g.segment.values - r.segment) <= SEG_WINDOW) & (g.setup.values != r.setup)
        if m.any():
            outv[j] = float(np.median(g[col].values[m]))
    return outv


def summarise(r):
    r = r[np.isfinite(r) & (r > 0)]
    if len(r) == 0:
        return {"n": 0}
    return {"n": int(len(r)), "median_ratio": round(float(np.median(r)), 3),
            "within_10pct": round(float(np.mean(np.abs(r - 1) <= 0.1)), 3)}


def main():
    cfg = load_config()
    print("=" * 70)
    print("STEP 19: density calibration between channel/resistor setups")
    print("=" * 70)
    sweeps = B.load_pickle(B.cache_dir(cfg) / "sweeps.pkl")
    c = pd.read_csv(out(cfg, "tables", "classical_baseline.csv"))
    m = pd.read_csv(out(cfg, "tables", "ml_predictions.csv"))
    assert len(c) == len(m) == len(sweeps)

    d = pd.DataFrame({
        "timestamp": [s.timestamp for s in sweeps], "idx": [s.idx for s in sweeps],
        "date": [str(s.date) for s in sweeps], "segment": [int(s.segment) for s in sweeps],
        "setup": [f"{int(s.adc_channel)}/{RES_NAME[float(s.probe_res)]}" for s in sweeps],
        "classical_ok": ~c["failed"].astype(bool).values,
        "is_clipped": m["clipped"].astype(bool).values,
        "recovered": m["recovered"].astype(bool).values,
        "cNe": c["Ne_cc"].values, "cTe": c["Te_eV"].values,
        "mNe": m["Ne_cc"].values, "mTe": m["Te_eV"].values,
    })
    N = len(d)
    usable = (d.classical_ok | d.recovered) & ~d.is_clipped
    net_rec = d.recovered & ~d.classical_ok & ~d.is_clipped
    cls_ok = d.classical_ok & ~d.is_clipped

    ref = d[cls_ok & d.setup.isin(REFERENCE)]
    ref_by_file = {ts: g for ts, g in ref.groupby("timestamp")}

    d["ratio_classical"] = np.nan
    d["ratio_network"] = np.nan
    d.loc[cls_ok, "ratio_classical"] = d.loc[cls_ok, "cNe"].values / neighbour_median(
        d[cls_ok], ref_by_file, "cNe")
    d.loc[net_rec, "ratio_network"] = d.loc[net_rec, "mNe"].values / neighbour_median(
        d[net_rec], ref_by_file, "cNe")
    te_ratio_cls = d.loc[cls_ok, "cTe"].values / neighbour_median(d[cls_ok], ref_by_file, "cTe")
    d["te_ratio_classical"] = np.nan
    d.loc[cls_ok, "te_ratio_classical"] = te_ratio_cls

    report = {"reference_setups": REFERENCE, "segment_window": SEG_WINDOW, "setups": {}}
    print(f"\n{'setup':<9}{'sweeps':>7}{'clipped':>8}{'usable':>7}  classical ratio (n, med, <10%)"
          f"   network ratio")
    for su in ["1/49.9k", "2/49.9k", "1/1M", "1/20M", "2/1M", "2/20M"]:
        s = d.setup == su
        rc, rn = summarise(d.loc[s, "ratio_classical"].values), summarise(d.loc[s, "ratio_network"].values)
        by_date = d[s & cls_ok].groupby("date").ratio_classical.median().round(2)
        entry = {"n_sweeps": int(s.sum()), "n_clipped": int((s & d.is_clipped).sum()),
                 "n_usable": int((s & usable).sum()), "classical_Ne_ratio": rc,
                 "network_Ne_ratio": rn,
                 "classical_Te_ratio_median": round(float(np.nanmedian(d.loc[s, "te_ratio_classical"])), 2)
                 if np.isfinite(d.loc[s, "te_ratio_classical"]).any() else None,
                 "classical_Ne_ratio_by_date": {k: (None if pd.isna(v) else float(v))
                                                for k, v in by_date.items()}}
        report["setups"][su] = entry
        print(f"{su:<9}{entry['n_sweeps']:>7}{entry['n_clipped']:>8}{entry['n_usable']:>7}  {rc}   {rn}")
        print(f"{'':<9}by date: {entry['classical_Ne_ratio_by_date']}")

    # Offset flag: setups/dates whose classical ratio is outside +-10 % of the reference.
    by_sd = d[cls_ok].groupby(["setup", "date"]).ratio_classical.median()
    off = {k for k, v in by_sd.items() if np.isfinite(v) and abs(v - 1) > 0.10 and k[0] not in REFERENCE}
    d["offset_flag"] = [((su, dt) in off) for su, dt in zip(d.setup, d.date)]
    d.loc[d.setup == "2/20M", "offset_flag"] = True  # cannot be checked (almost all clipped)
    flagged = usable & d.offset_flag
    on_ref = usable & d.setup.isin(REFERENCE)
    report["flagged_setup_dates"] = sorted(f"{a} {b}" for a, b in off)
    report["n_usable"] = int(usable.sum())
    report["usable_fraction"] = round(float(usable.mean()), 4)
    report["n_usable_with_offset_flag"] = int(flagged.sum())
    report["usable_with_offset_flag_fraction_of_archive"] = round(float(flagged.sum() / N), 4)
    report["n_usable_on_reference_setups"] = int(on_ref.sum())
    report["usable_on_reference_setups_fraction"] = round(float(on_ref.sum() / N), 4)
    report["n_usable_unflagged"] = int((usable & ~d.offset_flag).sum())
    report["usable_unflagged_fraction"] = round(float((usable & ~d.offset_flag).sum() / N), 4)

    # Network's own error where no setup offset is mixed in.
    net_ref = {}
    for su in REFERENCE:
        r = d.loc[(d.setup == su) & net_rec, "ratio_network"].values
        r = r[np.isfinite(r) & (r > 0)]
        net_ref[su] = {"n": int(len(r)), "median_ratio": round(float(np.median(r)), 3),
                       "within_10pct": round(float(np.mean(np.abs(r - 1) <= 0.1)), 3),
                       "median_abs_pct_error": round(float(np.median(np.abs(r - 1)) * 100), 1)}
    r_all = d.loc[d.setup.isin(REFERENCE) & net_rec, "ratio_network"].values
    r_all = r_all[np.isfinite(r_all) & (r_all > 0)]
    net_ref["all_reference"] = {"n": int(len(r_all)), "median_ratio": round(float(np.median(r_all)), 3),
                                "within_10pct": round(float(np.mean(np.abs(r_all - 1) <= 0.1)), 3),
                                "median_abs_pct_error": round(float(np.median(np.abs(r_all - 1)) * 100), 1)}
    report["network_on_reference_setups"] = net_ref

    # Sensitivity: clipped 1/20M sweeps.
    s20 = (d.setup == "1/20M") & d.is_clipped
    idx20 = np.flatnonzero(s20.values)
    pstart = np.array([plateau_start_V(sweeps[i], cfg) for i in idx20])
    idx_other = np.flatnonzero((d.is_clipped & (d.setup == "2/1M")).values)
    pstart_other = np.array([plateau_start_V(sweeps[i], cfg) for i in idx_other])
    # what the network would recover on these sweeps if clipped sweeps were not excluded
    ccfg = cfg["classical"]
    lo_te, hi_te = ccfg["te_plausible_eV"]; lo_ne, hi_ne = ccfg["ne_plausible_cc"]
    would = (m.Ne_cc.between(lo_ne, hi_ne) & m.Te_eV.between(lo_te, hi_te)
             & (m.recon_r2 >= ccfg["min_r2"]) & (m.Ne_dex_std < 0.5)).values
    cand = s20 & (d.classical_ok | would)
    # compare their values to the reference, as for the unclipped sweeps of this setup
    cr = d.loc[cand & d.classical_ok, "cNe"].values / neighbour_median(d[cand & d.classical_ok], ref_by_file, "cNe")
    nr_ = d.loc[cand & ~d.classical_ok, "mNe"].values / neighbour_median(d[cand & ~d.classical_ok], ref_by_file, "cNe")
    report["clipped_1_20M"] = {
        "n_clipped": int(s20.sum()),
        "plateau_start_V_median": round(float(np.median(pstart)), 2),
        "plateau_start_V_p10_p90": [round(float(x), 2) for x in np.percentile(pstart, [10, 90])],
        "for_comparison_2_1M_clipped_plateau_start_V_median": round(float(np.median(pstart_other)), 2),
        "n_classical_pass_or_network_recovery": int(cand.sum()),
        "n_classical_pass": int((cand & d.classical_ok).sum()),
        "classical_Ne_ratio": summarise(cr), "network_Ne_ratio": summarise(nr_),
        "unclipped_1_20M_classical_Ne_ratio": report["setups"]["1/20M"]["classical_Ne_ratio"],
        "usable_fraction_if_admitted": round(float((usable | cand).sum() / N), 4),
        "n_usable_if_admitted": int((usable | cand).sum()),
    }

    print("\n" + json.dumps({k: v for k, v in report.items() if k != "setups"}, indent=2))
    path = out(cfg, "tables", "setup_calibration.json")
    path.write_text(json.dumps(report, indent=2))
    d.assign(usable=usable.values)[["date", "timestamp", "idx", "segment", "setup", "usable",
                                    "is_clipped", "offset_flag"]].to_csv(
        out(cfg, "tables", "setup_flags.csv"), index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
