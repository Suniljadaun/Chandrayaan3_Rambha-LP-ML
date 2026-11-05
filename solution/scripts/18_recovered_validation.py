#!/usr/bin/env python3
"""
Step 18 — How good are the sweeps the network recovers?

WHY THIS SCRIPT EXISTS
----------------------
Every real-data check in steps 6 and 15 uses sweeps the classical fit already handles. The
sweeps the network recovers from classical failures are accepted on one condition that does the
work: the curve rebuilt from the network's parameters matches the sweep with R^2 >= 0.90. That R^2
is taken on the linear current over the whole sweep, where the large saturation branch dominates,
so it is a weaker test than the classical one. Four checks here look at the recovered sweeps
directly.

  (a) TWIN HALVES. Each commanded sweep has an up and a down half seconds apart. Where one half
      passes the classical fit and the other half was recovered by the network, the classical
      value on the twin is an independent reference for the recovered half.

  (b) THE CLASSICAL TEST, APPLIED TO THE NETWORK. Put the network's (Ne, Te, Vp) through the same
      test the classical fit must pass: in the classical retardation window [V_float,
      min(V_float + 1.0 V, Vp)], with the classical baseline, does ln(I) follow the network's
      exponential with R^2 >= 0.90? Run on the recovered sweeps and, as a reference, on sweeps the
      classical fit passes.

  (c) THE HARD REGIME, IN SIMULATION. Most recovered sweeps have no classical twin, so nothing
      real checks them. Build simulated sweeps with the parameters and probe-resistor settings
      the network reports for exactly those sweeps, with the measured noise and tilt, and measure
      the network's error there. Their parameters only set the regime; the truth of each
      simulated curve is known.

  (d) CLIPPING. On the 1 Mohm and 20 Mohm settings the front end saturates and the top of the
      curve goes flat. Clipped sweeps are excluded from recovery (config.yaml, recovery); this
      section shows why, by giving the same twin-half comparison for the clipped sweeps that
      would otherwise have passed every recovery condition.

Also draws F7: one real sweep the classical fit rejected, with the network's rebuilt curve.

Outputs:
  outputs/tables/recovered_validation.json
  outputs/figures/F7_recovered_example.png
"""
import _bootstrap as B
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rambhalp.config import load_config, out
from rambhalp import train, infer, physics, classical
from rambhalp.model import predict
from rambhalp.preprocess import common_grid

RESISTOR_CHANNEL = {49900.0: 0, 1000000.0: 1, 20000000.0: 2}   # order of noise_frac_by_channel


def med_abs(x):
    return float(np.median(np.abs(x)))


def twin_halves(j):
    """(a) recovered half against the classical value on its twin."""
    j = j.copy()
    j["cls"] = np.where(~j["failed"], "C", np.where(j["recovered"], "R", "X"))
    p = j.pivot_table(index=["timestamp", "segment"], columns="direction",
                      values=["cls", "Ne_cc_c", "Te_eV_c", "Ne_cc_m", "Te_eV_m"], aggfunc="first")
    cls = p["cls"].dropna()
    mixed = cls[((cls["up"] == "C") & (cls["down"] == "R")) | ((cls["up"] == "R") & (cls["down"] == "C"))]
    rec_half = np.where(mixed["up"] == "R", "up", "down")
    cl_half = np.where(mixed["up"] == "C", "up", "down")
    idx = mixed.index
    ne_r = np.array([p["Ne_cc_m"].loc[k, h] for k, h in zip(idx, rec_half)])
    ne_c = np.array([p["Ne_cc_c"].loc[k, h] for k, h in zip(idx, cl_half)])
    te_r = np.array([p["Te_eV_m"].loc[k, h] for k, h in zip(idx, rec_half)])
    te_c = np.array([p["Te_eV_c"].loc[k, h] for k, h in zip(idx, cl_half)])
    dn = np.log10(ne_r) - np.log10(ne_c)
    dt = te_r - te_c

    # the same comparison where BOTH halves pass the classical fit, as the reference level
    both = cls[(cls["up"] == "C") & (cls["down"] == "C")].index
    dn_ref = np.log10(p["Ne_cc_m"].loc[both, "down"]) - np.log10(p["Ne_cc_c"].loc[both, "up"])
    dt_ref = p["Te_eV_m"].loc[both, "down"] - p["Te_eV_c"].loc[both, "up"]

    rec = j[j["failed"] & j["recovered"]]
    twin_status = {}
    for k, d in zip(zip(rec["timestamp"], rec["segment"]), rec["direction"]):
        other = "down" if d == "up" else "up"
        s = cls.loc[k, other] if k in cls.index else None
        twin_status[s] = twin_status.get(s, 0) + 1
    return {
        "n_recovered": int(len(rec)),
        "recovered_by_twin_status": {"classical_pass": twin_status.get("C", 0),
                                     "also_recovered": twin_status.get("R", 0),
                                     "failed_both": twin_status.get("X", 0)},
        "mixed_pairs": int(len(mixed)),
        "Ne_dex_median_signed": float(np.median(dn)), "Ne_dex_median_abs": med_abs(dn),
        "Te_eV_median_signed": float(np.median(dt)), "Te_eV_median_abs": med_abs(dt),
        "reference_both_classical_pass": {
            "pairs": int(len(both)),
            "Ne_dex_median_abs": med_abs(dn_ref), "Te_eV_median_abs": med_abs(dt_ref)},
    }


def classical_style_test(sw, fit, Ne, Te, Vp, cfg):
    """(b) R^2 of ln(I) against the network's exponential, in the classical retardation window."""
    ccfg = cfg["classical"]
    V, I = np.asarray(sw.V, float), np.asarray(sw.I, float)
    o = np.argsort(V)
    V, I = V[o], I[o]
    if np.corrcoef(V, I)[0, 1] < 0:
        I = -I
    vf = fit.V_float
    bm = np.zeros(V.size, dtype=bool)
    if np.isfinite(vf):
        bm = V <= vf - ccfg["ion_floor_offset"]
    if bm.sum() < 3:
        bm[:] = False
        bm[:max(3, V.size // 10)] = True
    noise = float(np.std(I[bm])) + 1e-30
    Ie = I - float(np.median(I[bm]))
    if np.isfinite(vf) and V[0] < vf < V[-1]:
        anchor = vf
    else:
        lift = np.where(Ie > max(2.0 * noise, 0.02 * np.percentile(Ie, 97)))[0]
        if lift.size == 0:
            return None
        anchor = float(V[lift[0]])
    top = min(anchor + ccfg["retardation_span"], Vp)
    m = (V >= anchor) & (V <= top) & (Ie > 1.5 * noise)
    if m.sum() < ccfg["min_fit_points"]:
        return None
    y = np.log(Ie[m])
    ie0 = physics.electron_sat_current(Ne * 1e6, Te, cfg)
    yhat = np.log(ie0) + (V[m] - Vp) / Te
    ss = np.sum((y - y.mean()) ** 2)
    if ss <= 0:
        return None
    slope = np.polyfit(V[m], y, 1)[0]
    # Same slope, intercept free: tests the network's temperature alone, without the I_e0 and
    # baseline differences that a fixed intercept also carries.
    shape = V[m] / Te
    yshape = shape + np.mean(y - shape)
    return {"r2": float(1.0 - np.sum((y - yhat) ** 2) / ss),
            "r2_slope_only": float(1.0 - np.sum((y - yshape) ** 2) / ss),
            "te_local": float(1.0 / slope) if slope > 0 else np.nan}


def classical_style(sweeps, key_to_i, j, cfg):
    def run(rows):
        r2, r2s, ratio, n_test = [], [], [], 0
        for _, r in rows.iterrows():
            sw = sweeps[key_to_i[(r["timestamp"], r["idx"])]]
            res = classical_style_test(sw, classical.fit_sweep(sw, cfg), r["Ne_cc_m"], r["Te_eV_m"],
                                       r["Vp_V_m"], cfg)
            if res is None:
                continue
            n_test += 1
            r2.append(res["r2"])
            r2s.append(res["r2_slope_only"])
            if np.isfinite(res["te_local"]):
                ratio.append(r["Te_eV_m"] / res["te_local"])
        r2, r2s = np.array(r2), np.array(r2s)
        return {"n": int(len(rows)), "n_testable": n_test,
                "pass_fraction_r2_ge_0.90": round(float(np.mean(r2 >= 0.90)), 3) if n_test else None,
                "median_r2": round(float(np.median(r2)), 3) if n_test else None,
                "pass_fraction_slope_only_r2_ge_0.90":
                    round(float(np.mean(r2s >= 0.90)), 3) if n_test else None,
                "median_r2_slope_only": round(float(np.median(r2s)), 3) if n_test else None,
                "median_Te_network_over_local_slope": round(float(np.median(ratio)), 3) if ratio else None}

    return {"recovered": run(j[j["failed"] & j["recovered"]]),
            "classical_pass_reference": run(j[~j["failed"]])}


def hard_regime(model, j, sweeps, key_to_i, cfg, rng):
    """(c) network error on simulated sweeps placed where the recovered sweeps sit."""
    grid = common_grid(cfg)
    scfg = cfg["synthetic"]
    chans = scfg["noise_frac_by_channel"]
    ion_lo, ion_hi = scfg["ion_current_frac"]

    def simulate(rows):
        X = np.empty((len(rows), grid.size), dtype=np.float32)
        truth = rows[["Ne_cc_m", "Te_eV_m", "Vp_V_m"]].to_numpy(float)
        for i, (Ne, Te, Vp) in enumerate(truth):
            sw = sweeps[key_to_i[(rows.iloc[i]["timestamp"], rows.iloc[i]["idx"])]]
            lo, hi = chans[RESISTOR_CHANNEL.get(float(sw.probe_res), 1)]
            I = physics.forward_iv_np(grid, Ne, Te, Vp, cfg)
            span = np.ptp(I) + 1e-30
            mag = 10 ** rng.uniform(np.log10(ion_lo), np.log10(ion_hi))
            tilt = rng.choice([-1.0, 1.0]) * mag * span * (grid - grid.min()) / np.ptp(grid)
            nf = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
            X[i] = physics.to_input_np(I + tilt + rng.normal(0.0, nf * span, grid.size))
        p = predict(model, X, cfg, mc_samples=cfg["validate"]["mc_samples"])
        dn = np.log10(p["Ne_cc"]) - np.log10(truth[:, 0])
        dt = p["Te_eV"] - truth[:, 1]
        return {"n": int(len(rows)),
                "median_true_Ne_cc": round(float(np.median(truth[:, 0])), 1),
                "median_true_Te_eV": round(float(np.median(truth[:, 1])), 3),
                "Ne_dex_median_abs": round(med_abs(dn), 4),
                "Te_eV_median_abs": round(med_abs(dt), 4),
                "Te_relative_median_abs": round(float(np.median(np.abs(dt) / truth[:, 1])), 3),
                "Vp_V_median_abs": round(med_abs(p["Vp_V"] - truth[:, 2]), 3)}

    rec = j[j["failed"] & j["recovered"]].copy()
    cls = j.assign(c=np.where(~j["failed"], "C", "F")).pivot_table(
        index=["timestamp", "segment"], columns="direction", values="c", aggfunc="first")
    def twin_is_classical(r):
        other = "down" if r["direction"] == "up" else "up"
        k = (r["timestamp"], r["segment"])
        return k in cls.index and cls.loc[k, other] == "C"
    has_twin = rec.apply(twin_is_classical, axis=1)
    return {"no_classical_twin": simulate(rec[~has_twin]),
            "with_classical_twin": simulate(rec[has_twin])}


def clipping(j, sweeps, key_to_i, cfg):
    """(d) how often sweeps are clipped, and how wrong the network is on clipped ones."""
    ccfg = cfg["classical"]
    lo_te, hi_te = ccfg["te_plausible_eV"]
    lo_ne, hi_ne = ccfg["ne_plausible_cc"]
    res = np.array([sweeps[key_to_i[(t, i)]].probe_res for t, i in zip(j["timestamp"], j["idx"])])
    would = (j["Ne_cc_m"].between(lo_ne, hi_ne) & j["Te_eV_m"].between(lo_te, hi_te)
             & (j["recon_r2"] >= ccfg["min_r2"]) & (j["Ne_dex_std"] < 0.5))
    # the twin-half comparison, for failures that pass every condition except clipping
    jj = j.assign(cls=np.where(~j["failed"], "C", np.where(would, "W", "X")))
    p = jj.pivot_table(index=["timestamp", "segment"], columns="direction",
                       values=["cls", "clipped", "Ne_cc_c", "Te_eV_c", "Ne_cc_m", "Te_eV_m"],
                       aggfunc="first")
    cls = p["cls"].dropna()
    rows = []
    for k in cls.index:
        u, d = cls.loc[k, "up"], cls.loc[k, "down"]
        if {u, d} == {"C", "W"}:
            wh = "up" if u == "W" else "down"
            ch = "down" if wh == "up" else "up"
            rows.append((bool(p["clipped"].loc[k, wh]),
                         np.log10(p["Ne_cc_m"].loc[k, wh]) - np.log10(p["Ne_cc_c"].loc[k, ch]),
                         p["Te_eV_m"].loc[k, wh] - p["Te_eV_c"].loc[k, ch]))
    t = pd.DataFrame(rows, columns=["clipped", "dn", "dt"])

    def pct(dex):
        return round(float((10 ** dex - 1) * 100), 1)
    by_clip = {}
    for flag, g in t.groupby("clipped"):
        by_clip["clipped" if flag else "not_clipped"] = {
            "pairs": int(len(g)),
            "Ne_bias_percent": pct(g["dn"].median()),
            "Ne_typical_difference_percent": pct(g["dn"].abs().median()),
            "Te_bias_eV": round(float(g["dt"].median()), 3),
            "Te_typical_difference_eV": round(float(g["dt"].abs().median()), 3)}
    fail = j["failed"]
    return {
        "n_clipped": int(j["clipped"].sum()),
        "fraction_clipped": round(float(j["clipped"].mean()), 3),
        "fraction_clipped_by_probe_resistor_ohm": {
            f"{r:.0f}": round(float(j["clipped"][res == r].mean()), 3) for r in sorted(set(res))},
        "classical_pass_clipped": int((~fail & j["clipped"]).sum()),
        "classical_fail_clipped": int((fail & j["clipped"]).sum()),
        "failures_passing_all_but_clipping": int((fail & would & j["clipped"]).sum()),
        "twin_half_check_for_would_be_recoveries": by_clip,
    }


def figure(model, j, sweeps, key_to_i, cfg):
    """F7: a recovered sweep from a quiet channel, typical rebuild R^2, classical rejected it."""
    rec = j[j["failed"] & j["recovered"]].copy()
    rec["res"] = [sweeps[key_to_i[(t, i)]].probe_res for t, i in zip(rec["timestamp"], rec["idx"])]
    cand = rec[(rec["res"] == 1000000.0) & rec["fail_reason"].str.startswith("R2")]
    target = cand["recon_r2"].median()
    r = cand.iloc[int(np.argmin(np.abs(cand["recon_r2"] - target)))]
    sw = sweeps[key_to_i[(r["timestamp"], r["idx"])]]
    grid = common_grid(cfg)
    _, elec = infer.prepare_input([sw], cfg)
    data = elec[0] * 1e9
    fit = physics.forward_iv_np(grid, r["Ne_cc_m"], r["Te_eV_m"], r["Vp_V_m"], cfg) * 1e9

    fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax[0].plot(grid, data, ".", ms=3.5, color="#3f4a5a", label="measured (baseline removed)")
    ax[0].plot(grid, fit, color="#0f766e", lw=2,
               label=f"rebuilt from network: Ne {r['Ne_cc_m']:.0f} cm$^{{-3}}$, "
                     f"Te {r['Te_eV_m']:.2f} eV")
    ax[0].axvline(r["Vp_V_m"], color="#c2410c", ls="--", lw=1)
    ax[0].text(r["Vp_V_m"] + 0.2, data.max() * 0.08, f"$V_p$ = {r['Vp_V_m']:.1f} V",
               color="#c2410c", fontsize=9)
    ax[0].set_ylabel("Electron current (nA)")
    ax[0].legend(fontsize=8.5, loc="upper left")
    ax[0].grid(alpha=0.25)
    ax[0].set_title("(a)", loc="left", fontweight="bold")
    ax[1].plot(grid, (data - fit) / data.max() * 100, ".", ms=3.5, color="#3f4a5a")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("Residual (% of peak)")
    ax[1].set_xlabel("Bias voltage (V)")
    ax[1].grid(alpha=0.25)
    ax[1].set_title("(b)", loc="left", fontweight="bold")
    fig.tight_layout()
    path = out(cfg, "figures", "F7_recovered_example.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"timestamp": r["timestamp"], "idx": int(r["idx"]), "date": r["date"],
            "classical_fail_reason": r["fail_reason"], "recon_r2": round(float(r["recon_r2"]), 4),
            "Ne_cc": round(float(r["Ne_cc_m"]), 1), "Te_eV": round(float(r["Te_eV_m"]), 3),
            "Vp_V": round(float(r["Vp_V_m"]), 2), "figure": path.name}


def main():
    cfg = load_config()
    print("=" * 70)
    print("STEP 18: validation of the recovered sweeps")
    print("=" * 70)
    rng = np.random.default_rng(2026)
    sweeps = B.load_pickle(B.cache_dir(cfg) / "sweeps.pkl")
    key_to_i = {(s.timestamp, s.idx): i for i, s in enumerate(sweeps)}
    c = pd.read_csv(out(cfg, "tables", "classical_baseline.csv"))
    m = pd.read_csv(out(cfg, "tables", "ml_predictions.csv"))
    for d in (c, m):
        d["timestamp"] = d["timestamp"].astype(str)
        d["date"] = d["date"].astype(str)
    j = c.merge(m[["timestamp", "idx", "Ne_cc", "Te_eV", "Vp_V", "Ne_dex_std", "recon_r2",
                   "clipped", "recovered"]],
                on=["timestamp", "idx"], suffixes=("_c", "_m"))
    j["fail_reason"] = j["fail_reason"].fillna("")
    model = train.load_checkpoint(cfg, out(cfg, "checkpoints", "pinn.pt"))
    import torch
    torch.manual_seed(cfg["train"]["seed"])

    report = {"a_twin_halves": twin_halves(j)}
    print(json.dumps(report["a_twin_halves"], indent=2))
    report["b_classical_test_on_network_parameters"] = classical_style(sweeps, key_to_i, j, cfg)
    print(json.dumps(report["b_classical_test_on_network_parameters"], indent=2))
    report["d_clipping"] = clipping(j, sweeps, key_to_i, cfg)
    print(json.dumps(report["d_clipping"], indent=2))
    report["c_simulated_hard_regime"] = hard_regime(model, j, sweeps, key_to_i, cfg, rng)
    print(json.dumps(report["c_simulated_hard_regime"], indent=2))
    report["figure_F7"] = figure(model, j, sweeps, key_to_i, cfg)
    print(json.dumps(report["figure_F7"], indent=2))

    path = out(cfg, "tables", "recovered_validation.json")
    path.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
