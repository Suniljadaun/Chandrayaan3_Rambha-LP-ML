#!/usr/bin/env python3
"""
Step 11 — The local-temperature diagnostic, with the control it always needed.

WHY THIS SCRIPT EXISTS
----------------------
The entire two-population line of work (Phases 8-14) rests on one measurement: the local
temperature d(ln Ie)/dV varying five-fold across the retarding region of a stacked curve,
reported as 1.70 eV at 1.9 V below Vp falling to 0.36 eV near Vp over 1403 sweeps. A single
Maxwellian requires that profile to be flat, so a five-fold variation looked decisive.

That measurement was made by a throwaway script that was never saved. Two records of it in this
repository disagree with each other -- CODE_REVIEW.md has
"1.70 1.62 1.46 1.17 0.74 0.41 0.36 0.35 0.49" and twopop.py's docstring has
"1.70 1.46 0.98 0.55 0.36 0.35 0.49" -- and neither can be regenerated. This script makes the
diagnostic reproducible, and adds the control that was missing.

THE MISSING CONTROL
-------------------
A stacked curve is an AVERAGE of sweeps. Each sweep has its own temperature, and a sum of
exponentials with different decay constants is not an exponential -- it has a graded local slope.
So stacking sweeps drawn from a distribution of temperatures produces the two-population
signature even when every individual sweep is a textbook single Maxwellian.

This is not a small effect here. The archive's classical temperatures run from 0.24 eV at the 1st
percentile to 1.56 eV at the 99th, and the deep retarding region of a stack is dominated by its
hottest members: sweeps above the 90th percentile of Te supply 65% of the stacked current at
u = -1.9 V while being 10% of the sample.

So the diagnostic must always be run against a matched synthetic control: take the SAME sweeps,
give each a PERFECT single Maxwellian at its own measured Te and Ne, stack them identically, and
compare. The real profile is evidence for a second population only insofar as it exceeds the
control. Reporting the real profile alone -- which is what Phase 8 did -- measures the width of
the temperature distribution, not the number of populations.

Outputs:
  outputs_2pop/tables/local_te_diagnostic.json
"""
import _bootstrap as B
import argparse
import dataclasses
import json
import numpy as np
import pandas as pd
from rambhalp.config import load_config, out
from rambhalp import classical as classical_mod


def local_te(u, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.gradient(np.log(np.maximum(y, 1e-300)), u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u-lo", type=float, default=-2.2)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--min-stack", type=int, default=300)
    args = ap.parse_args()

    cfg = load_config()
    cfg["synthetic"]["two_population"] = True
    from pathlib import Path as _P
    bp = _P(cfg["_solution_dir"]) / "outputs_2pop"
    for k, sub in (("outputs", ""), ("figures", "figures"),
                   ("tables", "tables"), ("checkpoints", "checkpoints")):
        q = bp / sub if sub else bp
        q.mkdir(parents=True, exist_ok=True)
        cfg["paths"]["_" + k + "_abs"] = q

    print("=" * 78)
    print("STEP 11: local-temperature diagnostic, with matched single-Maxwellian control")
    print("=" * 78)

    sweeps = B.load_pickle(B.cache_dir(cfg) / "sweeps.pkl")
    cdf = pd.read_csv(out(cfg, "tables", "classical_baseline.csv"))
    fields = {f.name for f in dataclasses.fields(classical_mod.ClassicalResult)}
    usable = [c for c in cdf.columns if c in fields]
    ok = cdf[~cdf["failed"].astype(bool)]
    cmap = {(r["timestamp"], r["idx"]): classical_mod.ClassicalResult(**{k: r[k] for k in usable})
            for _, r in ok.iterrows()}

    pr = cfg["probe"]
    m_i = float(pr.get("m_i_amu", 1.0)) * 1.66053906660e-27
    K = float(np.log(np.sqrt(m_i / (2.0 * np.pi * pr["m_e"]))))
    for r in cmap.values():
        if not np.isfinite(r.V_float) and np.isfinite(r.Vp_V) and np.isfinite(r.Te_eV):
            r.V_float = float(r.Vp_V - r.Te_eV * K)

    UG = np.arange(args.u_lo, 0.0001, args.step)
    groups = {}
    for sw in sweeps:
        r = cmap.get((sw.timestamp, sw.idx))
        if r is None or not np.isfinite(r.Vp_V) or not np.isfinite(r.V_float):
            continue
        if not (np.isfinite(r.Te_eV) and r.Te_eV > 0 and np.isfinite(r.Ne_cc) and r.Ne_cc > 0):
            continue
        V = np.asarray(sw.V, float); I = np.asarray(sw.I, float)
        o = np.argsort(V); V, I = V[o], I[o]
        if np.corrcoef(V, I)[0, 1] < 0:
            I = -I
        ion = V <= (r.V_float - cfg["classical"]["ion_floor_offset"])
        if ion.sum() < 8:
            continue
        Ie = I - float(np.median(I[ion]))
        u = V - r.Vp_V
        if u.min() > args.u_lo - 0.1 or u.max() < 0.3:
            continue
        rec = (np.interp(UG, u, Ie, left=np.nan, right=np.nan), r.Te_eV, r.Ne_cc)
        groups.setdefault("ALL", []).append(rec)
        groups.setdefault(f"channel_{int(sw.adc_channel)}", []).append(rec)
        groups.setdefault(f"date_{sw.date}", []).append(rec)

    report = {"u_grid": UG.tolist()}
    show = [np.argmin(abs(UG + d)) for d in (1.9, 1.5, 1.0, 0.6, 0.3)]
    hdr = "  ".join(f"{-UG[j]:>5.1f}" for j in show)
    print(f"\nlocal Te (eV) at u = -{hdr} V below Vp\n")
    print(f"  {'group':<18} {'n':>6}  {'source':<22} {hdr}")

    for name in ["ALL"] + sorted(k for k in groups if k != "ALL"):
        g = groups[name]
        if len(g) < args.min_stack:
            continue
        R = np.vstack([x[0] for x in g])
        Te = np.array([x[1] for x in g]); Ne = np.array([x[2] for x in g])
        with np.errstate(invalid="ignore"):
            real = np.nanmean(R, axis=0)
        # matched control: every sweep a PERFECT single Maxwellian at its own Te, Ne
        ctrl = ((Ne * np.sqrt(Te))[:, None] * np.exp(UG[None, :] / Te[:, None])).mean(axis=0)
        tr, tc = local_te(UG, real), local_te(UG, ctrl)
        med = float(np.median(Te))

        def row(tag, t):
            return f"  {name:<18} {len(g):>6}  {tag:<22} " + "  ".join(
                f"{t[j]:>5.2f}" for j in show)
        print(row("REAL stacked", tr))
        print(row("control: 1 Maxwellian", tc))
        excess = [float(tr[j] - tc[j]) for j in show]
        print(f"  {'':<18} {'':>6}  {'REAL minus control':<22} " + "  ".join(
            f"{e:>+5.2f}" for e in excess))
        report[name] = {"n": len(g), "median_Te_eV": med,
                        "real_localTe": [float(tr[j]) for j in show],
                        "control_localTe": [float(tc[j]) for j in show],
                        "excess_over_control": excess,
                        "u_reported": [float(UG[j]) for j in show]}
        print()

    print("=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("The 'REAL minus control' row is the only line that carries evidence.")
    print()
    print("  clearly POSITIVE and growing towards the left  -> a second, hotter population")
    print("  near ZERO                                      -> the real curves are no more")
    print("       graded than a stack of single Maxwellians; a second population is NOT")
    print("       supported, and the raw profile was measuring the SPREAD of temperatures")
    print("  NEGATIVE -> the real curves are FLATTER than the control, which argues against")
    print("       any additional hot component")
    print()
    print("Phase 8 reported the REAL row alone (1.70 / 1.17 / 0.36 over 1403 sweeps) and read")
    print("its five-fold variation as two populations. Without the control row that inference")
    print("cannot be made, because a spread of single-population temperatures produces the")
    print("same shape.")

    path = out(cfg, "tables", "local_te_diagnostic.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
