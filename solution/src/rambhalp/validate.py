"""
validate.py — honest validation without ground truth.

We cannot check recovered numbers against a lunar 'truth' — there isn't one. So we validate
three ways, all of which the report can defend:

  1. Synthetic hold-out: on labelled synthetic sweeps we DO know the answer. Report the
     network's error there (this bounds best-case accuracy).
  2. Agreement where classical works: on sweeps the classical method fit cleanly, the network
     should broadly agree. Systematic disagreement there is a red flag.
  3. The density trend: the model is never told the date. If it learned physics rather than a
     dataset artefact, recovered density should vary smoothly and systematically across the
     mission. `density_trend` reports the per-day median over recovered sweeps and the
     monotonicity of that sequence. (This replaced a single-day "hop" test that this archive
     cannot support — see density_trend's docstring.)
  4. Ramp consistency: each commanded sweep is a triangle, so its rising and falling halves
     are two independent measurements of the same plasma seconds apart. Their disagreement is
     a systematic error bar derived from the data rather than from the model.

Uncertainty is reported throughout (MC-dropout), never hidden.
"""
from __future__ import annotations
import numpy as np


def synthetic_holdout_error(model, cfg, n=4000):
    """Median absolute error on a fresh synthetic set (best-case accuracy bound)."""
    from . import synthetic
    from .model import predict
    X, Y, params = synthetic.make_dataset(n, cfg, seed=999, regime="lunar")
    pred = predict(model, X, cfg, mc_samples=1)

    # Column layout depends on the mode. Three-parameter labels are [log10Ne, Te, Vp];
    # five-parameter labels are [log10Nc, Tc, log10(Nh+floor), Th, Vp], so Vp moves from
    # column 2 to column 4. Indexing Y[:, 2] unconditionally compares the predicted plasma
    # potential against a logarithmic density and reports a meaningless multi-volt error.
    two_pop = Y.shape[1] == 5
    vp_col = 4 if two_pop else 2

    err = {
        "Ne_dex_median_abs_err": float(np.median(np.abs(np.log10(pred["Ne_cc"]) - Y[:, 0]))),
        "Te_median_abs_err_eV": float(np.median(np.abs(pred["Te_eV"] - Y[:, 1]))),
        "Vp_median_abs_err_V": float(np.median(np.abs(pred["Vp_V"] - Y[:, vp_col]))),
        "n": n,
    }

    if two_pop and "Nh_cc" in pred:
        Ne_h_true = np.asarray(params["Ne_h"], dtype=float)
        Te_h_true = np.asarray(params["Te_h"], dtype=float)
        has_hot = Ne_h_true > 0.0
        no_hot = ~has_hot

        f_true = Ne_h_true / np.maximum(Ne_h_true + np.asarray(params["Ne"], dtype=float), 1e-12)
        f_pred = pred["Nh_cc"] / np.maximum(pred["Nh_cc"] + pred["Ne_cc"], 1e-12)

        err["hot"] = {
            "n_with_hot": int(has_hot.sum()),
            "n_without_hot": int(no_hot.sum()),
            # accuracy where a hot population exists
            "Th_median_abs_err_eV": (
                float(np.median(np.abs(pred["Th_eV"][has_hot] - Te_h_true[has_hot])))
                if has_hot.any() else None),
            "hot_fraction_median_abs_err": (
                float(np.median(np.abs(f_pred[has_hot] - f_true[has_hot])))
                if has_hot.any() else None),
            # the control: what it reports where there is none
            "false_hot_fraction_median": (
                float(np.median(f_pred[no_hot])) if no_hot.any() else None),
            "false_hot_fraction_p90": (
                float(np.percentile(f_pred[no_hot], 90)) if no_hot.any() else None),
            "note": ("false_* are measured on synthetic sweeps built with exactly zero hot "
                     "density; they bound how much of the archive hot component could be "
                     "the network inventing one"),
        }
    return err


def agreement_with_classical(ml_rows, classical_rows):
    """Compare ML vs classical on sweeps classical fit cleanly (matched by timestamp+idx)."""
    cmap = {(r.timestamp, r.idx): r for r in classical_rows if not r.failed}
    dNe, dTe = [], []
    for m in ml_rows:
        key = (m["timestamp"], m["idx"])
        if key in cmap:
            c = cmap[key]
            if c.Ne_cc and c.Ne_cc > 0 and m["Ne_cc"] > 0:
                dNe.append(np.log10(m["Ne_cc"]) - np.log10(c.Ne_cc))
            if np.isfinite(c.Te_eV):
                dTe.append(m["Te_eV"] - c.Te_eV)
    return {
        "n_overlap": len(dNe),
        "Ne_dex_median_diff": float(np.median(dNe)) if dNe else float("nan"),
        "Te_median_diff_eV": float(np.median(dTe)) if dTe else float("nan"),
    }


def density_trend(ml_rows, cfg):
    """Blind test, replacing the old single-day `hop_blind_test`.

    WHY THIS REPLACED THE HOP TEST. The hop test asked whether recovered density on the
    terminator day (3 Sep 2023) exceeds the pre-hop day (2 Sep). It cannot be answered with
    this archive: 2 Sep contributes ~1300 sweeps and 3 Sep only ~200, of which a fraction are
    recovered. A physical prediction tested against a couple of dozen sweeps is not a test,
    and the old implementation made it worse by taking the median over ALL rows including the
    ones the model itself flagged as not recovered.

    What this does instead is a stronger blind test on the same idea. The model is never told
    the date. If it has learned physics rather than a dataset artefact, recovered density
    should vary smoothly and systematically across the mission rather than jumping around.
    We report the per-day median over RECOVERED sweeps only, the Spearman rank correlation
    between day order and median density (monotonicity), and we mark days too thin to trust.

    Returns a dict with per-day medians, the trend statistic, and the pre-hop/terminator
    numbers kept as DESCRIPTIVE context with their sweep counts attached -- never as a
    pass/fail.
    """
    import collections
    vcfg = cfg.get("validate", {})
    min_n = int(vcfg.get("min_sweeps_per_day", 100))

    by_day = collections.defaultdict(list)
    for m in ml_rows:
        if m.get("recovered") and m["Ne_cc"] > 0:
            by_day[str(m["date"])].append(m["Ne_cc"])

    days = sorted(by_day)
    med = {d: float(np.median(by_day[d])) for d in days}
    cnt = {d: len(by_day[d]) for d in days}
    solid = [d for d in days if cnt[d] >= min_n]

    def _spearman(a, b):
        if len(a) < 3:
            return float("nan")
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        if ra.std() == 0 or rb.std() == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    rho = _spearman(np.arange(len(solid), dtype=float),
                    np.array([med[d] for d in solid], dtype=float))

    pre = str(vcfg.get("pre_hop_date", ""))
    term = str(vcfg.get("terminator_date", ""))
    return {
        "n_days": len(days),
        "min_sweeps_per_day_for_trend": min_n,
        "days_used_for_trend": solid,
        "days_too_thin": [d for d in days if cnt[d] < min_n],
        "median_Ne_cc_by_day": {d: round(med[d], 1) for d in days},
        "n_recovered_by_day": cnt,
        "spearman_day_vs_median_Ne": rho,
        "trend_direction": ("rising" if rho > 0.5 else
                            "falling" if rho < -0.5 else "no clear monotonic trend"),
        "context_pre_hop_date": pre,
        "context_terminator_date": term,
        "context_median_Ne_pre_hop_cc": med.get(pre, float("nan")),
        "context_n_pre_hop": cnt.get(pre, 0),
        "context_median_Ne_terminator_cc": med.get(term, float("nan")),
        "context_n_terminator": cnt.get(term, 0),
        "note": ("pre-hop/terminator values are descriptive context only, not a pass/fail: "
                 "the terminator session is far too thinly sampled to test a physical "
                 "prediction against. The crash epoch itself was acquired in zero-potential "
                 "mode and contains no sweeps at all."),
    }


def ramp_consistency(ml_rows):
    """Systematic-error estimate from the two halves of each commanded sweep.

    Every ops segment is a triangle ramp (-12 -> +12 -> -12) and preprocess.split_ramps now
    emits the rising and falling halves as two independent sweeps of the SAME plasma, seconds
    apart. Any disagreement between them is instrument systematics plus real short-timescale
    variability -- an error bar derived from the data itself, which is worth far more in a
    no-ground-truth setting than the MC-dropout number, since that only measures how unsure
    the network is about its own weights.

    Returns the paired scatter in dex, and the correlation between the two halves.
    """
    pairs = {}
    for m in ml_rows:
        if not m.get("recovered") or m["Ne_cc"] <= 0:
            continue
        key = (m["timestamp"], m.get("segment", -1))
        if key[1] == -1:
            continue
        pairs.setdefault(key, {})[m.get("direction", "")] = m["Ne_cc"]

    up, dn = [], []
    for v in pairs.values():
        if "up" in v and "down" in v:
            up.append(v["up"])
            dn.append(v["down"])
    if len(up) < 10:
        return {"n_pairs": len(up),
                "note": "too few up/down pairs — is preprocess.split_ramps enabled?"}
    up, dn = np.array(up), np.array(dn)
    d = np.log10(dn) - np.log10(up)
    return {
        "n_pairs": int(len(up)),
        "Ne_dex_median_offset_down_minus_up": float(np.median(d)),
        "Ne_dex_scatter_between_halves": float(np.std(d)),
        "Ne_dex_median_abs_difference": float(np.median(np.abs(d))),
        "correlation_log10_Ne": float(np.corrcoef(np.log10(up), np.log10(dn))[0, 1]),
        "note": ("scatter between the two halves of the same commanded sweep — a "
                 "data-derived systematic error bar, independent of MC-dropout"),
    }
