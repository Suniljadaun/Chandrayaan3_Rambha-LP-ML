"""
figures.py — every report figure, generated from saved tables so they are reproducible.

Figures produced:
  F1  example sweep + classical fit (what the baseline does)
  F2  classical failure map: fraction of sweeps failed per day + reason breakdown
  F3  training curves (total / supervised / physics loss)
  F4  synthetic hold-out: predicted vs true (Ne, Te)
  F5  recovery: recovered-sweep count and how it adds to the classical yield
  F6  blind test: per-day median recovered Ne across the mission, thin days marked

All use matplotlib's Agg backend so they render headless (laptop, Kaggle, Colab).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import out

C_ORANGE, C_TEAL, C_PURPLE, C_GREY = "#c2410c", "#0f766e", "#6d28d9", "#7a8ca8"


def fig_example_sweep(sweeps, classical_rows, cfg):
    """F1: one representative sweep with its classical fit annotations."""
    good = next((i for i, r in enumerate(classical_rows) if not r.failed), 0)
    sw, r = sweeps[good], classical_rows[good]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(sw.V, sw.I * 1e9, color=C_ORANGE, lw=1.8, label="binned I–V")
    ax.axhline(0, color="k", lw=0.6)
    if np.isfinite(r.V_float):
        ax.axvline(r.V_float, color=C_TEAL, ls="--", lw=1,
                   label=f"V_float = {r.V_float:.2f} V")
    txt = f"Te = {r.Te_eV:.2f} eV\nNe = {r.Ne_cc:.0f} cm⁻³\nR² = {r.r2:.2f}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=C_GREY, alpha=0.9))
    ax.set_xlabel("Bias voltage V (V)"); ax.set_ylabel("Probe current I (nA)")
    ax.set_title(f"RAMBHA-LP sweep — {sw.date}, classical fit")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); _save(fig, cfg, "F1_example_sweep.png")


def fig_failure_map(classical_df: pd.DataFrame, cfg):
    """F2: per-day classical failure fraction."""
    g = classical_df.groupby("date")["failed"].agg(["mean", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    xs = range(len(g))
    ax.bar(list(xs), g["mean"] * 100, color=C_ORANGE, alpha=0.85)
    ax.set_ylabel("classical FAIL rate (%)")
    ax.set_title(f"Where the classical method fails "
                 f"(overall {classical_df['failed'].mean()*100:.0f}% of "
                 f"{len(classical_df)} sweeps)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(g["date"].astype(str), rotation=60, ha="right", fontsize=7)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); _save(fig, cfg, "F2_failure_map.png")


def fig_training_curves(history: dict, cfg):
    """F3: loss curves."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(history["loss"], color=C_ORANGE, label="total")
    ax.plot(history["sup"], color=C_TEAL, label="supervised")
    ax.plot(history["phys"], color=C_PURPLE, label="physics")
    if history.get("val"):
        ax.plot(history["val"], color="k", ls="--", label="val (sup)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_yscale("log")
    ax.set_title("Physics-informed training"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); _save(fig, cfg, "F3_training_curves.png")


def fig_synthetic_scatter(model, cfg):
    """F4: predicted vs true on a synthetic hold-out (Ne, Te)."""
    from . import synthetic
    from .model import predict
    X, Y, _ = synthetic.make_dataset(3000, cfg, seed=2024, regime="lunar")
    pred = predict(model, X, cfg, mc_samples=1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].scatter(Y[:, 0], np.log10(pred["Ne_cc"]), s=4, alpha=0.25, color=C_TEAL)
    lim = [Y[:, 0].min(), Y[:, 0].max()]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("true log₁₀ Ne"); ax[0].set_ylabel("pred log₁₀ Ne")
    ax[0].set_title("Density"); ax[0].grid(alpha=0.25)
    ax[1].scatter(Y[:, 1], pred["Te_eV"], s=4, alpha=0.25, color=C_ORANGE)
    lim = [Y[:, 1].min(), Y[:, 1].max()]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlabel("true Te (eV)"); ax[1].set_ylabel("pred Te (eV)")
    ax[1].set_title("Temperature"); ax[1].grid(alpha=0.25)
    fig.suptitle("Synthetic hold-out: network vs known truth", y=1.02)
    fig.tight_layout(); _save(fig, cfg, "F4_synthetic_scatter.png")


def fig_recovery(n_total, n_classical_ok, n_recovered, cfg):
    """F5: yield bar — classical alone vs classical + ML-recovered."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.bar(["classical\nonly"], [n_classical_ok], color=C_GREY, label="classical")
    ax.bar(["classical\n+ ML"], [n_classical_ok], color=C_GREY)
    ax.bar(["classical\n+ ML"], [n_recovered], bottom=[n_classical_ok],
           color=C_TEAL, label="ML-recovered")
    ax.set_ylabel("usable sweeps")
    ax.set_title(f"Recovered {n_recovered} of {n_total-n_classical_ok} failed sweeps\n"
                 f"(total sweeps: {n_total})")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); _save(fig, cfg, "F5_recovery.png")


def fig_density_trend(ml_df: pd.DataFrame, cfg):
    """F6: per-day median recovered density across the mission — the blind test.

    Replaces the old single-day hop figure. Two changes of substance:
      * only RECOVERED sweeps are included. The previous version took the median over every
        row, including the ~20% the model itself flagged as not recovered.
      * days with too few recovered sweeps to trust are drawn hollow and labelled, instead of
        being presented as if they carried the same weight as a day with 1500 sweeps. The
        terminator session (3 Sep 2023) is one of those, which is why the hop test it used to
        anchor was never answerable from this archive.
    Every bar is annotated with its sweep count, so the figure carries its own caveat.
    """
    vcfg = cfg.get("validate", {})
    min_n = int(vcfg.get("min_sweeps_per_day", 100))
    d = ml_df[(ml_df["Ne_cc"] > 0)]
    if "recovered" in d.columns:
        d = d[d["recovered"].astype(bool)]
    g = d.groupby("date")["Ne_cc"].agg(["median", "count"]).reset_index()
    g["date"] = g["date"].astype(str)
    g = g.sort_values("date")

    thin = g["count"] < min_n
    xs = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(xs[~thin], g["median"][~thin], color=C_TEAL, label=f"n >= {min_n} sweeps")
    ax.bar(xs[thin], g["median"][thin], facecolor="none", edgecolor=C_GREY, hatch="///",
           linewidth=1.2, label=f"n < {min_n} — too thin to trust")

    for x, (mval, cval) in enumerate(zip(g["median"], g["count"])):
        ax.annotate(f"n={cval}", (x, mval), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=6.5, color=C_GREY)

    solid = g[~thin]
    rho = float("nan")
    if len(solid) >= 3:
        ra = np.argsort(np.argsort(np.arange(len(solid)))).astype(float)
        rb = np.argsort(np.argsort(solid["median"].to_numpy())).astype(float)
        if ra.std() and rb.std():
            rho = float(np.corrcoef(ra, rb)[0, 1])

    ax.set_ylabel("median recovered Ne (cm$^{-3}$)")
    ax.set_title("Blind test — recovered density across the mission\n"
                 f"model never told the date; Spearman(day, median Ne) = {rho:.2f} "
                 f"over {len(solid)} well-sampled days")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(g["date"], rotation=60, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); _save(fig, cfg, "F6_density_trend.png")


def _save(fig, cfg, name):
    path = out(cfg, "figures", name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {path.name}")
