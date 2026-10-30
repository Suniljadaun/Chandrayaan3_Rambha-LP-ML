#!/usr/bin/env python3
"""
Step 16 — Fisher-information bound on the electron temperature.

Reported in Section 3.3.3 of the report. For one representative sweep (450 cm^-3, 0.4 eV) on the
240-point grid, with quiet-channel noise at 0.24% of the curve's span, this computes the
Cramer-Rao bound on Te, its correlation with Vp, and how much not knowing Vp inflates the bound.

Outputs: outputs/tables/fisher_bound.json
"""
import _bootstrap as B
import json
import numpy as np
from rambhalp.config import load_config, out
from rambhalp import physics
from rambhalp.preprocess import common_grid

NE_CC, TE_EV, NOISE_FRAC = 450.0, 0.4, 0.0024
VP_VALUES = (-3.9, -3.0, -1.5)


def bound(V, Vp, cfg):
    def curve(theta):
        return physics.forward_iv_np(V, 10 ** theta[0], theta[1], theta[2], cfg)

    theta = np.array([np.log10(NE_CC), TE_EV, Vp])
    y = curve(theta)
    sigma = NOISE_FRAC * (y.max() - y.min())
    h = 1e-4
    J = np.stack([(curve(theta + d) - curve(theta - d)) / (2 * h) for d in h * np.eye(3)], axis=1)
    F = J.T @ J / sigma ** 2
    C = np.linalg.inv(F)
    te_joint = float(np.sqrt(C[1, 1]))
    te_vp_known = float(np.sqrt(np.linalg.inv(F[:2, :2])[1, 1]))
    return dict(
        Vp_V=Vp,
        te_bound_eV=te_joint,
        te_bound_percent=100 * te_joint / TE_EV,
        te_bound_vp_known_eV=te_vp_known,
        inflation_from_vp=te_joint / te_vp_known,
        corr_te_vp=float(C[1, 2] / np.sqrt(C[1, 1] * C[2, 2])),
    )


def main():
    cfg = load_config()
    cfg["synthetic"]["two_population"] = False
    V = np.asarray(common_grid(cfg), float)
    rows = [bound(V, Vp, cfg) for Vp in VP_VALUES]
    for r in rows:
        print(f"Vp {r['Vp_V']:5.1f} V: Te bound {r['te_bound_eV']:.4f} eV ({r['te_bound_percent']:.1f}%)  "
              f"with Vp known {r['te_bound_vp_known_eV']:.4f} eV  inflation {r['inflation_from_vp']:.2f}  "
              f"corr(Te, Vp) {r['corr_te_vp']:+.2f}")
    path = out(cfg, "tables", "fisher_bound.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
