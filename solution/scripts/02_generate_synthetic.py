#!/usr/bin/env python3
"""
Step 2 — Generate the labelled synthetic training set from the OML forward model.

Outputs:
  outputs/cache/synth_train.npz  (X, Y)
  outputs/cache/synth_val.npz    (X, Y)
"""
import _bootstrap as B
import numpy as np
from rambhalp.config import load_config
from rambhalp import synthetic


def main():
    cfg = load_config()
    print("=" * 70)
    print("STEP 2: synthetic sweep generation (OML forward model)")
    print("=" * 70)
    scfg = cfg["synthetic"]

    Xtr, Ytr, _ = synthetic.make_dataset(scfg["n_train"], cfg, seed=scfg["seed"])
    Xva, Yva, _ = synthetic.make_dataset(scfg["n_val"], cfg, seed=scfg["seed"] + 1)

    cd = B.cache_dir(cfg)
    np.savez_compressed(cd / "synth_train.npz", X=Xtr, Y=Ytr)
    np.savez_compressed(cd / "synth_val.npz", X=Xva, Y=Yva)

    print(f"train: X{Xtr.shape} Y{Ytr.shape}   val: X{Xva.shape}")
    names = (["log10Nc", "Tc", "log10Nh", "Th", "Vp"] if Ytr.shape[1] == 5
             else ["log10Ne", "Te", "Vp"])
    print("label ranges  " + "  ".join(
        f"{nm}[{Ytr[:, j].min():.2f},{Ytr[:, j].max():.2f}]" for j, nm in enumerate(names)))
    if Ytr.shape[1] == 5:
        from rambhalp.synthetic import _HOT_FLOOR
        n_none = int((Ytr[:, 2] <= np.log10(_HOT_FLOOR) + 1e-6).sum())
        print(f"single-population sweeps in training set: {n_none} "
              f"({n_none / Ytr.shape[0]:.1%})  <- negative examples")
    print(f"wrote {cd/'synth_train.npz'} and synth_val.npz")


if __name__ == "__main__":
    main()
