#!/usr/bin/env python3
"""
Step 3 — Train the physics-informed inversion network on the synthetic set.

Outputs:
  outputs/checkpoints/pinn.pt        trained weights
  outputs/tables/train_history.json  loss curves (for figure F3)
"""
import _bootstrap as B
import json
import numpy as np
from rambhalp.config import load_config, out
from rambhalp import train


def main():
    cfg = load_config()
    print("=" * 70)
    print("STEP 3: train physics-informed network")
    print("=" * 70)
    cd = B.cache_dir(cfg)
    tr = np.load(cd / "synth_train.npz"); va = np.load(cd / "synth_val.npz")

    model, hist = train.train_model(tr["X"], tr["Y"], cfg,
                                    X_val=va["X"], Y_val=va["Y"], verbose=True)
    train.save_checkpoint(model, cfg, out(cfg, "checkpoints", "pinn.pt"),
                          extra={"kind": "pinn", "history": hist})
    with open(out(cfg, "tables", "train_history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"wrote {out(cfg,'checkpoints','pinn.pt')}")


if __name__ == "__main__":
    main()
