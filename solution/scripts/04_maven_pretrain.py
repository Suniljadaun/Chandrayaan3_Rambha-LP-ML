#!/usr/bin/env python3
"""
Step 4 (optional) — MAVEN cross-planet transfer.

Pretrain on Mars-like (denser) sweeps, then fine-tune on the lunar synthetic set starting
from the pretrained weights. Saves a transfer checkpoint that Step 5 will prefer if present.

Honors the Day-32 gate: if maven.enabled is false, this step no-ops cleanly and the lunar-only
model is used downstream.

Outputs:
  outputs/checkpoints/maven_transfer.pt
  outputs/tables/maven_mode.json      records whether real/fallback data was used
"""
import _bootstrap as B
import json
import numpy as np
from rambhalp.config import load_config, out
from rambhalp import train, maven


def main():
    cfg = load_config()
    print("=" * 70)
    print("STEP 4: MAVEN pretrain + transfer")
    print("=" * 70)
    if not cfg["maven"]["enabled"]:
        print("maven.enabled = false -> skipping (frozen as future work).")
        return

    Xpt, Ypt, mode = maven.get_pretrain_data(cfg, verbose=True)
    print(f"[maven] pretrain data mode: {mode}")

    # 1) pretrain on Mars-like data
    pre_model, _ = train.train_model(
        Xpt, Ypt, cfg, epochs=cfg["maven"]["pretrain_epochs"], verbose=True)
    pre_state = {k: v.clone() for k, v in pre_model.state_dict().items()}

    # 2) fine-tune on lunar synthetic, starting from the pretrained weights
    cd = B.cache_dir(cfg)
    tr = np.load(cd / "synth_train.npz"); va = np.load(cd / "synth_val.npz")
    print("[maven] fine-tuning on lunar synthetic from pretrained weights ...")
    model, hist = train.train_model(tr["X"], tr["Y"], cfg, X_val=va["X"], Y_val=va["Y"],
                                    init_state=pre_state, verbose=True)

    train.save_checkpoint(model, cfg, out(cfg, "checkpoints", "maven_transfer.pt"),
                          extra={"kind": "maven_transfer", "mode": mode, "history": hist})
    with open(out(cfg, "tables", "maven_mode.json"), "w") as f:
        json.dump({"mode": mode, "final_val": hist["val"][-1] if hist["val"] else None}, f,
                  indent=2)
    print(f"wrote {out(cfg,'checkpoints','maven_transfer.pt')}  (mode={mode})")


if __name__ == "__main__":
    main()
