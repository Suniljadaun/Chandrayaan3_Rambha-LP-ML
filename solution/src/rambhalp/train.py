"""
train.py — train the inversion network with a physics-informed loss.

Loss = w_sup * supervised(label MSE) + w_phys * physics(reconstruction MSE)

  * supervised term: standard regression against the known synthetic labels.
  * physics term: take the network's predicted (Ne, Te, Vp), push them BACK through the OML
    forward model, and require the reconstructed curve to match the input curve. This is what
    makes the model 'physics-informed' — it is rewarded for predictions that are physically
    self-consistent, not merely close to a label. The physics term also has bite on REAL data,
    where no label exists (used in fine-tuning / self-consistency checks).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from . import physics
from .preprocess import common_grid


def _device(cfg):
    import torch
    d = cfg["train"]["device"]
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def _params_from_output(out, cfg=None):
    """Split network output (B,3) into (Ne_cc, Te_eV, Vp) tensors with valid ranges.
    Ne from log10 via 10**; Te forced positive with softplus to keep the forward model sane."""
    import torch
    import torch.nn.functional as F
    if out.shape[1] == 5:
        return _params_from_output_two(out, cfg)
    logNe, Te_raw, Vp = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    # Clamp BEFORE exponentiating. 10**logNe is an unbounded exponential map; an early,
    # unstable gradient step (or a training run destabilised for any other reason) can push
    # logNe far outside the label range and overflow to inf in float32, which then poisons
    # the loss to NaN for the rest of training with no way back. The clamp range is generous
    # (log10 Ne in [-2, 6] covers cm^-3 from 0.01 to 1e6, well beyond both the lunar synthetic
    # range [50, 5000] and the MAVEN range up to 1e5) so it never binds during normal training
    # -- it only stops a genuine blow-up. Same reasoning for Vp: real/synthetic Vp never
    # exceeds +-6 V, so +-30 V is pure safety margin, not a physics constraint.
    logNe = torch.clamp(logNe, -2.0, 6.0)
    Vp = torch.clamp(Vp, -30.0, 30.0)
    Ne_cc = torch.pow(10.0, logNe)
    Te_eV = F.softplus(Te_raw) + 1e-3
    return Ne_cc, Te_eV, Vp


def _params_from_output_two(out, cfg):
    """Split a five-wide output into (Nc, Tc, Nh, Th, Vp).

    Two details keep this consistent with how the labels are built in synthetic.py:

      * the hot density is carried as log10(Nh + floor) so that a zero hot population is a
        finite label rather than -inf; the floor is undone here;
      * the hot temperature is produced as an INCREMENT above the cold one, Th = 1.5*Tc + 0.05
        + softplus(raw), rather than independently. Without that ordering the two population
        slots are interchangeable wherever their ranges overlap, and the supervised loss
        punishes the network for picking the other, equally correct, assignment.
    """
    import torch
    import torch.nn.functional as F
    from .synthetic import _HOT_FLOOR
    logNc, Tc_raw, logNh, Th_raw, Vp = (out[:, 0:1], out[:, 1:2], out[:, 2:3],
                                        out[:, 3:4], out[:, 4:5])
    logNc = torch.clamp(logNc, -2.0, 6.0)
    logNh = torch.clamp(logNh, -2.0, 6.0)
    Vp = torch.clamp(Vp, -30.0, 30.0)
    Nc = torch.pow(10.0, logNc)
    Nh = torch.clamp(torch.pow(10.0, logNh) - _HOT_FLOOR, min=0.0)
    Tc = F.softplus(Tc_raw) + 1e-3
    Th = Tc * 1.5 + 0.05 + F.softplus(Th_raw)
    return Nc, Tc, Nh, Th, Vp


def _target_scales(cfg, dev):
    """Per-target normalisation for the supervised loss: [log10(Ne), Te, Vp] have very
    different natural spans (~2, ~2, ~12 respectively from config.yaml's synthetic ranges).
    A flat mean((pred-label)**2) across all three implicitly weights whichever target has the
    widest span (Vp) far more heavily than the others, starving Te/Ne of gradient signal even
    though the loss LOOKS like it treats all three equally. Dividing each channel's error by
    its own half-span before squaring puts all three on comparable footing."""
    import torch
    ne_lo, ne_hi = cfg["synthetic"]["ne_cc"]
    te_lo, te_hi = cfg["synthetic"]["te_eV"]
    vp_lo, vp_hi = cfg["synthetic"]["vp_V"]
    if cfg["synthetic"].get("two_population", False):
        from .synthetic import _HOT_FLOOR
        f_hi = cfg["synthetic"].get("hot_fraction", [0.0, 0.15])[1]
        th_lo, th_hi = cfg["synthetic"].get("te_hot_eV", [0.8, 4.0])
        nh_hi = ne_hi * f_hi / max(1.0 - f_hi, 1e-6)
        scales = [(np.log10(ne_hi) - np.log10(ne_lo)) / 2.0,
                  (te_hi - te_lo) / 2.0,
                  (np.log10(nh_hi + _HOT_FLOOR) - np.log10(_HOT_FLOOR)) / 2.0,
                  (th_hi - th_lo) / 2.0,
                  (vp_hi - vp_lo) / 2.0]
        return torch.tensor(scales, dtype=torch.float32, device=dev).view(1, 5)
    scales = [(np.log10(ne_hi) - np.log10(ne_lo)) / 2.0,
             (te_hi - te_lo) / 2.0,
             (vp_hi - vp_lo) / 2.0]
    return torch.tensor(scales, dtype=torch.float32, device=dev).view(1, 3)


def train_model(X, Y, cfg, X_val=None, Y_val=None, init_state=None, epochs=None,
                verbose=True):
    """Train and return (model, history). If init_state is given, start from those weights
    (used for MAVEN transfer: pretrain -> load -> fine-tune)."""
    import torch
    import torch.nn as nn
    from .model import build_model

    torch.manual_seed(cfg["train"]["seed"])
    dev = _device(cfg)
    grid = torch.as_tensor(common_grid(cfg), dtype=torch.float32, device=dev).view(1, -1)
    tscale = _target_scales(cfg, dev)          # (1, 3) -- see _target_scales docstring

    model = build_model(cfg).to(dev)
    if init_state is not None:
        model.load_state_dict(init_state)

    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
    ds = torch.utils.data.TensorDataset(Xt, Yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    # Fixed (not per-batch) reference scale for the physics loss's normalisation, computed
    # ONCE over the whole training set -- see physics_residual_torch's docstring for why:
    # per-batch normalisation let the physics term's effective scale jump around with
    # whatever mix of high/low-amplitude curves a given batch happened to sample, visible as
    # occasional large loss spikes mid-training.
    ref_ms = (Xt ** 2).mean().detach() + 1e-6

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    w_sup, w_phys = cfg["model"]["w_supervised"], cfg["model"]["w_physics"]
    n_epochs = epochs or cfg["train"]["epochs"]

    # Cosine decay to zero over the run. With a constant 1e-3 the loss was still bouncing at
    # the end of training (val 0.0674, 0.0689, 0.0694, 0.0663, 0.0711, 0.0631 over the last six
    # epochs) -- the step size, not the epoch budget, was setting the floor. Annealing lets the
    # last epochs actually settle.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(epochs or cfg["train"]["epochs"]))

    history = {"loss": [], "sup": [], "phys": [], "val": [], "lr": []}
    for ep in range(n_epochs):
        model.train()
        agg = {"loss": 0.0, "sup": 0.0, "phys": 0.0, "n": 0}
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            decoded = _params_from_output(out, cfg)

            # Supervised loss on the DECODED, physically-meaningful quantities -- not the
            # raw network output. Previously `sup` compared `out` directly against `yb`
            # while `phys` used softplus(out[:,1])+1e-3 as Te: two loss terms silently
            # training the SAME output channel toward two DIFFERENT targets every step (sup
            # wanted raw ~= Te directly; phys wanted softplus(raw) ~= Te), fighting each
            # other. This is almost certainly why Te has been the worst-predicted quantity.
            # log10(Ne_cc) recovers the (clamped) logNe channel exactly since Ne_cc =
            # 10**clamp(logNe), so this changes nothing for the Ne/Vp channels (already
            # consistent) and fixes only the Te channel's split-target problem.
            if len(decoded) == 5:
                from .synthetic import _HOT_FLOOR
                Nc, Tc, Nh, Th, Vp = decoded
                pred_for_sup = torch.cat(
                    [torch.log10(Nc), Tc, torch.log10(Nh + _HOT_FLOOR), Th, Vp], dim=1)
                recon = physics.forward_iv_two_torch(grid, Nc, Tc, Nh, Th, Vp, cfg)
            else:
                Ne_cc, Te_eV, Vp = decoded
                pred_for_sup = torch.cat([torch.log10(Ne_cc), Te_eV, Vp], dim=1)
                recon = physics.forward_iv_torch(grid, Ne_cc, Te_eV, Vp, cfg)
            sup = torch.mean(((pred_for_sup - yb) / tscale) ** 2)
            phys = torch.mean((recon - xb) ** 2) / ref_ms

            loss = w_sup * sup + w_phys * phys
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = xb.size(0)
            agg["loss"] += loss.item() * bs
            agg["sup"] += sup.item() * bs
            agg["phys"] += phys.item() * bs
            agg["n"] += bs

        for k in ("loss", "sup", "phys"):
            history[k].append(agg[k] / agg["n"])
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        sched.step()

        vmsg = ""
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                xv = torch.as_tensor(X_val, dtype=torch.float32, device=dev)
                yv = torch.as_tensor(Y_val, dtype=torch.float32, device=dev)
                # Same decode-before-comparing fix applied to the printed/logged val metric,
                # so it's actually comparable epoch to epoch and to `sup` above (previously
                # this compared raw output vs labels too -- same split-target problem, and
                # made the printed 'val' number not directly interpretable as an error).
                dv = _params_from_output(model(xv), cfg)
                if len(dv) == 5:
                    from .synthetic import _HOT_FLOOR
                    Nc_v, Tc_v, Nh_v, Th_v, Vp_v = dv
                    pred_v = torch.cat([torch.log10(Nc_v), Tc_v,
                                        torch.log10(Nh_v + _HOT_FLOOR), Th_v, Vp_v], dim=1)
                else:
                    Ne_v, Te_v, Vp_v = dv
                    pred_v = torch.cat([torch.log10(Ne_v), Te_v, Vp_v], dim=1)
                val = torch.mean(((pred_v - yv) / tscale) ** 2).item()
            history["val"].append(val)
            vmsg = f" | val {val:.4f}"

        if verbose and (ep % 5 == 0 or ep == n_epochs - 1):
            print(f"  epoch {ep:3d} | loss {history['loss'][-1]:.4f} "
                  f"(sup {history['sup'][-1]:.4f}, phys {history['phys'][-1]:.4f}){vmsg}")

    return model, history


def save_checkpoint(model, cfg, path: Path, extra: dict | None = None):
    """Serialise weights + the config used, so inference is reproducible."""
    import torch
    payload = {"state_dict": model.state_dict(),
               "grid_points": cfg["preprocess"]["grid_points"],
               "hidden": cfg["model"]["hidden"],
               "dropout": cfg["model"]["dropout"]}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(cfg, path: Path):
    """Rebuild the model and load weights from a checkpoint file."""
    import torch
    from .model import build_model
    dev = _device(cfg)
    payload = torch.load(path, map_location=dev)
    model = build_model(cfg).to(dev)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
