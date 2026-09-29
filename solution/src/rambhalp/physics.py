"""
physics.py — the OML (Orbit-Motion-Limited) forward model and the physics-informed loss.

This is the physical heart of the project. ONE forward model I(V; Ne, Te, Vp) is used in three
places, so they can never drift apart:
  * synthetic.py   uses it to GENERATE labelled sweeps
  * train.py       uses it as the physics-reconstruction term of the loss
  * infer/validate use it to check a prediction reconstructs the observed curve

------------------------------------------------------------------------------------------
The model (spherical probe, canonical electron-positive convention)
------------------------------------------------------------------------------------------
Electron thermal current magnitude at the plasma potential Vp:

    I_e0 = q_e * Ne * A * sqrt( q_e * Te / (2*pi*m_e) )        [Amps]

    where A = 4*pi*r^2 is the sphere surface area, Te is in eV, Ne in m^-3.

Electron current vs bias V (relative coordinate x = (V - Vp)/Te):

    V <  Vp  (electron-retarding) :  I_e = I_e0 * exp(x)              (Boltzmann)
    V >= Vp  (electron-saturation):  I_e = I_e0 * (1 + x)**alpha      (sheath expansion)

    alpha = probe.sheath_exponent in config.yaml. 1.0 is the textbook OML sphere (linear
    growth); it was hard-coded here until 2026-09-03. Fitting alpha freely against real
    RAMBHA sweeps gives 0.40 (median) with R^2 0.998, against 0.972 at alpha = 1.0 -- the
    real saturation branch is concave, not linear. See config.yaml for the full comparison.

Total probe current with a small ion floor I_i0 = ion_frac * I_e0 (creates a floating
potential where the curve crosses zero, exactly as the classical method looks for):

    I(V) = I_e(V) - I_i0

Everything downstream feeds current in amps scaled by a FIXED constant (I_SCALE), so the
absolute amplitude — which is what carries Ne — is preserved (we do NOT per-sweep normalise
amplitude away).
"""
from __future__ import annotations
import numpy as np

# Fixed normalisers (same for every sweep, so amplitude information survives).
I_SCALE = 1.0e-7     # amps -> ~O(1) network units (100 nA)
V_SCALE = 12.0       # volts -> [-1, 1]


def probe_area(radius_m: float) -> float:
    """Sphere surface area A = 4*pi*r^2 (m^2)."""
    return 4.0 * np.pi * radius_m ** 2


def electron_sat_current(Ne_m3, Te_eV, cfg) -> float:
    """I_e0 (amps): electron thermal current at the plasma potential."""
    p = cfg["probe"]
    A = probe_area(p["radius_m"])
    return p["q_e"] * Ne_m3 * A * np.sqrt(p["q_e"] * Te_eV / (2.0 * np.pi * p["m_e"]))


# --------------------------------------------------------------------------------------
#  Canonical curve = ELECTRON current with the ion/offset baseline at zero.
#
#  Why electron-only: the measured sweep carries an arbitrary additive baseline (ion
#  saturation + electronics offset) that varies by gain channel. We subtract that baseline
#  from every real sweep (infer.prepare_input) so the curve runs from ~0 (deep retardation)
#  up to the electron branch. The synthetic generator produces the SAME zero-baselined
#  electron curve, so training and real data live in one representation. Verified on real
#  RAMBHA sweeps: the OML electron model reconstructs them with median R^2 ~0.82.
# --------------------------------------------------------------------------------------
def forward_iv_np(V, Ne_cc, Te_eV, Vp_V, cfg):
    """Electron current I_e(V) in AMPS (ion baseline = 0).

    Ne_cc : electron density in cm^-3 (converted to m^-3 internally, *1e6).
    V     : array of bias voltages (V).
    """
    Ne_m3 = Ne_cc * 1e6
    Ie0 = electron_sat_current(Ne_m3, Te_eV, cfg)
    x = (V - Vp_V) / Te_eV
    # Cap the saturation branch the same way the retardation branch is capped (-60..0).
    # Unclamped, x = (V-Vp)/Te can reach the HUNDREDS for the low-Te end of the sampled
    # range (Te down to 0.05 eV) combined with wide Vp -- this is not a rare edge case,
    # it occurs in essentially every training batch. That blows a plain, unnormalised MLP's
    # activations to inf/nan within the first few batches (the observed epoch-0 NaN).
    # Capping at 60 keeps both branches on a comparable numerically-safe scale and does not
    # touch the physics anywhere real lunar sweeps live (|x| is well under 60 there).
    alpha = float(cfg["probe"].get("sheath_exponent", 1.0))
    retard = np.exp(np.clip(x, -60, 0))
    sat = (1.0 + np.clip(x, 0, 60)) ** alpha
    k = transition_width_te(cfg)
    if k > 0.0:
        # Blend in LOG space, not linear. Blending the currents directly is wrong and
        # actively dangerous here: deep in the retarding region the saturation branch is
        # clipped to (1+0)**alpha = 1, so a linear blend contributes w*1 there. With
        # w = sigmoid(x/k) that term decays as exp(x/k) -- SLOWER than the true exp(x) -- so it
        # dominates the deep tail and manufactures a spurious second population at temperature
        # k*Te. That is precisely the artefact this whole line of work exists to rule out.
        # Blending the logarithms makes the correction vanish as w -> 0 instead.
        w = _blend_weight_np(x / k)
        log_shape = (1.0 - w) * np.clip(x, -60, 0) + w * (alpha * np.log1p(np.clip(x, 0, 60)))
        shape = np.exp(log_shape)
    else:
        shape = np.where(V < Vp_V, retard, sat)
    return Ie0 * shape


def forward_iv_two_np(V, Ne_c, Te_c, Ne_h, Te_h, Vp_V, cfg):
    """Electron current in AMPS from TWO Maxwellian populations sharing one plasma potential.

    The RAMBHA-LP sweeps are not described by a single Maxwellian: the local temperature
    d(ln Ie)/dV varies by a factor of five across the retarding region (measured over 1,403
    sweeps), which is the signature of a cold majority plus a hot minority. Resolving the two
    moves the recovered temperature from 3903 K onto the published 2573 K. See twopop.py.

    Both populations are collected by the same probe through the same sheath, so each uses the
    same forward model and the same sheath exponent; the currents simply add.
    """
    return (forward_iv_np(V, Ne_c, Te_c, Vp_V, cfg)
            + forward_iv_np(V, Ne_h, Te_h, Vp_V, cfg))


# --------------------------------------------------------------------------------------
#  Retardation -> saturation transition
# --------------------------------------------------------------------------------------
# The model below is piecewise: exp((V-Vp)/Te) for V < Vp, then (1 + (V-Vp)/Te)**alpha for
# V >= Vp. Those two branches have DIFFERENT SLOPES at V = Vp, so the curve has a kink there.
# Real sweeps do not: electrons arrive with a spread of energies, so there is no single bias at
# which retardation stops and saturation begins -- the changeover happens over roughly a thermal
# width.
#
# That kink is not cosmetic. The matched-control diagnostic (scripts/11_local_te_diagnostic.py)
# found the real stacked curves flatter than a stack of single Maxwellians in the 0.6 V below Vp,
# reproducibly on NINE of nine observation days (sign test p = 0.002). Blending the two branches
# over a width of ~1.3*Te removes that excess entirely: +0.18 and +0.24 eV at u = -0.6 and -0.3
# become +0.00 and -0.02. Varying the sheath exponent over 0.40 / 0.70 / 1.00 changes it by
# nothing at all, because alpha acts only above Vp.
#
# Width 0 reproduces the sharp model exactly, so this is off unless probe.transition_width_te
# is set.
def _blend_weight_np(x_over_width):
    import numpy as _np
    return 1.0 / (1.0 + _np.exp(-_np.clip(x_over_width, -60.0, 60.0)))


def transition_width_te(cfg) -> float:
    """Changeover width at Vp, as a multiple of Te. 0 disables the smoothing."""
    return float(cfg["probe"].get("transition_width_te", 0.0))


def to_input_np(I_amps):
    """Scale a current curve (amps) into fixed network input units."""
    return I_amps / I_SCALE


# --------------------------------------------------------------------------------------
#  Torch forward model (used inside the training loss so it is differentiable)
# --------------------------------------------------------------------------------------
def forward_iv_torch(V, Ne_cc, Te_eV, Vp_V, cfg):
    """Differentiable OML electron forward model. V is (1,G); params are (B,1).

    Returns electron current in fixed input units (divided by I_SCALE), shape (B, G).
    """
    import torch
    p = cfg["probe"]
    A = 4.0 * torch.pi * p["radius_m"] ** 2
    Ne_m3 = Ne_cc * 1e6
    Ie0 = p["q_e"] * Ne_m3 * A * torch.sqrt(p["q_e"] * Te_eV / (2.0 * torch.pi * p["m_e"]))
    x = (V - Vp_V) / Te_eV                              # (B, G)
    retard = torch.exp(torch.clamp(x, min=-60.0, max=0.0))
    # Symmetric cap on the saturation branch -- see forward_iv_np for why. Without max=60.0
    # here, an untrained (or destabilised) model predicting an extreme Vp/Te combination can
    # push x arbitrarily high, overflowing 1+x to inf and poisoning the whole loss to NaN.
    alpha = float(p.get("sheath_exponent", 1.0))
    sat = (1.0 + torch.clamp(x, min=0.0, max=60.0)) ** alpha
    k = float(p.get("transition_width_te", 0.0))
    if k > 0.0:
        # Log-space blend -- see forward_iv_np for why a linear blend fabricates a slow tail.
        w = torch.sigmoid(torch.clamp(x / k, min=-60.0, max=60.0))
        log_shape = ((1.0 - w) * torch.clamp(x, min=-60.0, max=0.0)
                     + w * alpha * torch.log1p(torch.clamp(x, min=0.0, max=60.0)))
        Ie = Ie0 * torch.exp(log_shape)
    else:
        Ie = Ie0 * torch.where(V < Vp_V, retard, sat)
    return Ie / I_SCALE


def forward_iv_two_torch(V, Ne_c, Te_c, Ne_h, Te_h, Vp_V, cfg):
    """Differentiable two-population forward model. V is (1,G); params are (B,1).

    Returns total electron current in fixed input units, shape (B, G). Used as the physics
    term of the loss when the network predicts both populations.
    """
    return (forward_iv_torch(V, Ne_c, Te_c, Vp_V, cfg)
            + forward_iv_torch(V, Ne_h, Te_h, Vp_V, cfg))


def physics_residual_torch(curve_in, Ne_cc, Te_eV, Vp_V, V_grid, cfg, ref_ms=None):
    """Relative (self-normalised) reconstruction error between an observed (zero-baselined
    electron) curve and the OML forward model of the predicted parameters. This is the
    'physics-informed' loss term: it rewards predictions that actually reproduce the measured
    sweep.

    Normalised by a mean-square-current reference rather than left as raw MSE in current-input
    units, so w_phys in config.yaml actually means what it says (see the module-level note in
    train.py's history for why raw MSE here silently dominated the total loss by >10x).

    ref_ms: optional precomputed mean-square reference. train_model computes this ONCE over
    the whole training set and passes it in, rather than recomputing per-batch: per-batch
    normalisation makes the loss's effective scale jump around with whatever mix of high/low-
    Ne (hence high/low-amplitude) curves a given batch happens to sample -- visible in
    practice as occasional large loss spikes mid-training. A single fixed reference removes
    that batch-to-batch jitter while keeping the same scale-matching benefit. Falls back to
    per-batch mean(curve_in**2) when not supplied, for standalone use (e.g. tests) where a
    dataset-wide reference isn't available."""
    import torch
    recon = forward_iv_torch(V_grid, Ne_cc, Te_eV, Vp_V, cfg)
    denom = ref_ms if ref_ms is not None else (torch.mean(curve_in ** 2) + 1e-6)
    return torch.mean((recon - curve_in) ** 2) / denom
