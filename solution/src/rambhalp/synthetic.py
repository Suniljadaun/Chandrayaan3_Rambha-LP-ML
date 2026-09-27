"""
synthetic.py — generate labelled OML sweeps to train the inversion network.

Because there is NO ground truth for lunar plasma, we teach the network on physics we DO
trust: sweeps produced by the OML forward model with known (Ne, Te, Vp). We sample plausible
parameters, build the curve on the common voltage grid, add a realistic ion floor and noise,
and hand back (curve, labels). The network learns the inverse map curve -> (Ne, Te, Vp).

Labels are stored as [log10(Ne_cc), Te_eV, Vp_V]; log-density because Ne spans decades and a
network regresses a log target far more stably.
"""
from __future__ import annotations
import numpy as np

from . import physics
from .preprocess import common_grid

# Additive floor for the hot-population density label, in cm^-3. Keeps log10 finite when the
# sampled hot fraction is zero, at a level two orders below anything measured in the archive.
_HOT_FLOOR = 0.1


def sample_params(n: int, cfg: dict, rng: np.random.Generator, regime: str = "lunar"):
    """Draw n parameter sets. regime='lunar' uses synthetic ranges; 'maven' uses the
    denser Mars ranges (for MAVEN-like pretraining)."""
    scfg = cfg["synthetic"] if regime == "lunar" else cfg["maven"]
    ne_lo, ne_hi = scfg["ne_cc"]
    te_lo, te_hi = scfg["te_eV"]
    # log-uniform density, uniform temperature
    Ne = 10 ** rng.uniform(np.log10(ne_lo), np.log10(ne_hi), n)
    Te = rng.uniform(te_lo, te_hi, n)
    vp_lo, vp_hi = cfg["synthetic"]["vp_V"]
    Vp = rng.uniform(vp_lo, vp_hi, n)
    return Ne, Te, Vp


def sample_hot(n: int, Ne_cold, cfg: dict, rng: np.random.Generator):
    """Draw the hot minority population: its temperature, and its share of the total density.

    Parameterised by FRACTION rather than absolute density, because that is what the data
    constrains — the archive gives a hot component at 4-6% of the total. The range includes
    zero so the network also sees genuinely single-population curves and can learn to report
    no hot component when there is none; a model that always splits the plasma in two would be
    useless as evidence that two populations exist.
    """
    scfg = cfg["synthetic"]
    th_lo, th_hi = scfg.get("te_hot_eV", [0.8, 3.0])
    f_lo, f_hi = scfg.get("hot_fraction", [0.0, 0.15])
    absent = float(scfg.get("hot_absent_frac", 0.35))

    Te_h = 10 ** rng.uniform(np.log10(th_lo), np.log10(th_hi), n)
    frac = rng.uniform(f_lo, f_hi, n)

    # Spike-and-slab, not a plain uniform. A uniform fraction on [0, 0.15] puts essentially
    # zero probability mass on "no hot population at all", so the network never sees a
    # decisive negative example. Trained under a squared-error loss it then predicts the
    # conditional mean, and whenever the hot signature sits below the noise floor that mean
    # collapses onto the prior mean -- about 7.5% for this range. That is exactly the failure
    # observed: 8.9% hot density reported on curves built with none.
    #
    # Putting an explicit atom at zero fixes it by making "none" a represented outcome rather
    # than a measure-zero edge of a continuum. It is also the physically honest prior: the
    # archive spans sunlit and terminator geometry, and a photoelectron population is not
    # present in every sweep.
    if absent > 0.0:
        frac = np.where(rng.random(n) < absent, 0.0, frac)

    Ne_h = Ne_cold * frac / np.maximum(1.0 - frac, 1e-6)
    return Ne_h, Te_h


def _order_populations(Te_cold, Ne_hot, Te_hot):
    """Force the hot population to be the hotter one.

    Without this the two label slots are interchangeable wherever the sampled ranges overlap,
    and the supervised loss becomes ambiguous — the network is punished for choosing the other,
    equally correct, assignment. Ordering them removes the degeneracy; the decoder in train.py
    enforces the same ordering so the two cannot disagree.
    """
    return np.maximum(Te_hot, Te_cold * 1.5 + 0.05)


def make_dataset(n: int, cfg: dict, seed: int, regime: str = "lunar"):
    """Return (X, Y) where X is (n, G) scaled zero-baselined electron curves and Y is
    (n, 3) labels [log10(Ne), Te, Vp]. Also returns the raw param arrays for bookkeeping.

    Realism: we add gaussian measurement noise (fraction of the current span) AND a small
    random ion-like tilt so the network is robust to the residual left after baseline
    subtraction on real sweeps."""
    rng = np.random.default_rng(seed)
    grid = common_grid(cfg)
    Ne, Te, Vp = sample_params(n, cfg, rng, regime)
    noise_lo, noise_hi = cfg["synthetic"]["noise_frac"]
    chans = cfg["synthetic"].get("noise_frac_by_channel")
    ion_lo, ion_hi = cfg["synthetic"]["ion_current_frac"]

    two_pop = bool(cfg["synthetic"].get("two_population", False))
    if two_pop:
        Ne_h, Te_h = sample_hot(n, Ne, cfg, rng)
        Te_h = _order_populations(Te, Ne_h, Te_h)
    else:
        Ne_h = Te_h = None

    X = np.empty((n, grid.size), dtype=np.float32)
    for i in range(n):
        if two_pop:
            I = physics.forward_iv_two_np(grid, Ne[i], Te[i], Ne_h[i], Te_h[i], Vp[i], cfg)
        else:
            I = physics.forward_iv_np(grid, Ne[i], Te[i], Vp[i], cfg)      # electron amps
        span = np.ptp(I) + 1e-30
        # residual ion-like linear tilt (small; mimics imperfect baseline removal)
        # Magnitude log-uniform over the configured range, sign random: the measured residual
        # tilt on real sweeps spans two orders of magnitude (median 0.012 of span, p95 0.30),
        # so a single uniform scale misrepresents it badly at both ends.
        mag = 10 ** rng.uniform(np.log10(ion_lo), np.log10(ion_hi))
        tilt = rng.choice([-1.0, 1.0]) * mag * span * (grid - grid.min()) / np.ptp(grid)
        # Log-uniform, not uniform: the measured noise spans a factor of ~33 across the three
        # probe-resistance settings (0.002 at 49.9 kohm and 1 Mohm, 0.018 at 20 Mohm), so a
        # uniform draw would put almost all training curves in the noisy regime that only a
        # third of the archive actually occupies.
        if chans:
            lo, hi = chans[rng.integers(len(chans))]
        else:
            lo, hi = noise_lo, noise_hi
        nf = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
        I = I + tilt + rng.normal(0.0, nf * span, grid.size)
        X[i] = physics.to_input_np(I).astype(np.float32)

    if two_pop:
        # Hot density as log10 of a FLOOR-SHIFTED value, so a zero hot population is a finite
        # label rather than -inf. The floor is well below anything the archive shows.
        Y = np.stack([np.log10(Ne), Te, np.log10(Ne_h + _HOT_FLOOR), Te_h, Vp],
                     axis=1).astype(np.float32)
        return X, Y, dict(Ne=Ne, Te=Te, Ne_h=Ne_h, Te_h=Te_h, Vp=Vp)
    Y = np.stack([np.log10(Ne), Te, Vp], axis=1).astype(np.float32)
    return X, Y, dict(Ne=Ne, Te=Te, Vp=Vp)
