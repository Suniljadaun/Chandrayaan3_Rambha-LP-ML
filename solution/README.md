# Physics-Informed ML Inversion of Chandrayaan-3 Langmuir Probe Sweeps

Recovering electron density, electron temperature and plasma potential from the RAMBHA-LP
instrument on the Chandrayaan-3 lunar lander — including the 44% of sweeps that classical
curve-fitting cannot handle.

**Sunil Kumar** · M.Tech Independent Study, IIIT Hyderabad · Supervisor: Dr. Rama Chandra Pillutla

---

## What this does

RAMBHA-LP is a Langmuir probe: a sphere on a boom above the lunar surface whose voltage is swept
while the collected current is measured. The shape of that current curve encodes three plasma
parameters. The standard way to extract them is to fit straight lines to parts of the curve —
which works on clean sweeps and fails on messy ones.

This project trains a neural network on synthetic sweeps generated from an orbital-motion-limited
forward model, adds a physics-reconstruction term to the loss, and applies it to the real archive.
It is validated three ways, none of which requires ground truth (there is none for lunar plasma).

## Results

**Coverage.** 29,862 sweeps parsed from 144 PDS4 files.

| | sweeps | share |
|---|---|---|
| classical fit succeeds | 16,596 | 55.6% |
| …of which clipped (not usable) | 1,449 | |
| classical fails | 13,266 | 44.4% |
| …clipped by the instrument (not recoverable) | 7,625 | 57.5% of failures |
| …recovered by the network | 5,115 | 38.6% of failures (91% of unclipped ones) |
| **combined usable** | **20,262** | **67.9%** |

**Clipping.** 30% of all sweeps are clipped: the front end saturates and the top of the curve goes
flat (73% of ADC channel 2's 1 MΩ sweeps, all of its 20 MΩ sweeps, 9% of channel 1's 20 MΩ sweeps).
Neither method can read a clipped curve, so clipped sweeps are excluded from both. Clipping real
sweeps on purpose shows why: a flat top from about +3 V cuts the network's Te to about a third and
raises its density by about 10% (`scripts/18_recovered_validation.py`).

**Accuracy** on 4,000 held-out synthetic sweeps: density 0.021 dex (~5%), temperature 0.078 eV
(~20% at the archive's typical 0.4 eV), plasma potential 0.141 V.

**Validation without ground truth.**

- *Independent method.* On the 15,147 unclipped sweeps both methods fit there is no meaningful
  density offset (1.4%) and individual sweeps differ by 4.4%. Temperature differs by 0.091 eV (25%)
  per sweep, with little offset.
- *Recovered sweeps.* For 2,157 recovered sweeps the other half of the same commanded sweep passes
  the classical fit unclipped; against it the recovered density differs by 5.3%, the same level as
  where both halves pass (4.7%).
- *Instrument against itself.* Every commanded sweep is a triangle, so its two halves are
  independent measurements seconds apart: 9,895 pairs agree to 0.010 dex (2.4%), r = 0.992,
  with no directional bias.
- *Mission timeline.* The network never sees the date, yet its daily medians rise from 246 to
  500 cm⁻³ (Spearman 0.78, p = 0.008). The classical fit shows the same rise, so this is a
  consistency check rather than an independent test.

**Against the published analysis** (Manju et al. 2025, MNRAS 542, 2647): median temperature and
plasma potential, and daily median density from 27 August onwards, fall inside the published
ranges (density on the first three days is lower); for 2 September, 19–20 h, the network gives a mean of 431 cm⁻³ and the classical fit
432 cm⁻³, inside the published 409–680 cm⁻³ (mean 478, Manju et al. 2026), and a plasma potential of
−3.8 V (classical fit −3.9 V) against −3.9 V.

## What is novel here

**A two-population Langmuir inversion with a working negative control.** Every ML Langmuir-probe
paper surveyed assumes a single Maxwellian electron distribution, and no published inversion of a
*planetary* Langmuir probe uses machine learning at all. This one predicts five parameters (cold
and hot density and temperature, plus plasma potential) and — the part that makes it meaningful —
declines to report a second population when none exists (1.7% false rate, verified).

**Applied to this archive it returns a negative result**, in agreement with the published
analysis. Getting there produced two findings of wider use:

- *The standard diagnostic for multiple electron populations is not valid evidence.* Averaging
  sweeps that each have a different temperature bends the averaged curve exactly like a second
  population would. On this archive the hottest 10% of sweeps supply 65% of the signal in the
  region where the inference is made. A matched single-Maxwellian control reproduces the entire
  apparent signal.
- *A synthetic hold-out cannot adjudicate a sim-to-real question.* Demonstrated twice, in
  opposite directions. The physics-loss term looks actively harmful on synthetic data (removing
  it improves temperature error 0.075 → 0.056 eV over three runs) while on real sweeps it cuts
  the disagreement between sweep halves by 30% and removes a 5% density offset against the
  classical fit. Conversely, a temperature transformation that nearly halves synthetic error
  (22.3% → 12.2%, about 13σ over three seeds) degrades real repeatability by 69% and was rejected.

**A quantified account of why temperature is the hard parameter.** A Fisher-information analysis
gives corr(Te, Vp) = +0.92 — temperature and plasma potential are nearly indistinguishable in
this measurement — inflating the achievable error about 2.5× over the bound with the plasma potential known. On this
instrument the temperature signal and the forward model's error occupy the same region of the
curve and cannot be separated by choosing a fit window.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# point config.yaml -> data.archive_root at your RAMBHA-LP PDS4 archive
python scripts/run_all.py --no-maven        # full pipeline -> outputs/
python -m pytest tests/ -q                  # sanity tests
```

CPU is sufficient; training takes a few minutes. `GUIDE.md` walks through it step by step.

## Layout

```
src/rambhalp/
  io_pds.py preprocess.py     PDS4 parsing, sweep segmentation, binning
  classical.py                classical OML fit + failure classification
  physics.py synthetic.py     forward model, labelled synthetic sweep generator
  model.py train.py infer.py  network, training loop, application to real sweeps
  twopop.py                   classical two-population decomposition
  te_refine.py                plasma-potential-aligned temperature head (evaluated, not adopted)
  validate.py figures.py      validation checks and report figures

scripts/
  01-07   the pipeline: baseline -> synthetic -> train -> apply -> validate -> figures
  08      physics-loss ablation, scored on synthetic AND real-data metrics
  09      is the second population plasma, or photoelectrons?
  10      second population vs baseline artefact: model comparison
  11      local-temperature diagnostic WITH its matched single-Maxwellian control
  12      per-sweep slope-curvature test (no averaging, selection-matched control)
  13      temperature / plasma-potential decoupling experiment
  14      does the temperature refinement survive real sweeps?
  15      are the MC-dropout error bars honest? (synthetic truth + real up/down pairs)
  16      what is the Fisher-information bound on Te, and how much does Vp inflate it?
  17      does the data alone choose the sheath exponent 0.40? fit-window sensitivity
  18      are the recovered sweeps reliable? twin halves, classical test, clipping
```

Every diagnostic a conclusion depends on is a numbered script and re-runs with one command.
`CODE_REVIEW.md` is the full engineering log, including the claims that were withdrawn and why.

## Honest limitations

- **No absolute calibration is possible.** There is no second instrument on the lander and no
  lunar equivalent of ground radar. Every number here is consistency, not verified accuracy.
  For scale: Swarm at Earth discovered a 400 K temperature bias only because ground radar
  existed to reveal it.
- The sheath-expansion exponent is held at a single fitted value; two independent studies report
  it varies per sweep and that a wrong fixed value can bias density by a factor of three.
- Photoemission from the probe and lander is absent from the forward model.
- A channel-dependent systematic distorts the curve near the plasma potential: the two probe
  channels disagree at 11σ while returning nearly identical density and temperature. Unexplained.
- Per-sweep temperature scatter exceeds the noise model by a factor of ~3.6, so single-sweep
  temperatures are far less repeatable than a synthetic accuracy figure implies.
- MAVEN-to-Moon transfer learning was implemented and abandoned: it degraded held-out performance
  ~15×. Reported as a negative result about this particular attempt, not a general claim.

## Data

RAMBHA-LP PDS4 archive, ISRO Science Data Archive (PRADAN). Not redistributed here; set
`data.archive_root` in `config.yaml` to your local copy.
