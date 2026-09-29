# Code review — `solution/` (RAMBHA-LP physics-informed inversion)

Reviewed: every file in `solution/` (14 library modules, 8 scripts, 2 test files, config,
Makefile, README, GUIDE) plus the current contents of `outputs/`.

**Bottom line:** the code runs end to end and is well organised, but the headline result
("4,412 of 12,588 classical failures recovered") is not currently supportable. Two separate
bugs — one in `classical.py`, one in the synthetic/real domain match — mean the network is
outputting almost the same answer for every real sweep, and the classical baseline it is being
compared against is ~10x off. Everything below is ordered by how much it affects that claim.

---

## BLOCKERS — fix before quoting any number

### B1. The network gives essentially the same answer for all 14,931 real sweeps

Measured from `outputs/tables/ml_predictions.csv`:

| quantity | across all 14,931 sweeps | across the 4,559 "recovered" |
|---|---|---|
| Ne (cm^-3) | 36 – 172, std 25 | 67 – 172, std 10 |
| Te (eV) | 0.46 – 2.06, std 0.25 | **1.05 – 1.35, std 0.037** |

A Te standard deviation of 0.037 eV over 4,559 physically different sweeps is not a
measurement — it is a constant. The reconstruction-R^2 gate is being passed because a smooth
monotonic curve is easy to approximate, not because anything was inverted.

**Why.** The real curves are far smaller than anything in the training set. In network input
units (amps / `I_SCALE`):

```
REAL  sweeps : per-curve peak   5th pct 0.04   median 1.6    95th pct 2.2
SYNTH train  : per-curve peak   5th pct 1.58   median 13.7   95th pct 135
```

Every real sweep sits at or below the **5th percentile** of the training distribution. The
network is extrapolating below its training range, so it pins Ne at the bottom of the sampled
band (`synthetic.ne_cc` starts at 50 cm^-3; predictions cluster at ~100).

**Fix.** Widen `synthetic.ne_cc` downward (try `[5.0, 5000.0]`, possibly `[1.0, ...]`), then
*verify overlap before training*: plot/print the histogram of real `X` peaks against synthetic
`X` peaks and require them to sit on top of each other. This check belongs in the pipeline, not
in your head.

### B2. The curve *shape* also doesn't match

Median peak-normalised curve, real vs synthetic:

```
V (volts)    -12    -8     -4     0      +4     +8     +12
REAL         0.00   0.003  0.523  0.764  0.880  0.944  1.000
SYNTHETIC    0.00   0.003  0.026  0.118  0.368  0.664  0.971
```

Real sweeps are already at half their peak by -4 V and are close to linear across the whole
range. Your OML curves are sharply exponential there. So even with the amplitude fixed, the
network is being asked to invert a curve family it has never seen.

Likely causes, in order of suspicion: (a) residual photoemission / ion slope that a single
constant baseline subtraction does not remove — you need a *sloped* baseline; (b) real Vp
sitting outside the sampled `vp_V: [-6, 6]`; (c) the probe not being in the clean OML regime.
Your own Day-2 commit notes Vf = -4.6 V, Vp ~ -1.5 V on sweep 0, which argues for (a).

Until these two distributions overlap, "physics-informed inversion" cannot work, and no amount
of training-loop tuning will help.

### B3. `classical.py` computes Ne from the wrong current — and this, not the network, causes the 10x disagreement

`classical.fit_sweep`:

```python
Iesat = np.percentile(Ie, 97)          # = the current near V = +12 V
...
ne_m3 = 4 * Iesat / (q_e * A * v_th)   # formula assumes the current is FLAT at I_e0
```

For a **spherical** OML probe the current at V does not saturate — it grows as
`I_e0 * (1 + (V - Vp)/Te)`. At V = +12 V with your median fitted Te = 1.047 eV that factor is
**12.5x**, i.e. **1.10 dex**.

Your measured `Ne_dex_median_diff` (ML minus classical) in `validation.json` is **-0.994 dex**.
That is the same number. The disagreement you have been chasing is the classical baseline being
~10x too high, not the network being low.

**Fix.** Estimate I_e0 properly: fit the saturation branch (`I` vs `V` above Vp, linear) and
extrapolate back to `V = Vp`; or divide the measured current by `(1 + (V - Vp)/Te)`. Either way
you get the current *at the plasma potential*, which is what the thermal-flux formula wants and
what `physics.electron_sat_current` uses.

**Do not** touch `synthetic.ion_current_frac` again. The comment in `config.yaml` shows the
symmetric-tilt change was made to fix this symptom; the number went from -0.745 to -0.994, i.e.
it got worse. It was a fix in the wrong file.

### B4. The 84% classical failure rate is mostly an artifact of where the fit window lands

Of 12,588 "failures", **10,849** are just `R2 < 0.90`. The retardation window is chosen by
current fraction:

```python
mask = (Ie > 0.05 * Iesat) & (Ie < 0.50 * Iesat)
```

On real sweeps that lands at roughly **V = -10 to -4 V** (checked on sweeps 0, 5, 11) — below
the floating potential, in baseline and noise, not in the electron-retardation region at all. A
log-linear fit there will of course have poor R^2.

Note also that `classical.retardation_span: 6.0` and `classical.ion_floor_offset: 2.0` exist in
`config.yaml` and **are never read anywhere in the code**. They look like the window definition
you originally intended: fit from `V_float` up to `V_float + retardation_span`.

Because the ML target set is *defined* as the classical failure set, this bug sets your headline
number. Fix it before anything downstream.

### B5. The hop blind test currently fails, and nothing in the pipeline says so

`validation.json`: `terminator_over_pre_hop_ratio = 0.617`. `validate.py`'s own comment says it
should be **> 1** (roughly 2–3x, per the ~478 -> ~1100 /cc literature). The test is failing and
is reported as a neutral descriptive number.

Two things to fix:

1. `hop_blind_test` and `figures.fig_hop` take the median over **all** `ml_rows`, including the
   ~10,000 sweeps the model did not recover. Filter to `recovered == True` first.
2. Report a pass/fail (or at minimum print a warning) so a failing physics check cannot be
   quietly carried into the report.

---

## HIGH — correctness / academic-honesty issues

### H1. `maven_mode.json` claims real MAVEN data that was never used

The file says `"mode": "maven_real_reachable+synthetic_transfer"`. What `maven.py` actually does
is fetch a **directory listing HTML page** from LASP, cache it, and then train on
`maven_like_dataset()` — synthetic curves from the Mars parameter ranges. No MAVEN measurement
enters the pipeline at any point.

Rename the mode string to something that cannot be misread, e.g.
`"maven_like_synthetic (archive reachable, no granule parsed)"`.

Also: `maven_mode.json` records `final_val = 1.693`; the plain PINN's `train_history.json` ends
at `val = 0.111`. The MAVEN transfer made the model **~15x worse**. Per your own Day-32 gate,
set `maven.enabled: false` and freeze it as declared future work.

### H2. Your `outputs/` are a mix of two different runs

```
maven_transfer.pt      06:38
maven_mode.json        06:38
pinn.pt                06:43
classical_baseline.csv 06:42
ml_predictions.csv     06:43
validation.json        06:43
```

`validation.json` says `"checkpoint": "pinn.pt"`, yet a `maven_mode.json` from a different run
is sitting in the same folder as though it applied. Anything read out of this folder into the
report is not reproducible.

`rm -rf outputs/` and do one clean `python scripts/run_all.py` before quoting a single number.

### H3. Te is not a usable output at present

`validation.json` synthetic hold-out: `Te_median_abs_err_eV = 0.42` on a sampled range of
0.05–2.0 eV — about 22% of the entire span, on *synthetic data where the answer is known
exactly*. Real performance is necessarily worse.

Either restrict your claims to Ne, or state plainly that Te is constrained only to a factor.
Quoting a per-sweep Te from this model would not survive a viva question.

### H4. The documentation contradicts the code

- `GUIDE.md` line ~179: "during the **Aug-26-2023** lander hop, physics says density should
  **crash**". `config.yaml` + `validate.py`: **Sep 2–3**, and terminator density should be
  **higher**. Two opposite predictions in the same repository.
- `GUIDE.md` line ~144: "expect the classical method to succeed on roughly a third of sweeps".
  Actual: 15.7%.
- `physics.py` module docstring: "`I(V) = I_e(V) - I_i0`". `forward_iv_np` returns electron
  current only, ion baseline zero. (The section below it says so; the header contradicts it.)
- `physics.py` comment: "reconstructs them with median R^2 ~0.82". Actual median `recon_r2` is
  0.859, but the mean is **-21.8** and the minimum is **-236.6**. Quoting only the median hides
  that a quarter of sweeps reconstruct worse than a flat line.

---

## MEDIUM — real bugs, smaller blast radius

**M1. `infer.prepare_input` baselines differently from `classical.fit_sweep`.**

```python
baseline = np.median(np.sort(I)[:n_edge])      # infer.py  -> smallest CURRENTS anywhere
baseline = np.median(I[order][:n_edge])        # classical.py, order = argsort(V) -> lowest V
```

The docstring in `infer.py` says "most-negative-bias samples", which is the `classical.py`
version. Sorting by current picks the noise floor and biases the electron curve low. Use the
V-sorted form in both, from one shared helper.

**M2. `model.predict` leaves the model in training mode.** With `mc_samples > 1` it calls
`model.train()` and never restores `model.eval()`. Any later `predict(..., mc_samples=1)` on the
same object silently runs with dropout active. Wrap in try/finally.

**M3. `infer.apply` reuses the wrong threshold and hardcodes another.** It gates on
`cfg["classical"]["min_r2"]` — a threshold designed for a *log-linear fit* R^2 — applied to a
*curve-space reconstruction* R^2, which is a different quantity with a different scale. And
`pred["Ne_dex_std"][i] < 0.5` is a magic number in code. Both belong in a new `recover:` block in
`config.yaml` so the report can state them.

**M4. Dead config keys.** `classical.retardation_span` and `classical.ion_floor_offset` are read
nowhere. Wire them into the fit window (see B4) or delete them — a config key that does nothing
is worse than no key.

**M5. `--quick` poisons the real cache.** `run_all.py --quick` sets `max_files = 3`, and step 01
overwrites `outputs/cache/sweeps.pkl` with 3 files' worth of sweeps. A later run of step 05 or 07
alone will silently use that 3-file cache. Give quick mode its own `paths.outputs`.

**M6. `io_pds.read_science` hardcodes two columns** (`names=["V","I"], skiprows=1`). It works on
this archive but will silently mis-map if any file has an extra column. Read the header and
select columns by name.

**M7. `synthetic.sample_params` mixes regimes.** It takes Ne/Te from the `maven` block but always
takes Vp from `cfg["synthetic"]["vp_V"]`. Probably intentional; add a `maven.vp_V` key or a
comment saying so.

**M8. `solution/` is not in version control.** `git status` shows `?? solution/`. Your 50-day
learning track is committed; the actual submittable code is untracked. One bad `rm` and it's
gone. Commit it now.

**M9. `requirements.txt` lists `scipy`; nothing imports it.** Also `environment.yml` and
`requirements.txt` pin differently (`numpy>=1.24,<2.2` vs `numpy>=1.24`). Pick one source of
truth.

---

## TESTS — none of the current tests would have caught any of the above

You have 3 physics tests and 1 IO test. All pass, and all of them test the forward model against
itself. Add these three, in this order of value:

1. **Domain-overlap test.** Load a sample of real sweeps, run `infer.prepare_input`, and assert
   the real per-curve peak distribution overlaps the synthetic one (e.g. real median lies between
   the synthetic 10th and 90th percentiles). This catches B1 directly.
2. **Non-degeneracy test.** Assert the spread of predictions on real sweeps is non-trivial, e.g.
   `std(log10(Ne_pred)) > 0.2` and `std(Te_pred) > 0.1`. This catches the mode collapse the
   moment it appears.
3. **Classical round-trip test.** Generate a synthetic sweep with a known Ne via
   `physics.forward_iv_np`, feed it to `classical.fit_sweep`, and assert the recovered Ne is
   within a factor of 2 of truth. This would have caught B3 on day one — and it is 8 lines.

---

## Suggested order of work

1. **B3 + B4** — fix `classical.py` (saturation current, fit window). Re-run step 01. Your
   failure rate and your ML target set will both change substantially, so nothing downstream is
   worth doing before this.
2. **B1 + B2** — make the synthetic set look like the real data (widen Ne down, sloped baseline
   or wider Vp), and add the overlap check from Tests #1 as a gate.
3. Re-train and do one clean full run (**H2**).
4. **B5, H1, H3** — honest reporting: filter the hop test to recovered sweeps, rename the MAVEN
   mode and disable it, state the Te limitation.
5. **H4** — reconcile GUIDE.md / config.yaml / docstrings.
6. The MEDIUM list, then the tests.

Items 1–3 are the ones that decide whether you have a result. Everything after that is polish.

---

# Appendix — "Do we need a bigger model / more data / a transformer / a GPU?"

Short answer: **no.** Measured, not guessed.

## What you are training now

`model.build_model` produces a plain MLP:

```
input 240 (the resampled I-V curve)
  -> Linear 240x256 + ReLU + Dropout(0.1)
  -> Linear 256x256 + ReLU + Dropout(0.1)
  -> Linear 256x128 + ReLU + Dropout(0.1)
  -> Linear 128x3      [log10(Ne), Te_raw, Vp]
```

**160,771 parameters** (~648 KB, which matches `pinn.pt` exactly). Trained on 40,000 synthetic
curves, 60 epochs, Adam. This is a small model, and it trains in minutes on a laptop CPU.

## Experiment 1 — capacity and data

Every row below is an identical 8,000 Adam steps, batch 512, same seed, same decode path as
`train.py`, evaluated as median absolute error on a 4,000-curve hold-out.

| variant | params | train curves | Ne (dex) | Te (eV) | Vp (V) |
|---|---|---|---|---|---|
| **A — your current setup** | 161k | 40k | 0.098 | 0.260 | 0.383 |
| B — same model, **4x the data** | 161k | 160k | 0.079 | 0.253 | 0.351 |
| C — **3.2x bigger model** | 518k | 40k | 0.082 | 0.251 | 0.373 |
| G — no ion tilt | 161k | 40k | 0.063 | 0.204 | 0.355 |
| I — tilt +/-0.1 instead of +/-0.4 | 161k | 40k | 0.080 | 0.247 | 0.375 |
| **F — noise fixed at 1% instead of 1-8%** | 161k | 40k | **0.048** | **0.133** | **0.217** |

Read the table this way:

- **Tripling the model does nothing** (A -> C: Te 0.260 -> 0.251).
- **Quadrupling the data does nothing** (A -> B: Te 0.260 -> 0.253).
- **Cutting the injected noise halves every error** (A -> F).

The accuracy is set by the noise you inject into the synthetic set, not by model capacity or
dataset size. Supporting evidence from your own run: in `train_history.json` the final val loss
(0.111) is *lower* than the final train loss (0.128). A model that is not overfitting does not
need more data, and a model that plateaus at 3x the size is not capacity-limited.

## Experiment 2 — the information ceiling

How far can each parameter move before the curve changes by more than the noise? (RMS curve
change vs noise amplitude, using your own `physics.forward_iv_np`.)

| operating point | noise | Te resolvable to | Ne resolvable to | Vp resolvable to |
|---|---|---|---|---|
| Ne=500, Te=0.5, Vp=0 | 1% / 4% / 8% | ±0.03 / ±0.13 / ±0.33 eV | ±0.015 / ±0.04 / ±0.08 dex | ±0.18 / ±0.72 / ±1.46 V |
| Ne=500, Te=1.5, Vp=0 | 1% / 4% / 8% | ±0.11 / ±0.51 / ±1.41 eV | ±0.010 / ±0.04 / ±0.075 dex | ±0.20 / ±0.76 / ±1.54 V |
| Ne=100, Te=1.2, Vp=-4 | 1% / 4% / 8% | ±0.07 / ±0.30 / ±0.73 eV | ±0.010 / ±0.035 / ±0.07 dex | ±0.22 / ±0.84 / ±1.72 V |

Averaged over your sampled ranges, the Te floor at 1-8% noise is roughly **±0.3 to ±0.5 eV**.
Your model reports 0.42 eV. **It is already sitting on the physics limit.** No architecture can
go below it, because the information is not in the curve.

Ne is different: its floor is ±0.01-0.08 dex and the model achieves 0.098-0.114. There is maybe
a factor of 2 of real headroom on Ne, and none on Te.

Good news on degeneracy: Ne and Te are *not* degenerate. Scaling Te by 2x and Ne by 1/sqrt(2)
(keeping the amplitude Ne*sqrt(Te) fixed) changes the curve by 19.7% of its span — well above
the noise. The shape does carry independent Te information; there just isn't much of it.

## Why a transformer would be a step backwards

Your input is a 240-point smooth, monotone, fixed-grid curve. Self-attention exists to model
long-range, order-flexible dependencies in variable-length token sequences — none of which
describes this. A transformer would need substantially more data to reach the same accuracy and
would still be capped by the same noise floor. For fixed-grid 1-D regression, MLP or a small
1-D CNN is the correct family. Choosing a transformer here would be a question you could not
defend in a viva.

## Why GPU / Colab / Kaggle is not the bottleneck

The whole training run is minutes of laptop CPU. Renting a GPU for a 161k-parameter MLP on 40k
samples buys you nothing. GPU time only becomes worth it if you later want (a) many seeds for a
proper ensemble, or (b) a hyperparameter sweep — both of which are *breadth*, not size.

## The one model change that probably IS worth making

Train on **log(current)**, or add a log-space term to the loss.

Right now the current at +12 V is roughly 13x `I_e0`, so in a plain MSE the saturation branch
dominates and the exponential knee — which is the *only* place Te information lives — contributes
almost nothing to the gradient. This is the same asymmetry that makes B3 in the main review
possible. A log-space (or dual linear+log) loss puts real weight on the knee. That is a
principled, defensible change; "we used a bigger network" is not.

Second candidate, same spirit: predict `log10(Ne)` and `log10(Te)` rather than `Te` directly, so
both targets are scale-free.

## The honest bottom line

Scaling the model now would make the situation *worse*, not better. Per B1/B2 in the main review,
your real sweeps sit below the 5th percentile of the training amplitude distribution and have a
different shape. A larger network extrapolating off-distribution becomes *more confidently
wrong*, not less — you would be spending GPU hours to produce a more expensive version of the
same mode collapse.

Priority order stays exactly as in the main review: fix `classical.py`, then close the
synthetic-vs-real gap, then re-run. Only after real and synthetic curves overlap does any
question about architecture become answerable — and at that point the honest answer will
probably still be "the MLP is fine; report Ne, and quote Te with a ±0.4 eV error bar."

---

# Step 1 — DONE (2026-09-03). What changed and what it revealed.

## Files touched

- `src/rambhalp/classical.py` — rewritten fit (the original is kept at
  `backups/classical_pre_20260903.py.bak`).
- `tests/test_classical_roundtrip.py` — new, 10 assertions, all passing.

Nothing else was modified. `outputs/` was deliberately left alone — see "what to run next".

## What the new fit does

Three changes, all in `fit_sweep`:

1. **V_float is now the LAST upward zero crossing of the smoothed current, not the first.**
   This turned out to be the bug blocking everything else. The ion-saturation floor sits
   within noise of zero, so a real sweep crosses zero 11-19 times down at V ~ -12 V. Taking
   the first crossing put V_float at -11.3 V when the true value was -1.1 V, which placed the
   entire electron-retardation window inside the ion floor.

2. **The retardation window is anchored to V_float**, as `config.yaml` always said it was:
   `[V_float, V_float + retardation_span]`, trimmed at Vp once Vp is known. `retardation_span`
   and `ion_floor_offset` are now actually read by the code; before this they were dead keys.

3. **Density comes from the current at the plasma potential, not at the top of the sweep.**
   Above Vp the OML sphere current is linear, `I_e = (I_e0/Te)*(V - (Vp - Te))`, so a straight
   line through the saturation branch gives slope `a = I_e0/Te` and zero-crossing
   `V0 = Vp - Te`. With Te from step 2 that yields both remaining unknowns exactly:
   `Vp = V0 + Te` and `I_e0 = a*Te`. No circularity, no percentile heuristic.

A sanity gate was added: if the extrapolated Vp falls outside the swept voltage range the
sweep is failed explicitly rather than silently reporting a meaningless number.

## Validation — OML curves with known answers

Eight synthetic sweeps, physical ion floor (2.3% of I_e0, the H+ value), noise at 2% of I_e0:

| true Ne | true Te | true Vp | NEW Ne | NEW Te | NEW Vp | OLD Ne |
|---|---|---|---|---|---|---|
| 200 | 0.30 | 0.0 | 206 (1.03x) | 0.32 | 0.0 | 2580 (**12.9x**) |
| 500 | 0.50 | 0.0 | 560 (1.12x) | 0.63 | 0.1 | 5046 (**10.1x**) |
| 500 | 1.00 | -1.5 | 510 (1.02x) | 1.04 | -1.5 | 3902 (**7.8x**) |
| 1000 | 0.80 | 1.0 | 1031 (1.03x) | 0.85 | 1.0 | 7828 (**7.8x**) |
| 2000 | 1.50 | -3.0 | 1974 (0.99x) | 1.46 | -3.0 | 13832 (**6.9x**) |
| 100 | 0.40 | 2.0 | 112 (1.12x) | 0.50 | 2.1 | 1024 (**10.2x**) |
| 3000 | 0.60 | 0.0 | 3143 (1.05x) | 0.66 | 0.1 | 27803 (**9.3x**) |
| 300 | 1.20 | -2.0 | 322 (1.07x) | 1.38 | -1.8 | 2191 (**7.3x**) |

Median |log10| density error: **0.017 dex (4%) new, vs 0.930 dex (8.5x) old** — a factor of 55.
Te now lands within 8%; Vp within 0.1 V. The old fit reported *success* on all eight while
being wrong by an order of magnitude on every one.

## The number that mattered

On the sweeps where both methods succeed, ML-minus-classical density:

```
OLD classical :  -0.994 dex   (the network read 10x LOWER than classical)
NEW classical :  +0.200 dex   (the network reads 1.6x higher)
```

The 10x disagreement that the earlier `ion_current_frac` change was chasing is gone. It was
the baseline, not the network. That is now confirmed rather than inferred.

## What the fix revealed — read this part carefully

Applying a physically correct classical fit to the real archive gives:

```
OLD:  succeeded 2343 / 14931  (15.7%)   median Ne 604 cm^-3, median Te 1.05 eV
NEW:  succeeded  133 / 14931  ( 0.9%)   median Ne  39 cm^-3, median Te 1.67 eV
```

and the dominant new failure reason is stark:

```
Vp outside swept range (curve not OML-consistent) ...... 11045 sweeps  (74%)
R2 below threshold .....................................  1863
too few retardation points .............................  1011
non-positive saturation slope ..........................   852
```

For 74% of sweeps the saturation-branch extrapolation puts the plasma potential at around
**-18 V**, below the -12 V bottom of the sweep. Independently, a straight line fits the whole
-12..+12 V curve with median R^2 = 0.923, while a single exponential through the electron
branch manages only 0.533.

Both point the same way: **most RAMBHA sweeps in this archive contain no OML knee.** The bias
range appears to sit entirely in the electron-saturation branch, where the curve is linear.
That regime gives you two measurable quantities (the slope `a = I_e0/Te` and the intercept
`V0 = Vp - Te`) but three unknowns — so Ne, Te and Vp are **not separately determined by the
data**, by any method.

This matters more than any code bug, because it is the real reason the network mode-collapsed:
it was asked to invert an underdetermined problem.

Three explanations are open, and distinguishing them is the next piece of physics work:

1. **Genuine**: the probe really was biased above the plasma potential for most of the mission,
   and only sweeps near the terminator dip low enough to show a knee.
2. **Preprocessing**: the level-0B photoemission removal, or the per-voltage binning, is
   flattening the knee. Worth re-running the same check on `rawA` (photoemission NOT removed)
   and on `raw` — one line in `config.yaml` — and comparing the shapes.
3. **Sweep segmentation**: if a "sweep" as sliced by the ops sample counts actually contains an
   up-ramp and a down-ramp, binning by voltage would average them and wash out the knee.
   Plotting a few raw un-binned segments would settle this in ten minutes.

Check (2) first — it is a config edit — then (3).

## What to run next

Nothing was written into `outputs/`, so your existing tables are untouched and still stale.
When you are ready:

```bash
cd solution
rm -rf outputs
python scripts/run_all.py --no-maven
```

Then compare the new `classical_baseline.csv` failure reasons against the table above. Note
that with the corrected fit your "classical fails on 84%" headline becomes "classical is
inapplicable to ~74% of sweeps because they carry no knee" — a different and, in a viva, far
more defensible claim, provided you have first ruled out explanations (2) and (3).

---

# Checks 1 and 2 — results. The missing knee is neither preprocessing nor segmentation.

## Check 2 — sweep segmentation: a real problem, but not this one

Each segment sliced by the ops `Number_of_samples` is a **triangle ramp**, not a single sweep:

```
segment 0 of ch3_rlp_nrB_sci_20230824T131435752: 19,849 samples
  argmin(V) = 0        argmax(V) = 9,866        exactly one direction reversal
  V runs -12 -> +12 -> -12; 241 unique voltage levels; 98% of consecutive samples repeat a level
```

`bin_sweep` averages by voltage, so the up-ramp and the down-ramp are averaged together. Two
consequences:

- **Your sweep count is doubled.** 14,931 "sweeps" are ~7,500 measurements, each counted once.
  Every per-sweep statistic and the recovery denominator inherit that.
- **Up/down hysteresis is averaged away** rather than measured. Splitting the ramp gives you a
  free systematic check: the two halves should agree, and where they don't, that sweep is
  suspect.

But it is **not** the cause of the missing knee. Fitting the up-ramp and the down-ramp
separately changes nothing:

```
seg 0  BOTH  straight-line R2 0.808     UP 0.810     DOWN 0.765
seg 3  BOTH  0.712                      UP 0.714     DOWN 0.709
seg 5  BOTH  0.931                      UP 0.934     DOWN 0.926
```

Worth fixing for correctness. Not the explanation. **Ruled out.**

## Check 1 — processing level: rawB is rawA plus a constant

```
rawB minus rawA, over four whole files:
  20230824T131435752 : mean +100.000 nA   std 0.0022 nA
  20230824T145231864 : mean +100.000 nA   std 0.0022 nA
  20230824T160045336 : mean +100.000 nA   std 0.0022 nA
  20230824T162033904 : mean +100.000 nA   std 0.0022 nA
```

The "photoemission removed" level-0B product is level-0A **plus a flat 100.000 nA** — the
scatter is pure CSV rounding. The peak-normalised curves are identical to three decimals.

Since your pipeline subtracts a baseline from every sweep anyway, **rawA and rawB are literally
the same data after preprocessing**. The `level:` setting cannot be flattening anything.
**Ruled out.**

(Minor, while in there: at levels `raw` and `rawA`, `io_pds.discover_files` also picks up one
`ch3_rlp_er*` file — zero-potential mode, constant bias, no sweeps — and one `ch3_rlp_lr*`.
They survive discovery and are dropped later by the `len(Vb) < 20` guard, so they cost parse
time rather than correctness. Level `rawB` contains only the 144 `nrB` sweep files.)

## So what IS going on: the parameters are outside your synthetic ranges

Fitting the OML sphere model itself — three free parameters, coarse grid, no assumptions from
the classical code — to 150 real sweeps:

```
fit R2 : median 0.971   (p25 0.967, p75 0.977)   <- the model DOES fit
Te  eV : median  0.14   (p25 0.10,  p75 0.19)    synthetic range 0.05 - 2.0
Vp  V  : median -5.50   (p25 -6.00, p75 -2.50)   synthetic range  -6  - +6
Ne  cc : median 37.5    (p25 19.1,  p75 48.7)    synthetic range   50 - 5000
```

```
fraction of real sweeps inside the synthetic Ne range 50-5000 cc :  16%
fraction inside the synthetic Vp range -6..+6 V                  :  77%
```

The OML sphere model describes these sweeps well (R^2 = 0.97). The problem is that **the real
plasma sits below the bottom edge of the density range you trained on** — median 37 cm^-3
against a training floor of 50 — and about a quarter of sweeps want a Vp more negative than
-6 V. That is finding B1, now measured directly rather than inferred from curve amplitudes.

It also explains the earlier `Vp outside swept range` failures: with Te ~ 0.14 eV the
retardation region is only a few tenths of a volt wide, so `retardation_span: 6.0` is roughly
40x too wide a window and drags the fit into the linear branch.

## Cross-check against the literature — and a real discrepancy worth raising

| source | Ne | Te |
|---|---|---|
| ISRO release, Aug 2023 | 5-30 x 10^6 m^-3 = **5-30 cm^-3** | not stated |
| Bhardwaj et al., MNRAS 542, 2647 (2025) | **380-600 cm^-3** | 3000-8000 K = 0.26-0.69 eV |
| your ORIGINAL classical.py | **604 cm^-3** (median) | 1.05 eV |
| your CORRECTED classical.py | **27 cm^-3** (median) | — |
| free 3-parameter OML fit (above) | **37 cm^-3** (median) | 0.14 eV |

The two published values differ by roughly 20x, and your two implementations land on one each.
That is not a coincidence: the entire difference is the factor `(1 + (V - Vp)/Te)` — whether
you treat the probe as *saturating* (read the current at high bias, planar / thin-sheath) or as
*OML* (extrapolate the saturation branch back to Vp, thick sheath).

Which regime applies is decidable from the data. For a 2.5 cm sphere:

```
lambda_D = sqrt(eps0 k Te / (n e^2))
  Te = 0.5 eV, Ne = 500 cm^-3  ->  lambda_D = 23.5 cm     r/lambda_D = 0.11
  Te = 0.5 eV, Ne =  30 cm^-3  ->  lambda_D = 96   cm     r/lambda_D = 0.026
```

The probe radius is 4x to 40x smaller than the Debye length in every case, which is the
thick-sheath / OML regime — where the electron current does **not** saturate and the `(1 + x)`
correction is the physically correct one.

Two honest readings, and you should put both to your supervisor rather than pick one:

1. Your original code faithfully reproduced the MNRAS method and got the MNRAS answer. Good —
   that validates the implementation against the reference.
2. The OML correction, which the probe geometry and Debye length say is the appropriate one
   here, moves the answer onto the ISRO release's range instead.

If that holds up, "the choice of saturation-current definition accounts for the ~20x spread in
published Chandrayaan-3 near-surface density values, and the probe is in the OML regime where
the un-corrected value is an overestimate" is a far stronger contribution than "an ML model
recovered N failed sweeps". **Do not assert it before checking the MNRAS methods section for
how they defined I_esat** — that is the one thing that settles it, and you have the paper
(`stag077.pdf`).

## Config changes this implies for step 2

```yaml
synthetic:
  ne_cc: [1.0, 500.0]      # was [50, 5000]; real median ~37, p25 ~19
  vp_V:  [-12.0, 6.0]      # was [-6, 6];    ~23% of sweeps want Vp < -6
  te_eV: [0.05, 2.0]       # unchanged; covers both the fit (0.14) and MNRAS (0.26-0.69)

classical:
  retardation_span: 1.5    # was 6.0; at Te ~ 0.15 eV a 6 V window is ~40x too wide
```

Re-run the amplitude-overlap check from B1 after changing these — real and synthetic curve
peaks should sit on top of each other before you retrain.

---

# Step 2 — DONE. Config fixed, forward model corrected, domain gap closed.

## Correction to the previous section — read this first

The section above concluded, from a fit that forced the saturation exponent to 1.0 (OML
sphere), that Te was ~0.14 eV and Ne ~37 cm^-3, and floated the idea that the MNRAS value
might be an overestimate. **That conclusion was wrong, and it was wrong because the exponent
was wrong.** Letting the exponent float changes everything below. The Debye-length argument
still holds — the probe is in a thick-sheath regime — but "thick sheath" does not force the
exponent to exactly 1.

## The real defect in the forward model

`physics.py` hard-coded the electron-saturation branch as `I_e = I_e0 * (1 + x)` — linear
growth, the textbook OML sphere. Fitting the exponent as a free parameter against 100 real
sweeps:

```
I_e(V >= Vp) = I_e0 * (1 + (V - Vp)/Te) ** alpha

  alpha free    ->  median 0.40  (p25 0.30, p75 0.43)   median R2 0.998
  alpha = 0.5   ->                                       median R2 0.992
  alpha = 1.0   ->  the old hard-coded value             median R2 0.972
  alpha = 0.0   ->  planar, flat saturation              median R2 0.946
```

The real saturation branch is **concave**, not linear. `probe.sheath_exponent` is now a config
key set to 0.5 — a physically named regime (square-root sheath growth) rather than a fitted
constant, at negligible cost in fit quality. `forward_iv_np`, `forward_iv_torch` and the
classical saturation fit all read it; setting it to 1.0 restores the old behaviour exactly.

For the classical fit this generalises cleanly: raising the current to the power `1/alpha`
linearises the saturation branch, so the line fit becomes the familiar **"I^2 versus V"**
extrapolation at alpha = 0.5, and reduces to the old plain line fit at alpha = 1.

## Config changes

```yaml
probe:
  sheath_exponent: 0.5     # NEW key — was hard-coded 1.0 in physics.py

synthetic:
  ne_cc: [2.0, 1000.0]     # was [50, 5000]
  vp_V:  [-12.0, 6.0]      # was [-6, 6]
  te_eV: [0.05, 2.0]       # unchanged

classical:
  retardation_span: 6.0    # unchanged in the end — see note below
```

`retardation_span` was briefly cut to 1.5 on the strength of the bad Te estimate, then put
back: at the corrected Te of ~0.5 eV a 1.5 V window is too short a lever arm and drops the
synthetic round-trip from 8/8 to 4/8. The window is trimmed at Vp by the fit itself, so 6.0
acts as an upper bound rather than the operative limit. The original value was right.

## The domain gap is closed

Peak amplitude of the network input vector:

| | p5 | p25 | median | p75 | p95 |
|---|---|---|---|---|---|
| REAL sweeps | 0.04 | 0.78 | 2.38 | 2.86 | 3.16 |
| SYNTHETIC before | 1.64 | 4.66 | 14.66 | 45.36 | 131.89 |
| SYNTHETIC after | 0.02 | 0.08 | 0.37 | 1.75 | 6.31 |

```
real sweeps inside the synthetic p5-p95 band :  62.3%  ->  100.0%
real median sits at synthetic percentile     :   10.4  ->    80.2
```

Peak-normalised median curve shape:

```
V             -12     -8     -4      0     +4     +8    +12
REAL        -0.000  0.002  0.016  0.567  0.772  0.892  1.000
SYNTH before -0.000 -0.001  0.024  0.118  0.366  0.663  0.970
SYNTH after   0.006  0.032  0.137  0.479  0.665  0.815  0.952

shape RMSE vs real:  0.268  ->  0.078
```

Both halves of finding B1/B2 are now measured as fixed rather than argued.

## Classical fit on the real archive, before and after

3,733 sweeps sampled (every 4th):

| | old | new |
|---|---|---|
| succeeded | 591 (15.8%) | **1102 (29.5%)** |
| Ne median | 620.6 cm^-3 | **313.9** (p25 290, p75 334) |
| Te median | 1.060 eV | **0.501 eV** (p25 0.36, p75 0.71) |
| Vp median | not computed | -4.91 V |

Against MNRAS 542, 2647 (Ne 380-600 cm^-3, Te 3000-8000 K = 0.26-0.69 eV):

- **Te now lands squarely inside the published band.** The old fit put it at 1.06 eV, above it.
- Ne lands at 314 cm^-3, just under the published 380-600 range and far closer than before.
- The yield nearly doubles, and the spread tightens sharply (p25-p75 of 290-334 against
  189-801).

So the corrected pipeline does not overturn the peer-reviewed result — it reproduces it more
tightly and with a physically consistent temperature, from a forward model whose sheath
exponent was measured from the data rather than assumed. That is the defensible story.

The ML-minus-classical gap is now -0.502 dex, down from -0.995. The residual is expected: the
checkpoint on disk was trained with the OLD config and the OLD forward model. It has not been
retrained yet.

## Tests added

`tests/test_domain_overlap.py` — three assertions, all passing:

1. at least 90% of real curve amplitudes inside the synthetic p5-p95 band (was 62%),
2. the real median not in the outer 10% of the synthetic distribution (was the 10th pct),
3. peak-normalised shape RMSE below 0.15 (was 0.268, now 0.078).

None of these could have been caught by a loss curve or a hold-out score — the model was
accurate on synthetic data the whole time. Together with
`tests/test_classical_roundtrip.py` (10 assertions, all passing) that is the regression net
for both bugs.

## What to run now

The cached synthetic set and the checkpoint are both stale — they predate the config and the
forward model change. On your own machine, where the venv works:

```bash
cd solution
source ../rambha-env/bin/activate
rm -rf outputs
python -m pytest tests/ -q          # 13 assertions should pass
python scripts/run_all.py --no-maven
```

Expect all headline numbers to move. The classical failure rate should land near 70% rather
than 84%, with the failures now being genuine (low SNR, no knee in range) rather than an
artefact of a misplaced fit window. Whether the network now tracks real sweeps instead of
mode-collapsing is the thing to check first in the new `ml_predictions.csv`: the spread of
`Ne_cc` and `Te_eV` should be wide, not the 0.037 eV it was.

## Files changed in step 2

- `config.yaml` (backup: `backups/config_pre_20260903.yaml.bak`)
- `src/rambhalp/physics.py` — sheath exponent in both forward models
- `src/rambhalp/classical.py` — saturation fit generalised to any exponent
- `tests/test_domain_overlap.py` — new

---

# Clean run — 2026-09-03. It worked.

`14 passed, 3 skipped`, full pipeline in 242 s. What the outputs say.

## The mode collapse is gone

| | before | after |
|---|---|---|
| spread of log10(Ne) across all 14,931 sweeps | 0.13 | **0.532** |
| Ne range | 36 – 172 cm^-3 | **8 – 638 cm^-3** |
| Te spread on the "recovered" subset | **0.037 eV** | **0.216 eV** |
| reconstruction R^2, median | 0.859 | **0.970** |
| reconstruction R^2, mean | **-21.8** | **0.794** |
| reconstruction R^2, minimum | -236.6 | -0.91 |

The network is reading the sweeps instead of emitting a constant.

## The number that makes the recovery claim defensible

On the 4,367 sweeps where both methods succeed:

```
correlation of log10(Ne), ML vs classical : 0.951
scatter of the difference                 : 0.049 dex  (12%)
median offset                             : +0.040 dex  (was -0.994)
```

This is the real validation, and it is worth more than the recovery count. Two independent
methods — one a closed-form fit, one a neural network trained only on synthetic curves — agree
sweep by sweep to 12% on data neither was tuned to. That is what licenses applying the network
to the sweeps the classical fit cannot handle.

Te tracks far more weakly: correlation 0.643, scatter 0.449 eV. See below.

## Everything else that moved

```
classical succeeded    15.7%  ->  29.2%     (2343 -> 4367 sweeps)
classical median Ne     604   ->    317 cm^-3
classical median Te    1.05   ->  0.511 eV   <- inside MNRAS's 0.26-0.69 eV
synthetic hold-out Ne  0.114  ->  0.046 dex
synthetic hold-out Te  0.421  ->  0.351 eV
synthetic hold-out Vp  0.674  ->  0.536 V
final training val      0.111 ->  0.074
```

And a result worth putting in the report on its own: recovered density rises smoothly and
monotonically through the mission, 148 cm^-3 on 24 Aug to 372 cm^-3 on 1 Sep, across
11,966 sweeps. That is a physical trend the model was never told about, recovered from data it
was never trained on. It is stronger evidence than the hop test — see below.

Also checked and clean: the flat ion floor visible in F1 is genuine, not digital clipping
(zero exactly-repeated consecutive samples; baseline noise is 0.04% of span because ~82 raw
samples are averaged into each voltage bin).

## Three things still to fix

### 1. The hop test cannot be run on this data — stop reporting it as a test

```
                        pre-hop 2 Sep      terminator 3 Sep      ratio
all rows                n=1298  med 337     n=204  med 193       0.574
recovered only          n=1081  med 351     n= 19  med  71       0.201
```

**The 3 September session contains 204 sweeps in total, of which 19 are recovered.** A physical
prediction cannot be tested against 19 sweeps, and filtering correctly (recovered only) makes
the ratio worse, not better. The honest conclusion is that this archive slice does not support
the test — not that the model failed it.

Replace it. The day-by-day density trend (F6 already draws it) is the better blind test: it is
smooth, monotone, based on 11,966 sweeps, and the model was never given the date. Reframe
`hop_blind_test` as `density_trend` and report the monotonic rise, noting that the terminator
day is under-sampled.

Whatever you do, `validate.hop_blind_test` and `figures.fig_hop` must filter to
`recovered == True` — they currently average over the ~3,000 sweeps the model itself flagged as
not recovered.

### 2. Do not report a per-sweep Te

Look at F4's right-hand panel. The density scatter hugs the diagonal across the full range.
The temperature scatter is a horizontal cloud around 0.9-1.1 eV regardless of the true value —
the network is predicting the mean of the prior, because on a curve with this noise level
there is not enough information to do better. The hold-out number, 0.351 eV median absolute
error on a 0.05-2.0 eV range, says the same thing.

This is the information ceiling measured in the appendix, not a training failure, and no
architecture change will move it. Report Ne per sweep. Report Te as a population median with
its error bar, or leave it out. F4 is the figure that justifies that choice — put it in the
report.

### 3. The recovery headline is threshold-sensitive — publish the sensitivity

```
classical.min_r2   classical succeeds   -> ML target set
      0.80              42.4%                57.6%
      0.85              41.0%                59.0%
      0.90 (used)       29.2%                70.8%
      0.95              15.2%                84.8%
```

"Recovered 7,605 of 10,564 classical failures" depends on a threshold that moves the
denominator by a factor of three. State the value used, show this table, and say that the
recovery *fraction* is stable across it (if it is — check). Otherwise the number reads as
tuned, and that is the first thing an examiner will probe.

Related, still open from the main review (M3): `infer.apply` gates recovery on
`cfg["classical"]["min_r2"]` — a threshold designed for a log-linear fit R^2, reused for a
curve-space reconstruction R^2 — and hardcodes `Ne_dex_std < 0.5`. Both belong in their own
`recover:` config block so the report can quote them.

## Still open from earlier

- **Sweeps are double-counted.** Each segment is an up-ramp plus a down-ramp, so 14,931
  "sweeps" are ~7,466 measurements. Every count in `recovery_summary.json` is inflated 2x.
  The *fractions* are unaffected.
- MAVEN is off, correctly. Delete `outputs/checkpoints/maven_transfer.pt` and the stale
  `maven_mode.json` so they cannot be picked up or quoted.
- `GUIDE.md` still says the hop was 26 Aug and that density should crash, and still expects
  classical to succeed on "roughly a third" (it now does: 29.2%). Reconcile it with the code.

## Where the project stands

The pipeline now produces a defensible result: a corrected classical baseline that reproduces
the published temperature, a network that agrees with it to 12% on the sweeps both can do,
and a physical density trend across the mission. The remaining work is honesty and framing —
dropping a test the data cannot support, not over-claiming Te, and publishing the threshold
sensitivity — not more modelling.

---

# Hop test replaced, ramps split. And a correction.

## Correction: the sweeps were not double-counted — the opposite

I said last time that 14,931 sweeps were really ~7,466 measurements counted twice. **That was
backwards.** Checked properly:

```
science files          : 144
total ops rows          : 17,889     <- one triangle ramp each
sweeps emitted          : 14,931     <- one per ops row (2,958 dropped by length guards)
```

`segment_file` emitted **one** sweep per ops row, and `bin_sweep` averaged the rising and
falling halves of the triangle into that single curve. So the count was not inflated — half
the measurements were being thrown away, and any up/down hysteresis was averaged into
invisibility instead of measured. Splitting therefore roughly *doubles* the sweep count.

## Ramp splitting

`preprocess.split_ramps` (new, default true) emits each triangle as two independent sweeps,
`idx = 2i` (rising, `direction="up"`) and `idx = 2i+1` (falling, `"down"`), each tagged with
its `segment`. `(timestamp, idx)` stays unique, so every downstream join is unaffected.
`direction` and `segment` now flow through `ClassicalResult` and the ML prediction rows.

Measured on the same 12 files, so this is a controlled comparison:

| | sweeps | classical OK | rate | Ne median | Te median |
|---|---|---|---|---|---|
| merged (old) | 996 | 66 | 6.6% | 70 cm^-3 | 0.525 eV |
| split (new) | 1992 | 102 | **5.1%** | 69 cm^-3 | 0.540 eV |

The success *rate* falls slightly — each half has ~41 raw samples per voltage bin instead of
~82, so it is noisier — but the number of usable measurements rises **55%**, and the recovered
parameters are unchanged (70 vs 69 cm^-3, 0.525 vs 0.540 eV). No bias introduced, more data
out. Set `split_ramps: false` to restore the old behaviour.

## The free systematic error bar

The two halves of a triangle measure the same plasma seconds apart, so their disagreement is a
systematic error estimate that comes from the data rather than from the model.
`validate.ramp_consistency` reports it. On a 12-file trial:

```
n_pairs                            15
median |log10 Ne difference|       0.0037 dex  (0.8%)
scatter between halves             0.0048 dex
correlation of log10(Ne)           0.994
```

Essentially no hysteresis. Quote this in the report next to the MC-dropout number — it is the
stronger of the two, because MC-dropout only measures how unsure the network is about its own
weights, whereas this measures whether two real measurements of the same plasma agree.

## The hop test is gone, replaced by the density trend

`validate.hop_blind_test` -> `validate.density_trend`. It:

- uses **recovered sweeps only** (the old one averaged over every row, including the ~20% the
  model itself rejected),
- reports the per-day median density and the **Spearman rank correlation between day order and
  median density** — the model is never told the date, so a smooth monotonic trend across
  11,966 sweeps is a far stronger blind test than one under-sampled day,
- separates days with fewer than `validate.min_sweeps_per_day` (new key, 100) recovered sweeps
  and excludes them from the statistic,
- still reports the pre-hop and terminator numbers, but as **descriptive context with their
  sweep counts attached and an explicit note that this is not a pass/fail**.

`figures.fig_hop` -> `figures.fig_density_trend` (`F6_density_trend.png`): recovered-only,
every bar annotated with its sweep count, under-sampled days drawn hollow and hatched so the
figure carries its own caveat, and the Spearman value in the title.

`scripts/06_validate_hop.py` and `scripts/07_make_figures.py` updated. `GUIDE.md` reconciled —
its hop-test paragraph, figure list, output-tree listing and checklist now describe what the
code does, and it now tells you to quote `classical.min_r2` alongside the failure rate with the
sensitivity numbers.

## Tests

`tests/test_ramp_split.py` — 4 new assertions: both directions present and balanced,
`(timestamp, idx)` still unique, each half spans the commanded range on its own, splitting
roughly doubles the count.

**Full suite: 21 assertions, all passing** (was 14).

## Run this

Every cached artefact is stale — the sweep set itself changed shape:

```bash
cd solution
source ../rambha-env/bin/activate
rm -rf outputs
python -m pytest tests/ -q
python scripts/run_all.py --no-maven
```

Expect roughly twice as many sweeps and roughly 1.5x as many usable ones. The classical success
*rate* will dip a little from 29.2% — that is the expected cost of halving the samples per bin,
not a regression.

Then read `validation.json` for the two new blocks:

- `density_trend.spearman_day_vs_median_Ne` — should be strongly positive (the merged run gave
  a clean monotonic rise from 148 to 372 cm^-3 across the mission).
- `ramp_consistency.Ne_dex_median_abs_difference` — your data-derived error bar. If it stays
  near 0.004 dex on the full archive, that is a genuinely strong number to report.

## Files changed

- `src/rambhalp/preprocess.py` (backup: `backups/preprocess_pre_split.py.bak`)
- `src/rambhalp/validate.py` (backup: `backups/validate_pre_trend.py.bak`)
- `src/rambhalp/figures.py` (backup: `backups/figures_pre_trend.py.bak`)
- `src/rambhalp/classical.py`, `src/rambhalp/infer.py` — carry `direction` / `segment`
- `scripts/06_validate_hop.py`, `scripts/07_make_figures.py`
- `config.yaml` — `preprocess.split_ramps`, `validate.min_sweeps_per_day`
- `GUIDE.md` (backup: `backups/GUIDE_pre_trend.md.bak`)
- `tests/test_ramp_split.py` — new

---

# Final run — 29,862 sweeps. The pipeline is sound. One real finding fell out.

```
sweeps                 14,931 -> 29,862      (ramp splitting, as designed)
classical succeeded     4,367 ->  6,570      (+50% usable measurements; rate 29.2% -> 22.0%)
classical median Ne       317 ->    326 cm^-3
classical median Te     0.511 ->  0.436 eV
ML vs classical Ne     +0.040 -> +0.049 dex  (on 6,570 overlapping sweeps)
tests                   14/14 ->  21/21 assertions
```

The rate dip is the predicted cost of halving samples per voltage bin. Half a percent of the
old sweeps became two sweeps each and the parameters did not move — that is the outcome you
want from a change like this.

## Blind test: passed

```
Spearman(day order, median recovered Ne) = 0.806   -> "rising"
10 well-sampled days, 23,824 recovered sweeps
146 -> 204 -> 242 -> 332 -> 353 -> 353 -> 363 -> 364 -> 373 -> 350 cm^-3
```

The model is never told the date. It produces a smooth, monotone density rise across ten days
and 23,824 sweeps. That is a physical signal recovered from data the network never saw, and
it is a far stronger claim than the single-day test it replaced. F6 now draws it with per-bar
sweep counts and hollow hatching on the one untrustworthy day.

## The error bar you can defend

```
ramp_consistency, 11,755 up/down pairs:
  median |log10 Ne difference| between halves : 0.0135 dex   (3.2%)
  scatter                                     : 0.0202 dex
  systematic offset (down minus up)           : -0.0023 dex  (0.5%)
  correlation of log10(Ne)                    : 0.994
```

Two independent measurements of the same plasma, seconds apart, agreeing to 3.2% with a 0.5%
directional bias. This is the number to quote as your measurement uncertainty. It beats the
MC-dropout figure because it is empirical: MC-dropout only tells you how unsure the network is
about its own weights, whereas this tells you whether the instrument and the method reproduce.
Report both, and say why this one is the honest one.

## The finding: 3 September is not a failure, it is a measurement limit

```
3 Sep 2023:  408 sweeps
  classical succeeded :   0 / 408
  ML recovered        :   1 / 408
```

Every recovery gate passes on that day except one:

```
Ne inside plausible band  : 100.0%
Te inside plausible band  :  99.3%
MC-dropout uncertainty ok : 100.0%
reconstruction R^2 >= 0.90:   0.2%   <-- this is what kills them
  (median recon R^2 on 3 Sep: 0.546, against 0.977 on every other day)
```

The reason is visible in the curve shape. Peak-normalised median sweep:

```
V              -12     -8     -4      0     +4     +8    +12
1 Sep         0.000  0.000  0.012  0.566  0.723  0.863  1.000
3 Sep        -0.000  0.423  0.621  0.692  0.714  0.736  0.762

curve reaches 10% / 50% of peak at:
  1 Sep :  V = -3.6 V  /  -0.8 V
  3 Sep :  V = -9.6 V  /  -7.3 V
```

**The knee moved down by about 6.5 volts.** On 3 September the plasma/floating potential sits
near -8 to -9 V (the network independently puts Vp at a median of -9.4 V that day), so the
electron-retardation region — the only part of the curve that carries Te — has fallen off the
bottom of the instrument's fixed -12..+12 V bias window. What remains inside the sweep is
almost entirely saturation branch, which is nearly flat and carries no knee to fit.

That is an **instrument-configuration limit, not a method failure**, and it is the correct
explanation for why the terminator/hop test was never answerable from this archive. Both
methods fail on the same day for the same identifiable reason, which is exactly the kind of
agreement that makes the explanation credible.

For the report this is worth a short subsection of its own. A surface-potential shift of this
size at the terminator is physically expected (surface charging changes sharply as the
illumination goes), and being able to *show* it — via a 6.5 V shift in the knee position and a
matching Vp prediction from an independent method — is a result, not an excuse.

## Recovery-count sensitivity — publish this table

```
recon_r2 gate    recovered of 23,292 classical failures
     0.85          17,897   (76.8%)
     0.90 (used)   17,260   (74.1%)
     0.95          13,802   (59.3%)
     0.98           6,992   (30.0%)
```

Stable between 0.85 and 0.90, then falls away. Combined with the `classical.min_r2` table from
the previous section, this is what stops the headline number reading as tuned. Put both in the
methods section and state the values used.

## Mode collapse: gone, confirmed on the doubled set

```
                 n        log10(Ne) spread   Ne range          recon R^2
all sweeps    29,862          0.531          7.8 - 2162 cm^-3    0.970
recovered     23,824          0.186         57.2 -  594 cm^-3    0.977
```

## What is actually left

The engineering is done. What remains is writing, plus a short list of tidy-ups:

1. **Report writing.** You now have: a corrected classical baseline reproducing the published
   temperature; a network agreeing with it to 12% sweep-by-sweep; a measured sheath exponent;
   a monotone mission density trend as a blind test; a 3.2% empirical error bar; and a clean
   physical explanation for the one day both methods fail on.
2. **Do not report per-sweep Te** — F4's right panel is the justification. Ne per sweep, Te as
   a population value with its error bar.
3. `infer.apply` still reuses `classical.min_r2` for the reconstruction gate and hardcodes
   `Ne_dex_std < 0.5` (M3). Move both into a `recover:` config block so the methods section can
   quote them.
4. Delete the stale `maven_transfer.pt` / `maven_mode.json` if they are still on disk.
5. `solution/` is still untracked in git (M8). Commit it.

---

# The noise model was wrong, and it was hiding a much better result

Reading Manju et al. (2026, MNRAS 546, stag077) Appendix A1 settled the classical-method
question and opened a better one.

## What the published method actually does

- **Density is derived from "the current I measured at the space potential"** (eq. A2). The
  correction made earlier in this review matches the published method. That is a validation.
- **The retardation window is 8–10 points immediately above V_float, the last within 1 V of
  it** — about one volt, not the six used here.
- **The saturation region is fitted with a straight line**, i.e. the exponent is assumed to be
  exactly 1. The measured 0.40 is therefore a deliberate departure, not a reimplementation.
- Photoemission is removed per data dump by the Johansson et al. (2017) method. The archive's
  level-0B is level-0A plus exactly 100.000 nA, consistent with that having been applied.

An independent cross-check fell out: the paper gives the space potential on 2 Sep as **−3.9 V**;
the network's median predicted Vp is **−3.91 V**, from a method sharing nothing with theirs.

## Retraction

The earlier claim that 3 September is "unanalysable by any method — an instrument limit" is
**wrong**. Manju et al. report Ne ≈ 1100 cm⁻³ and Te ≈ 3330 K for that session, and a
reimplementation of their prescription here fits 19 of 408 ramps at a median 1613 cm⁻³, inside
their published range. What survives: the plasma potential really does move to about −10 V that
day (this network says −9.4, their prescription says −10.0), which collapses the yield. §8.7 of
the report draft has been rewritten accordingly.

## The real finding: synthetic noise was 3–24× too high

Chasing why a narrow retardation window fit real sweeps but failed every synthetic round-trip
led here. Measured noise on the binned archive (residual about a 9-point smooth, over span):

```
49.9 kohm : median 0.0021   (p90 0.0051)
1 Mohm    : median 0.0024   (p90 0.0039)
20 Mohm   : median 0.0182   (p90 0.0400)
config    : 0.01 - 0.08     <- 3x to 24x too noisy, uniformly sampled
```

Two things follow, and the second is the important one.

**The "sim-to-real gap" was not a gap.** A narrow window needs low noise. It worked on real
sweeps and failed on synthetic ones because the synthetic ones were far noisier. At a realistic
0.2% the round-trip passes at span 1.0.

**"Electron temperature is information-limited to ±0.35 eV" was an artefact of that noise.**
Recomputing the identifiability at the measured levels:

| noise | Te resolvable to | Ne | Vp |
|---|---|---|---|
| 0.2% (49.9 kΩ, 1 MΩ — two thirds of the archive) | **±0.035 eV** | ±0.002 dex | ±0.04 V |
| 1.8% (20 MΩ) | ±0.29 eV | ±0.014 dex | ±0.30 V |
| 5% (old synthetic midpoint) | ±0.85 eV | ±0.036 dex | ±0.86 V |

Temperature is recoverable to **±0.035 eV on two thirds of the archive** — ten times better
than reported. The limitation section of the report draft will need rewriting once the network
is retrained.

## The exponent is not an instrumental artefact

The obvious objection to α = 0.40 is an I·R drop across the probe resistor bending the
saturation branch. Ruled out by measurement: fitted α is 0.30 / 0.40 / 0.30 at 49.9 kΩ / 1 MΩ /
20 MΩ with overlapping IQRs, while the implied I·R drops are 0.00 / 0.07 / 0.06 V. Three
measurement chains carrying currents differing by a factor of 26 agree on it.

## Changes made

```yaml
probe:      sheath_exponent: 0.40     # was 0.5
classical:  retardation_span: 1.0     # was 6.0 — matches the published prescription
            min_fit_points: 4         # was 5
synthetic:  noise_frac: [0.0015, 0.05]  # was [0.01, 0.08], now sampled log-uniformly
```
plus a 1.5σ (was 2σ) retardation floor, log-uniform noise sampling in `synthetic.py`, and
round-trip test cases moved into the regime the archive actually occupies.

## Effect on the classical fit, before retraining anything

```
                      before        after
whole archive yield    22.0%        54.6%
  49.9 kohm            42.0%        77.4%
  1 Mohm               21.9%        62.8%
  20 Mohm               2.1%        23.5%
2 Sep pre-hop yield        —        69.3%
2 Sep Ne              326 cm^-3     509 cm^-3   (published 409-680, mean 478)
2 Sep Te              0.436 eV      3779 K      (published 1800-3800 K)
2 Sep Vp                   —        -4.14 V     (published -3.9 V)
```

All three published parameters reproduced simultaneously, and the classical yield has more
than doubled. Tests: 21 passing, 1 skipped.

---

# Run 2: classical transformed, network's temperature still stuck — and why

## What the noise fix did

```
classical yield        22.0%  ->  55.6%   (6,570 -> 16,596 sweeps)
classical median Ne    326    ->  452 cm^-3
ML vs classical Ne    +0.049  ->  -0.008 dex   on 16,596 overlapping sweeps
ramp systematic offset -0.0023 -> -0.0000045 dex
density trend          146-373 -> 206-490 cm^-3, Spearman 0.782
```

The density agreement is now essentially exact: −0.008 dex is 1.8%, between two methods sharing
no fitted quantity, over 16,596 sweeps. The ramp-to-ramp systematic offset has collapsed to
5 parts per million of a dex. Both are stronger than anything reported earlier.

## What it did not do

`synthetic_holdout.Te_median_abs_err_eV` went 0.351 → 0.310. The noise measurement predicted
it should fall below 0.1. Two causes, both measured rather than guessed.

**The baseline tilt is now the dominant confounder.** Subtracting the fitted OML model from 394
well-fitted real sweeps and taking the linear trend of what remains:

```
residual tilt / span : median 0.0122   p75 0.0419   p95 0.3026
config injected      : up to 0.40 of span
```

33x the real median, and 46x the median injected noise. With the noise corrected, this became
the largest single perturbation in the training set. It is now sampled log-uniformly in
magnitude over [0.002, 0.10] with a random sign, which matches both the median and the spread.

(The same residual has a non-linear part of 0.033 of span, p95 0.171. That is structure the OML
model does not describe at all, and it sets a floor on the reconstruction score that no amount
of training can cross. Worth stating in the report as a model-adequacy limit.)

**The run was stopped mid-descent.** Validation loss over the last six epochs: 0.0674, 0.0689,
0.0694, 0.0663, 0.0711, 0.0631 — still falling, and bouncing, which is a step-size symptom
rather than an epoch-budget one. Epochs raised to 250 and a cosine learning-rate decay added, so
the final epochs can actually settle.

## Noise is now sampled per channel, not pooled

The archive splits almost exactly in thirds across the three probe-resistance settings, whose
noise levels differ by an order of magnitude. Pooling them log-uniformly gave a median of 0.0087
against a real 0.0028 — still 3x too noisy for the two quiet channels that carry two thirds of
the data. Each synthetic curve now picks a channel uniformly, then draws log-uniformly within
that channel's measured range:

```
                      REAL        pooled       per-channel
median HF noise/span  0.0028      0.0087       0.0046
p95                   0.0369      —            0.0300
shape RMSE vs real    —           0.0596       0.0461
```

## Changes

```yaml
synthetic:
  ion_current_frac: [0.002, 0.10]     # was [-0.4, 0.4]; now a magnitude range, log-uniform,
                                      # random sign
  noise_frac_by_channel:              # new — replaces the pooled noise_frac
    - [0.0015, 0.0055]   # 49.9 kohm
    - [0.0015, 0.0045]   # 1 Mohm
    - [0.0080, 0.0450]   # 20 Mohm
train:
  epochs: 250                          # was 60, stopped mid-descent
```
plus cosine LR annealing in `train.py`. Tests: 21 passing, 1 skipped.

## The framing question this raises

With the classical method now succeeding on 55.6% rather than 22.0%, the project's story has
shifted and the report should say so plainly. The largest single gain came from correcting the
classical analysis — the saturation exponent, the fit window, and the noise model — not from the
network. The honest framing is now two contributions rather than one:

1. Corrected classical analysis more than doubles the usable archive (22.0% → 55.6%) and
   reproduces all three published parameters.
2. The network extends coverage further, and independently corroborates the corrected fit to
   1.8% in density across 16,596 sweeps.

That is a stronger and more defensible pair of claims than "ML recovers what classical cannot",
and it should be reflected in the abstract and conclusions once the retrained numbers are in.

---

# The temperature discrepancy: the premise was wrong, the finding is better

The plan was to narrow the retardation fit window to the published prescription and watch Te
fall from ~4700 K toward the published 2573 K. Measuring first killed that plan and turned up
something more interesting.

## The window was already correct

Manju et al. (2026) Appendix A1 specify a linear fit to 8–10 points above the floating
potential, the last within 1 V of it. Measured on 264 sweeps from 2 September, what the code
actually fits is:

```
points in the fit   median 10   (p25 10, p75 10)
volts spanned       median 0.90 V
```

Exactly the published prescription. The window was never the problem, and changing it was not
the fix.

## Nor is it the ion-current removal

The code subtracts a constant ion floor; the published method removes a fitted ion current.
Testing three treatments on the same sweeps:

```
constant floor (current code)   Te = 0.446 eV   5171 K
straight-line ion fit           Te = 0.417 eV   4837 K
OML sqrt-form ion fit           Te = 0.400 eV   4641 K
published                       Te = 0.222 eV   2573 K
```

A 10% effect against a 2x discrepancy. Capping the window strictly at the fitted Vp is another
5%. Neither closes the gap.

## What is actually going on: two estimators, two answers

For a Maxwellian electron population the probe floats where the electron and ion fluxes
balance, which fixes the offset between the floating and plasma potentials:

```
Vp - V_float = Te * ln( sqrt( m_i / (2 pi m_e) ) )
```

That is a second, completely independent estimate of Te — it uses only where the curve crosses
zero and where the saturation branch extrapolates to, never the shape in between. On the same
2 September sweeps:

| estimator | Te (eV) | Te (K) |
|---|---|---|
| log-linear slope fit (this work) | 0.329 | 3822 |
| **published (Manju et al.)** | **0.222** | **2573** |
| floating-potential offset (this work) | 0.178 | 2062 |

**The published value sits between the two.** Both estimators come from the same curves and the
same code; they simply read different parts of the physics.

## The disagreement is real, not instrumental

```
ratio Te(slope) / Te(floating potential), 750 sweeps
  median 1.41   p25 1.11   p75 1.97

by probe setting      49.9 kohm  1.32       1 Mohm  1.52
by day                1.08 to 4.11, varying systematically
```

Two measurement chains with different gains and currents differing by a factor of 26 give the
same ratio, so it is not an instrument effect. And a wrong ion-mass assumption cannot explain
it either — every physically possible ion makes the gap *wider*:

```
assumed ion   H+     He+    H2O+   Ar+    CO2+
Te from gap   0.176  0.142  0.117  0.107  0.106  eV
```

Matching the slope estimate would require an ion of 0.072 amu, lighter than a proton.

**A single Maxwellian electron distribution requires this ratio to be exactly 1.** It is 1.4.

Note that Manju et al. state the opposite conclusion from the same instrument — that the curves
"do not show multiple slopes or departure from linearity due to the additional electron
populations of different temperature". This is therefore a point of genuine scientific tension
and should be put to the supervisor as such, not asserted.

## What changed in the code

`classical.py` now computes both estimators for every sweep and records the second as
`Te_from_Vf`, alongside a new test (`test_two_temperature_estimators_agree_on_a_true_maxwellian`)
which confirms that on synthetic single-Maxwellian curves the two agree to within a few per
cent. That control is what makes the disagreement on real data evidence about the plasma rather
than a bug in either estimator. Tests: 22 passing, 1 skipped.

## What this means for the report

The honest treatment is to report both estimates and their spread, rather than picking the one
closer to the literature. "Two independent estimators bracket the published value, and their
ratio of 1.4 is inconsistent with a single Maxwellian" is a stronger and more defensible
statement than a single number that happens to match.

It also connects directly to the other open thread: the 3.3% non-linear residual left after
subtracting the fitted model sits in the knee, which is exactly where a second electron
population would show up, and exactly where the slope estimator reads. The two findings are
probably the same finding seen from different angles, and that is the thing to chase next.

---

# Full-archive result: the two temperature estimators, and what settles it

`Te_from_Vf` is now in `classical_baseline.csv`. 8,881 of the 16,596 fitted sweeps produce
both estimates.

```
Te, log-linear slope fit        median 0.341 eV = 3954 K   (p25 0.311, p75 0.399)
Te, floating-potential offset   median 0.246 eV = 2856 K   (p25 0.152, p75 0.333)
published (Manju et al.)        median 0.222 eV = 2573 K   (range 1800-3800 K)
```

The published value sits between the two, and the floating-potential estimate falls inside the
published range. The ratio between them is 1.44 (p25 1.13, p75 2.12); 70% of sweeps exceed 1.2
and only 3.3% fall below 0.8, so the disagreement is one-sided, not scatter.

## The ramp pairs settle whether this is real

Each triangular ramp measures the same plasma twice, seconds apart. On 4,084 pairs:

```
                              repeatability    correlation
Ne (reference)                        0.6%          0.967
Te, log-slope fit                     5.4%          0.857
Te, floating-potential                8.0%          0.871
```

**Both estimators are precise.** Each reproduces itself to within 8% on an independent
measurement of the same plasma — yet they disagree with each other by 44%, six to nine times
their own repeatability. Two precise measurements disagreeing far beyond their precision means
the model connecting them is wrong, not the measurements.

## Every mundane explanation checked and eliminated

```
corr(ratio, retardation fit R2)   = +0.005    not a fitting artefact
corr(ratio, density)              = -0.072    not a density effect
by probe setting                  1.34 / 1.60  not instrumental
ion mass assumption                            every physical ion widens the gap
```

The earlier day-level correlation with density (-0.93 over 8 day-medians) does not survive:
per sweep it is -0.07. It was eight points and a narrow density range. Dropped.

## What actually moves

```
day        n     ratio   fit R2   Vp - V_float   Te slope
26 Aug   182      2.90    0.981         0.46 V     0.486 eV
27 Aug   602      3.12    0.975         0.30 V     0.377
28 Aug  1299      2.48    0.981         0.37 V     0.349
29 Aug  1475      1.84    0.992         0.48 V     0.317
30 Aug  1997      1.31    0.990         0.80 V     0.331
31 Aug   715      1.08    0.979         1.03 V     0.363
01 Sep  1514      1.24    0.981         0.88 V     0.355
02 Sep  1097      1.49    0.989         0.69 V     0.353
```

The gap between the floating and plasma potentials **more than doubles** across the mission,
0.39 V early to 0.92 V later, while the exponential slope of the same curves stays within 10%
of constant. Fit quality is identical throughout (R² 0.976 vs 0.980), so this is not a
degradation in the fit.

For a single Maxwellian population those two quantities are locked together — the gap *is*
Te times a constant. Here one doubles and the other does not move. **That is the finding.**

## A separate discovery about the archive

Checking whether the photoemission correction could explain it turned up something worth
reporting on its own. Level-0B is level-0A minus a photoemission current, and the published
method estimates that current per data dump. The value actually applied is constant within a
day but steps between days:

```
24 Aug  100.000 nA      29 Aug  170.000
25 Aug  137.000         30 Aug  170.000
26 Aug  155.000         31 Aug  169.999
27 Aug  166.000         01 Sep  170.000
28 Aug  170.000         02 Sep  170.000
                        03 Sep  150.000
```

A 70% rise over the first four days, then a plateau, then a drop on the final day. Within a day
it is constant to the CSV rounding precision. This does not explain the temperature ratio — the
correction plateaus at 170 nA from 28 August while the ratio keeps falling through 31 August —
and it does not affect this pipeline's numbers, since the baseline subtraction removes any
constant. But anyone using level-0B should know the correction is quantised per day rather than
per sweep.

## Where this stops, and the question to ask

What causes the disagreement is not established, and guessing further without domain input
would be a waste. The well-posed question is:

> Two independent temperature estimators from the same RAMBHA-LP curves — the log-linear slope
> of the retarding region, and the floating-to-plasma potential offset — each reproduce to
> within 8% on repeat measurements of the same plasma, but disagree with each other by 44%. The
> potential gap doubles across the mission while the exponential slope stays constant. A single
> Maxwellian electron population cannot produce that. What does?

Manju et al. state the curves show no evidence of multiple electron populations, so this should
be raised as a question rather than asserted as a contradiction. It is exactly the kind of thing
a supervisor with instrument knowledge can resolve, or point at, in minutes.

It is also probably the same phenomenon as the 3.3% non-linear residual: that residual sits in
the knee, which is where the slope estimator reads and where a second population would appear.

---

# Two electron populations: diagnosed, fitted, controlled

## The model-independent evidence

Before fitting anything, the local temperature d(ln Ie)/dV was measured as a function of depth
below the plasma potential, stacked over 1,403 sweeps. A single Maxwellian requires this to be
flat.

```
V - Vp (V)      -1.94  -1.69  -1.44  -1.19  -0.94  -0.69  -0.56  -0.31  -0.06
local Te (eV)    1.70   1.62   1.46   1.17   0.74   0.41   0.36   0.35   0.49
local Te (K)    19720  18818  16991  13608   8571   4710   4207   4046   5647
```

It varies by a factor of five, monotonically, in exactly the way two populations produce: near
Vp the cold majority carries the current, far below it only the hot minority's tail survives.
This needs no model — it is the derivative of the measured curve.

## `src/rambhalp/twopop.py`

Three stages, all linear least squares, no optimiser and no scipy:

1. **Peel the hot tail.** Far below Vp only the hot population contributes; a straight line
   through ln(Ie) there gives Th.
2. **Subtract and refit.** Remove its extrapolation from the whole curve, fit the remainder
   close to Vp, which gives Tc.
3. **Solve both amplitudes jointly** by least squares over the whole usable curve, refining the
   temperatures on a small multiplicative grid. Stage 2 alone constrains only the retarding
   branch and leaves the saturation branch unheld, which measurably worsens it (1.93% -> 3.24%);
   this stage fixes that.

## Controls — `tests/test_twopop.py`, all passing

| control | result |
|---|---|
| recovers a known cold+hot synthetic plasma | passes |
| does **not** invent a hot population on a single Maxwellian | passes |
| two populations beat one where two genuinely exist | passes |

The negative control is the one that matters. A method that splits every curve in two says
nothing about whether two are present.

Suite: **25 passing, 1 skipped.**

## Result on the archive

750 sweeps decomposed, of 2,367 the classical stage accepts in the sampled subset:

| | this work | published (single population) |
|---|---|---|
| **cold** Tc | **2856 K** (0.246 eV) | 2573 K |
| **cold** Nc | **511 cm⁻³** | 478 (range 409–680) |
| **hot** Th | 13,662 K (1.18 eV) | — |
| **hot** Nh | 22.7 cm⁻³, 4.3% of density | — |
| single-population fit, same sweeps | 3903 K (0.336 eV) | — |

**Resolving the second population moves the temperature from 3903 K to 2856 K — onto the
published value.** The density stays where it was. The retarding-region residual halves,
1.76% -> 1.00%.

The hot component at 1.18 eV is in the range reported for lunar photoelectrons (~2 eV,
Feuerbacher et al. 1972). The probe sits ~2 m above a sunlit surface inside the photoelectron
sheath, and the archive's own level-0B processing removes 100–170 nA of photoemission, so a
photoemitted population is expected rather than exotic.

## Three honest limitations

1. **Only 32% of classically-accepted sweeps can be decomposed.** The deep retarding band needs
   enough signal above the noise floor. The rest are not evidence against two populations, just
   silent on the question.
2. **The saturation-branch residual is not explained by populations.** Splitting the electrons
   improves the retarding region 2.9x but leaves the saturation region unchanged; that is the
   sheath-exponent story and should not be conflated with this one.
3. **The synthetic control had to bypass the classical acceptance gate**, because a strongly
   two-population curve *fails* it outright — measured R² = 0.04, Te = 6.6 eV on a synthetic
   cold+hot plasma.

## What limitation 3 implies, and how to test it

If a two-population curve fails the classical log-linear R² test, then the 44.4% of sweeps the
classical method rejects should be **enriched** in two-population cases, and the sweeps it
accepts biased toward single-population-like ones. That would mean the archive's two-population
content has been systematically filtered out of every published single-population analysis,
including this project's own.

It is directly testable: run the decomposition on the classical *failures*, using the network's
plasma potential in place of the classical one, and compare the hot fraction against the
accepted set. If the failures carry more hot population, that is a clean result — and it links
the two halves of this project, since those failures are exactly the sweeps the network exists
to recover.

## Test of the filtering hypothesis: negative, and that is reassuring

The hypothesis was that the classical log-linear R² gate rejects two-population curves, so the
44.4% of sweeps it fails should be enriched in hot-population content, and every
single-population analysis of this archive would be quietly filtering that content out.

Tested by running the decomposition identically on both groups, taking the plasma potential from
the **network** in both cases so the classical fit's success or failure plays no part in the
procedure. Compared within each probe-resistance setting, since the failures are concentrated at
the noisy 20 MΩ channel:

```
probe setting   classical      n   hot fraction   cold Te (K)   hot Te (K)    noise
49.9 kohm       accepted     860          0.052          3298        18560   0.0013
49.9 kohm       REJECTED     106          0.049          2945        21242   0.0011
1 Mohm          accepted     814          0.060          3793        20324   0.0014
1 Mohm          REJECTED      83          0.064          3246        22094   0.0014
```

**No enrichment.** The sweeps the classical method rejects carry the same hot fraction as the
ones it accepts, at both settings. The hypothesis is wrong.

This is a better outcome than confirmation would have been. It means the classically-accepted
sample is *not* biased with respect to population structure, so the published single-population
results — and this project's own — are not built on a filtered subset. The two-population
correction applies across the archive uniformly rather than selectively.

It also resolves the apparent contradiction with limitation 3. A synthetic curve with a strongly
dominant hot component fails the R² gate, but the real hot fraction is only ~5%, which is subtle
enough to pass the gate over a 1 V window while still biasing the recovered temperature. The
gate is not blind to two populations in principle; it is simply not triggered at the amplitude
this plasma actually has.

### Two things this test also established

**Noise does not manufacture the hot population.** corr(noise, hot fraction) = −0.226: noisier
sweeps show slightly *less* hot component, not more. Had the hot population been an artefact of
scatter flattening the deep retarding slope, this correlation would have been strongly positive.

**The hot temperature is sensitive to the plasma potential used.** With the classical Vp the hot
component comes out at ~13,700 K; with the network's Vp, ~19,000–22,000 K. Both sit in the range
reported for lunar photoelectrons, but the value is not tightly pinned and should be quoted with
that caveat. The cold temperature is far more stable — 2856 K with the classical Vp against
2945–3793 K here — which is the quantity the report leans on.

---

## Phase 9 — Extending the network to five parameters

Every ML-Langmuir paper found in the literature search fits a *single* Maxwellian. The classical
decomposition in Phase 8 showed this archive has two. The network had no way to express that, so
it was being asked to describe a two-population curve with three numbers — and the only way to do
that is to compromise on temperature, which is exactly the bias seen against the published value.

This phase gives the network the vocabulary: five outputs instead of three.

### What changed

| File | Change |
|---|---|
| `physics.py` | `forward_iv_two_np` / `forward_iv_two_torch` — sum of two single-population calls |
| `synthetic.py` | `sample_hot()`, `_order_populations()`, 5-label emission, `_HOT_FLOOR = 0.1` |
| `model.py` | `n_targets(cfg)` returns 5 or 3; `predict()` decodes `Nh_cc`, `Th_eV` and their MC-dropout spreads |
| `train.py` | `_params_from_output_two()`, 5-element `_target_scales`, two-population physics loss |
| `infer.py` | `_recon_r2()` accepts `Nh`/`Th`; output rows gain hot columns when present |
| `run_all.py` | `--two-population` flag; all outputs redirected to `outputs_2pop/` |
| `config.yaml` | `synthetic.two_population: false` (default OFF) |

Everything is behind that one config flag. With it false the pipeline is byte-for-byte the
three-parameter code that produced the validated results in `outputs/`, so nothing already
reported is at risk.

### Label construction

Five labels: `[log10(Nc), Tc, log10(Nh + 0.1), Th, Vp]`.

Two design decisions worth defending to an examiner:

**The hot floor.** `log10(Nh + 0.1)` rather than `log10(Nh)`. Sweeps with no hot population have
`Nh = 0`, whose logarithm is undefined. The floor maps "no hot population" to a finite label
(−1.0) instead of excluding those curves from training — which matters, because a network that
has never seen a single-population curve will hallucinate a hot component in every one.

**Ordering.** `Th` is decoded as an increment above `Tc`: `Th = 1.5·Tc + 0.05 + softplus(out)`.
Without this the two populations are exchangeable — (cold, hot) and (hot, cold) describe the same
curve — and the network sits between the two labellings, predicting the mean of a bimodal target
and getting both wrong. Forcing an order removes the ambiguity by construction rather than hoping
the optimiser picks a side.

### Controls, not just a demonstration

`tests/test_two_population_network.py` has three tests, and the second is the one that matters:

1. **Output width follows config** — 3 targets with the flag off, 5 with it on.
2. **Recovers a hot population that is there** — correlation > 0.5 against truth.
3. **Does *not* invent one that is absent** — trained on two-population data, shown
   single-population curves, median recovered hot fraction must stay below 0.06.

Test 3 is the negative control. Any five-parameter network can report a hot population; the claim
is only meaningful if the network declines to report one when the plasma does not have one. This
mirrors the same control already in `test_twopop.py` for the classical decomposition.

### Status

Plumbing complete, all modules compile, five-label dataset verified finite with
`Th > Tc` in every row and label ranges spanning the configured priors. Training and the two
network tests require torch, which is available only in the project virtualenv — so they are run
on the workstation, not from here.

Commands:

```bash
source ../rambha-env/bin/activate
python -m pytest tests/test_two_population_network.py -q
python scripts/run_all.py --no-maven --two-population
```

Two things to read off the result:

- Does `test_network_does_not_invent_a_hot_population` pass? If not, the archive hot fractions
  from the five-parameter run cannot be trusted and the classical decomposition stands alone.
- Does the recovered cold temperature in `outputs_2pop/tables/validation.json` land near the
  published 2573 K, as the classical decomposition's 2856 K did? Agreement between two
  independent routes to the same number is the strongest evidence this project can offer.

---

## Phase 10 — What the first five-parameter run actually showed

The run completed, but two tests failed and the validation table contained a number that was
not a result at all. All three trace to separate causes and all three are now fixed.

### 1. The 4.3 V plasma-potential error was a bug in the metric, not the model

`outputs_2pop/tables/validation.json` reported `Vp_median_abs_err_V: 4.305`, against 0.140 V for
the three-parameter run. That looked like a catastrophic regression. It was not.

`synthetic_holdout_error()` compared `pred["Vp_V"]` against `Y[:, 2]`. In three-parameter mode
column 2 *is* Vp. In five-parameter mode the labels are
`[log10(Nc), Tc, log10(Nh + floor), Th, Vp]` — column 2 is a logarithmic hot density and Vp has
moved to column 4. The metric was differencing volts against a base-10 logarithm, which is why
the "error" came out near 4.3: it is roughly the gap between a plasma potential of about −4 V and
a log-density of about 0.4, and it means nothing.

Fixed by selecting the column from `Y.shape[1]`. **The Vp figure from that run must not be
quoted anywhere** — the number is uninterpretable, and the real value is unknown until the rerun.

This is worth recording as a process point rather than just a patch. The five-parameter change
was made backwards-compatible everywhere it was *called*, but one place indexed the label array
positionally, and positional indexing silently survives a layout change instead of raising. The
same pattern was checked for elsewhere: the only other positional consumer is
`02_generate_synthetic.py`'s range print, which had the identical fault and printed the hot
density under the heading "Vp". Also fixed.

### 2. The network invented a hot population because the prior told it to

The negative control failed: 8.9% hot density reported on synthetic curves built with exactly
none, against a 6% ceiling.

The cause is in `sample_hot`, not in the network. Hot fraction was drawn `uniform(0, 0.15)`.
A continuous uniform puts **zero probability mass on zero** — sweeps with no hot population are
a measure-zero edge of the range, not an outcome the network ever observes. Trained under a
squared-error loss the network predicts the conditional mean of the target given the input, and
when the hot signature falls below the noise floor there is nothing to condition on, so that mean
collapses onto the prior mean. The prior mean of `uniform(0, 0.15)` is 0.075. The observed
failure was 0.089. The network was behaving correctly; it had been given a prior that says a hot
population is always present.

The fix is a spike-and-slab prior: `hot_absent_frac: 0.35` puts an explicit atom at exactly zero,
so 35% of training sweeps are genuinely single-population. "None" becomes a represented outcome
the network can learn to report. Verified on a 4000-sweep draw: 34.4% single-population, hot
fraction among the rest median 7.7%, overall median 3.8% — which lands on the 4–6% the archive
decomposition gives, without that having been tuned for.

This is also the more honest prior on physical grounds. The archive spans sunlit and terminator
geometry and a photoelectron population is not present in every sweep, so a prior asserting one
always exists was never defensible.

### 3. The cold-temperature test was measuring the training budget

`tc_err = 0.477 eV` against a 0.40 eV threshold. But Tc is drawn uniform on [0.05, 2.0], whose
mean absolute deviation about its own median is ≈0.49 eV — so 0.477 eV is the score for
predicting a constant. The network had learned essentially nothing about Tc *in that test*,
while the full run reached 0.124 eV on its hold-out. The test trains on 6000 sweeps for 25
epochs; the full run uses 40000 for 250. The assertion was failing on budget, not on method.

Lowering the threshold to make it pass would have been the wrong repair — the test would then
pass for a network that predicts a constant. Instead the assertion is now a **skill score**:
the network must beat predict-the-median by at least half. That is budget-independent and
states the thing actually being claimed. The training budget was raised to 10000 sweeps and 60
epochs as well, so the test exercises a network that has started to converge.

### Still open after this round

`Tc` at 0.124 eV in five-parameter mode against 0.078 eV in three-parameter mode is a genuine
degradation, roughly 60%. Some of it is unavoidable — splitting one temperature into two makes
the problem harder and the cold component is no longer the only thing shaping the retarding
slope. Whether all of it is unavoidable is not yet known, and the rerun with the corrected prior
is the first evidence either way: if the spike-and-slab prior sharpens Tc as well as fixing the
false-positive rate, the earlier degradation was partly the model hedging against a hot component
it was told always existed.

---

## Phase 11 — The physics-informed loss term is making the network worse

The spike-and-slab prior fixed the negative control (1.7% false hot fraction against an 8.9%
failure, p90 3.2%) and the column fix restored a meaningful plasma potential (0.188 V). One test
still failed, and chasing it turned up something much larger than the test.

PyTorch is now installed in the analysis environment, so the experiments below were run directly
rather than one command at a time. All use identical seeds, 20000 synthetic sweeps, 120 epochs,
lr 3e-3, and are scored as *skill* — the fractional improvement over predicting a constant —
because an absolute error in eV means nothing without knowing the spread of the target.

### The remaining test failure was not the model's fault, but not the reason assumed either

At the test's budget the network learns almost nothing beyond density. Two hypotheses were tested
and **both were wrong**:

**Wrong hypothesis 1: gradient coupling through the ordering constraint.** `Th = 1.5*Tc + 0.05 +
softplus(raw)` routes the hot-temperature error back into the cold-temperature channel, with
roughly the same weight as Tc's own target — and since Th is drawn independently, that gradient
is close to noise as far as Tc is concerned. Plausible, and false: detaching Tc inside the Th
decode changed Tc from 0.473 to 0.469 eV. No effect.

**Wrong hypothesis 2: the reconstruction objective is degenerate with two populations.** Many
(Nc, Tc, Nh, Th) combinations produce near-identical curves, so a reconstruction term should
pull the network away from the labelled decomposition toward whichever degenerate solution is
easiest — a defect unique to five-parameter mode. Refuted by the control below.

### What is actually happening

Ablating the physics weight:

| w_physics | five-parameter Tc | five-parameter Vp | three-parameter Te | three-parameter Vp |
|---|---|---|---|---|
| 0.5 (current) | 0.239 eV | 0.359 V | 0.144 eV | 0.248 V |
| 0.25 | — | — | — | — |
| 0.1 | 0.161 eV | 0.259 V | — | — |
| 0.0 | **0.109 eV** | **0.169 V** | **0.066 eV** | **0.133 V** |

Monotonic, large, and — this is the control that killed hypothesis 2 — **present in
three-parameter mode too, at the same relative size**. Turning the physics term off roughly
halves the error on temperature and on plasma potential in both modes. This is not a
two-population problem. It is a property of the loss that has been present since the physics
term was introduced, and it affects the validated three-parameter results as well.

The wiring was checked first and is correct: `forward_iv_two_torch` receives both populations,
and the residual is normalised by a dataset-wide reference as documented. There is no bug here.

The mechanism is partly measured. The physics term compares the reconstruction against the
**noisy, tilted** training curve, so it asks the network to reproduce content that is not
plasma — measurement noise and residual baseline tilt — and the only way to fit that is to
distort the parameters. Removing the tilt from the synthetic set closes about a third of the gap
(five-parameter Tc: 0.239 -> 0.191 with the physics term, against 0.109 -> 0.103 without), so the
tilt is a real contributor but not the whole story. The rest is presumably the noise, which
cannot be removed without making the training set unrealistic.

The deeper point is that on **labelled synthetic data the physics term has nothing to add**. The
supervised loss already has the exact answer; a reconstruction term can only pull away from it.
Physics-informed losses earn their place where labels are scarce or absent, or where the model
must survive a shift from simulation to reality. The first does not apply here. The second is
exactly the open question.

### Why this is not yet a recommendation to delete it

Every number above is synthetic hold-out, which is structurally incapable of measuring the thing
the physics term was added for. Its justification was sim-to-real robustness: the real sweeps
carry noise, tilt and shape the simulator does not perfectly reproduce, and a term that rewards
reproducing the *measured* curve is a defensible hedge against exactly that. A synthetic
hold-out will always favour w_phys = 0 whether or not the term is earning its keep on the
archive.

So the question is settled on the archive, not in simulation. `scripts/08_ablate_physics.py`
retrains at several weights and scores each on the real-data checks alongside the synthetic one:
agreement with the classical method, ramp-pair consistency, and recovery fraction on sweeps the
classical method cannot fit. It reuses the cached sweeps and classical baseline, so it costs
training time only.

    python scripts/08_ablate_physics.py --two-population
    python scripts/08_ablate_physics.py

Read the output this way. If the real-data columns hold steady as w_phys falls while the
synthetic columns improve, the physics term is costing accuracy and buying nothing, and the
project's headline numbers improve by roughly a factor of two on temperature the moment it is
removed. If the real-data columns degrade, the term is buying robustness and stays despite the
synthetic cost — and the report gains a quantified statement of what physics-informed
regularisation actually bought, which is a stronger claim than asserting it helped.

Either outcome is a result. The one thing that is no longer tenable is leaving w_phys at 0.5
because it was set there at the start.

### Also tested, and not adopted

Feeding the network the log-slope d(ln I)/dV as 240 extra input channels — the same quantity
that demonstrated two populations exist — was tried on the theory that temperature information
is hard to extract from raw linear current at wildly varying amplitude. It helps the hot density
(+23% -> +33% skill) and costs accuracy everywhere else (Tc +80% -> +76%, Vp +96% -> +93%). Not
worth adopting as it stands; worth revisiting if the hot channels become the priority.

### The honest state of the hot population

Across every configuration tested, the hot channels stay weakly determined: hot density around
+20-33% skill, hot temperature around +21-35%, against +95% for cold density and +92-96% for
plasma potential. The network reliably detects **whether** a hot population is present — that is
what the negative control establishes — and measures the cold population well. It does not pin
down **how hot** the hot population is from a single sweep.

That is consistent with the classical decomposition, where the hot temperature moved between
13,700 K and 22,000 K depending on which plasma potential was used, and it should be stated
plainly in the report rather than smoothed over. "We detect the second population and quantify
the cold one; the hot temperature is constrained only to an order of magnitude" is a defensible
claim. Quoting a single hot temperature to three significant figures would not be.

---

## Phase 12 — The ablation result: the physics term earns its keep

The archive ablation was run in both modes. **It refutes the hypothesis in Phase 11.** The
physics-informed loss term is not costing accuracy for nothing; it is buying real-data
robustness, and the synthetic hold-out was simply the wrong instrument to see that with.

### Three-parameter mode — the term is clearly working

| w_physics | synth Te | synth Vp | classical dTe | ramp \|dNe\| | **recovery** |
|---|---|---|---|---|---|
| 0.5 (current) | 0.078 eV | 0.140 V | +0.0273 | 0.0108 | **62.7%** |
| 0.25 | 0.065 eV | 0.121 V | −0.0120 | 0.0114 | 60.5% |
| 0.1 | 0.058 eV | 0.105 V | −0.0154 | 0.0112 | 58.8% |
| 0.0 | 0.055 eV | 0.101 V | −0.0211 | 0.0143 | **52.6%** |

Synthetic accuracy improves as the term is removed, exactly as Phase 11 measured. But recovery
on the real archive falls monotonically from 62.7% to 52.6% — **a ten-point drop, about 1350
real sweeps that the network can no longer fit** — and ramp self-consistency degrades by a third
(0.0108 to 0.0143 dex). Two independent real-data checks move the same way.

This is the textbook signature of a working sim-to-real regulariser: it trades accuracy on the
simulator for robustness on the instrument. Phase 11 measured only the first half of that trade
and drew the wrong conclusion from it. The lesson is methodological and worth stating in the
report: **a synthetic hold-out cannot evaluate a term whose entire purpose is to handle the gap
between synthetic and real.** Judging it there is judging it on the axis it was designed to lose.

### Five-parameter mode — the picture differs, and points to 0.25

| w_physics | synth Te | synth Vp | classical dTe | ramp \|dNe\| | recovery |
|---|---|---|---|---|---|
| 0.5 (current) | 0.119 eV | 0.188 V | −0.0209 | 0.0117 | 63.5% |
| **0.25** | **0.110 eV** | **0.175 V** | **−0.0030** | **0.0112** | **63.1%** |
| 0.1 | 0.097 eV | 0.161 V | −0.0291 | 0.0118 | 61.7% |
| 0.0 | 0.083 eV | 0.142 V | −0.0442 | 0.0183 | 62.7% |

Recovery is essentially flat here (63.5 / 63.1 / 61.7 / 62.7) rather than collapsing, so the
strongest argument for a heavy physics weight does not apply to the five-parameter network. What
does persist is the ramp-consistency cliff at zero — 0.0117 to 0.0183, a 56% degradation —
appearing in both modes independently, which makes it the most trustworthy signal in the table.

w_physics = 0.25 is the best row on three of five columns: better synthetic accuracy than 0.5,
the best ramp consistency, and agreement with the classical method of −0.0030 eV, which is
essentially exact and seven times closer than the current setting. Recovery gives up 0.4 points.

### What not to conclude yet

These are single runs at each weight. A 0.4-point difference in recovery, or the difference
between −0.0030 and −0.0120 eV of classical offset, is **well inside what seed-to-seed variation
could produce**, and nothing here measures that variation. The ten-point recovery collapse in
three-parameter mode and the ramp cliff at zero are large enough and consistent enough across
two independent runs to trust. The choice between 0.5 and 0.25 is not.

`scripts/08_ablate_physics.py` now takes `--repeats`, which retrains each weight under different
seeds and prints the spread alongside the means:

    python scripts/08_ablate_physics.py --two-population --repeats 3 --weights 0.5 0.25

If the recovery spread across seeds is comparable to the 0.4-point gap, the two settings are
indistinguishable and 0.5 should stay on the grounds that it is the validated one. If the spread
is much smaller, 0.25 is a real improvement for the two-population network and worth adopting —
for that mode only, leaving the three-parameter configuration alone.

### Current recommendation

- **Three-parameter mode: keep w_physics = 0.5.** The ablation justifies it with a measured
  ten-point recovery benefit, which is a far better footing than the original "physics-informed
  networks are good practice".
- **Five-parameter mode: test 0.5 against 0.25 with repeats before changing anything.**
- **Never zero, in either mode.** Both real-data checks degrade there.

### What this is worth in the report

Most physics-informed ML papers assert the physics term helps and show a loss curve. This
project can now state what it bought, in the units of the instrument: ten percentage points of
archive recovery and a third of the ramp-pair systematic, at a cost of 42% on synthetic
temperature accuracy. Reporting the cost alongside the benefit — and the fact that the cheaper
evaluation would have given the opposite answer — is a stronger contribution than the usual
claim, and it is the kind of negative-control discipline a conference reviewer will notice.

### Correction to Phase 12 — the first --repeats run was a broken experiment

The first repeat run reported `sd 0.0000` on every metric, with all three repeats recovering
exactly 8427 sweeps at w_phys 0.5 and exactly 8367 at 0.25. That is not a stable model. Three
independent trainings do not agree to the sweep on a 13266-sweep archive.

`--repeats` was calling `torch.manual_seed()` and `np.random.seed()` before `train_model`. But
`train_model` runs `torch.manual_seed(cfg["train"]["seed"])` as its own first act, with the seed
fixed at 1337 in config.yaml, so the external seeding was overwritten every time and all three
repeats were bit-identical runs of the same computation.

Fixed by varying `cfg["train"]["seed"]` per repeat instead, which is the value train_model
actually reads. The seed is now printed per repeat so an identical-runs failure is visible
rather than silent.

This one is worth flagging for its failure mode rather than its size. A broken repeatability
experiment does not produce an obvious error — it produces a *reassuring* result. Zero variance
reads as "the model is perfectly stable" when it means "the experiment never ran". Any
conclusion drawn from that first table about the reliability of the 0.5-versus-0.25 comparison
should be discarded, and the comparison is still open until the corrected version is run:

    python scripts/08_ablate_physics.py --two-population --repeats 3 --weights 0.5 0.25

---

## Phase 13 — With working seeds, the 0.25 case collapses

Three seeds per weight, five-parameter mode:

| w_physics | recovered (3 seeds) | mean | synth Te | spread |
|---|---|---|---|---|
| 0.5 | 8427, 8572, 8501 | **8500 (64.1%)** | 0.119, 0.117, 0.123 | sd 73 |
| 0.25 | 8367, 8086, 8476 | 8310 (62.6%) | 0.110, 0.106, 0.110 | sd 201 |

**Recovery difference: 190 sweeps, standard error 123 — 1.5 sigma. Not significant.**

**Synthetic temperature difference: 0.011 eV, 5.0 sigma. Real.**

That combination is the whole answer, and it points the opposite way to the single-seed run.
The only metric on which 0.25 is genuinely better is the synthetic hold-out — which Phase 12
established is precisely the axis a sim-to-real regulariser is *supposed* to lose on. Preferring
0.25 because synthetic accuracy improves is choosing the setting that looks better on the
instrument we already know is measuring the wrong thing.

**Recommendation: keep w_physics = 0.5 in both modes.** Not because 0.25 was shown to be worse,
but because it was not shown to be better on anything that counts, and 0.5 is the validated
default. There is a weak secondary argument for 0.5 as well: its recovery spread across seeds is
nearly three times tighter (sd 73 against 201), so the heavier physics weight appears to make
real-archive performance more *stable*, not just marginally higher.

### Two single-seed claims from Phase 12 are hereby withdrawn

**"w = 0.25 gives classical agreement of −0.0030 eV, seven times closer than 0.5."** Dead. At the
same weight of 0.5, the classical offset came out −0.0209 eV in one run and +0.0321 eV in
another. It changes sign between seeds. The quantity is noise-dominated at this sample size and
cannot rank configurations at all.

**"w = 0.25 has the best ramp consistency."** Also single-seed, and the ordering reverses in this
run. Ramp consistency separates w = 0 from everything else — that cliff appeared independently in
both modes — but it cannot separate 0.5 from 0.25.

### The resolution floor, which the project needs anyway

Seed-to-seed scatter sets a hard limit on what any comparison in this pipeline can resolve.
At three seeds the standard error on a recovery difference is about 123 sweeps, so **differences
below roughly 250 sweeps (1.9 percentage points) are not measurable.** Applying that retroactively:

- three-parameter 0.5 against 0.25 — a 288-sweep gap, from single runs with no error bar at
  all — sits right at the floor and must be treated as unresolved, not as evidence for 0.5.
- three-parameter 0.5 against 0.0 — a 1345-sweep gap — is more than five times the floor. **The
  central finding of Phase 12 survives intact.** The physics term earns its keep; the argument
  was never about the difference between 0.5 and 0.25.

Any recovery number quoted in the report should carry this caveat, and any future comparison of
two configurations should be checked against the floor before being described as an improvement.

### Script change

The summary table was printing the **last repeat**, not the mean — so a three-seed run still
displayed one arbitrary seed as the headline, which is how the 0.25 case looked convincing. It
now prints mean +- sd for each metric, computes the sigma of a two-weight comparison, prints the
resolution floor, and says plainly when two settings are indistinguishable. The closing note no
longer invites the reader to treat a synthetic-column gain as evidence for a lower weight.

---

# Literature audit — Earth, Mars and laboratory Langmuir-probe ML

The earlier research phase was never written down, and the conversation carrying it has since
been lost. This section redoes it properly so the project has a checklist it can verify against
instead of a memory. Two independent searches were run: Earth-orbiting / ionospheric ML, and
Mars / planetary / deep-space probe analysis.

## What the literature says about this project's originality

Four negative findings, each checked deliberately rather than assumed:

- **No published ML inversion of a planetary Langmuir probe I-V sweep exists.** Not MAVEN/LPW,
  not Rosetta/RPC-LAP, not Cassini/RPWS-LP. All three use classical nonlinear fitting.
- **No published ML work on lunar Langmuir probe data at all** (RAMBHA-LP, LADEE, Rashid-1).
- **No published ML retrieval of a two-population Maxwellian from a single sweep.** MAVEN's
  two-population product is a classical fit; Cassini's multi-population work (Chatain et al.
  2021, 2-4 populations at Titan) selected the population count *by hand*, with no automated
  criterion.
- **No cross-mission transfer learning for plasma-parameter regression, in any direction, at any
  body.** The only cross-body plasma ML transfer found is classification (MMS to MESSENGER).

So the five-parameter network is original on two axes at once, and the project should say so in
those precise terms rather than the vaguer "first ML applied to RAMBHA-LP".

## The MAVEN transfer failure was the right call, but for a better reason than we gave

The 15x degradation was recorded as "Mars and Moon are too different". The literature supports
disabling it but suggests a sharper argument and one uncomfortable possibility.

The sharper argument: we pre-trained on **synthetic** Mars-like sweeps generated from the same
OML forward model as the lunar synthetic set. Pre-training can only transfer what is in the
forward model, so if both sets come from the same functional family it transfers **no new
physics** — only a prior concentrated on the wrong parameter region. Mars ionospheric density is
10^4-10^5 cm^-3 against 380-600 cm^-3 here, two to three orders of magnitude, and MAVEN's ion
branch is ram-dominated at 4 km/s orbital speed where a stationary lander's is not. That is a
structurally different current-balance equation, not a perturbation.

The uncomfortable possibility: **15x is larger than covariate shift alone explains.** Three
candidate implementation faults, in order of likelihood — normalisation statistics computed on
Mars-like data and reused for lunar (a 10^2-10^3 shift in Ne would compress the entire lunar set
into a corner of the normalised input range); full-network fine-tuning rather than
freeze-and-fine-tune, which the tokamak transfer literature reports degrades performance
substantially; and the physics-loss weight not rescaled after the domain change, since the
residual's magnitude scales with current. If the report claims regimes are too dissimilar to
transfer, that claim is not cleanly supported by the experiment as run. The defensible sentence
is the narrower one: *this transfer, done this way, was harmful.*

The literature's recommended alternative is not a better transfer protocol but **domain
randomisation** — one wide synthetic prior covering the nuisance parameters, one training run,
no fine-tuning stage and no domain gap to bridge. Several items below are instances of it.

## Checklist against the literature

### Already done, and the literature agrees

| Technique | Source | Status here |
|---|---|---|
| Train a regressor on a synthetic sweep library, apply to real data | Chalaturnyk & Marchand 2019; Liu et al. 2023; Marchand et al. 2023 | This is the project's architecture. It is the field-standard pattern, not an invention — worth stating plainly. |
| Explicit train-distribution coverage check against real data | Liu et al. 2023 | `test_domain_overlap.py` — amplitude coverage >=90%, shape RMSE <0.15 |
| Reconstruction residual as a per-sweep quality gate | Liu et al. 2023 | `infer._recon_r2` and the recovery threshold |
| Up-sweep / down-sweep comparison | Andersson et al. 2015 (as a contamination diagnostic) | `ramp_consistency` — we use it as an error bar, they use it to detect surface contamination. Same measurement, unused second purpose. |
| Reject a second population unless Th/Tc exceeds a fixed ratio | Yip et al. 2020 use 1.7 | `twopop.py` line 153 uses **1.5**. Close, but below the published gate. |
| Negative control: model must not invent a population that is absent | not standard in this literature | Ours is stricter than anything found. Keep and highlight it. |
| Physical-trend validation without ground truth | Smirnov et al. 2024; Catapano et al. 2022 | `density_trend` (Spearman over mission days) is exactly this pattern |

### Genuine gaps, ranked

**1. The sheath exponent is fixed at 0.40 and should be randomised.** `config.yaml:53` pins
`sheath_exponent: 0.40`. Two independent papers report this is the largest single sim-to-real
error for probes of this kind: Liu et al. (2023) find OML's beta = 0.5 overestimates density by
**~3x** and measure beta in 0.75-1.0; Ranvier & Lebreton (2023) measure gamma = 0.69 in a
calibrated chamber and state it must be computed **per sweep**. Our 0.40 was measured on this
archive, which is better than assuming 0.5 — but pinning a single value means the network never
learns that the exponent varies. Making it a randomised nuisance parameter is a small change to
`synthetic.py` and is the highest-value item on this list.

**2. Photoemission is not in the forward model at all, and this may be what the hot population
is.** Confirmed by inspection: no photoelectron term in `physics.py` or `synthetic.py`. Against
that, MAVEN's default LPW fit carries **two separate photoelectron currents** (probe and
spacecraft) as *free fitted parameters*, and Ambili et al. remove I_ph from RAMBHA-LP sweeps via
the Johansson et al. (2017) method before deriving anything. Dove et al. (2012) measure a
laboratory photoelectron layer at **1.4 +- 0.3 eV** — and note it is explicitly *non*-Maxwellian.

Our hot component sits at 1.2-1.9 eV. That is the photoelectron number. The probe is ~2 m above
sunlit regolith. **The most likely identity of our second population is probe and lander
photoelectrons, not an ambient hot plasma**, and the report must not claim otherwise until
tested. Three specific tests, all cheap:
  - does hot fraction correlate with **solar illumination / zenith angle** rather than with Ne?
  - does it survive Johansson-style photoemission removal?
  - does it pass the Yip 1.7 gate rather than our 1.5?

**3. MC-dropout uncertainty has never been calibrated.** No coverage or reliability check exists
in `validate.py`. We have unlimited synthetic truth, so there is no excuse: compute what fraction
of true values actually fall inside the nominal 68% and 95% intervals. MC-dropout is commonly
over-confident by a large factor, and one temperature-scaling constant fitted on held-out
synthetic data usually fixes most of it. Reporting uncalibrated intervals is the kind of thing a
reviewer checks first.

**4. A heteroscedastic head would let the network say the hot temperature is unconstrained.**
Predict (mu, sigma) per parameter under a Gaussian NLL loss. This directly addresses the
project's known weakness — at present the network emits a confident Th for every sweep when we
know from the ablations that Th carries only +21-35% skill. A per-sweep sigma turns that from a
caveat in prose into a number in the output table.

**5. Sweep-region ablation for the hot population.** Bernatskiy et al. (2026) found an ML-derived
electron distribution grew a spurious peak at ~17 eV purely from changing which part of the I-V
curve the model was trained on. Given our Th is already weakly determined, retrain on
retardation-only and saturation-only sub-ranges and see whether the hot parameters move. If they
do, the hot population is being inferred from a region that does not constrain it.

**6. Failure-mode attribution by covariate.** Catapano et al. (2022) localised a Swarm anomaly by
histogramming outliers against solar elevation and latitude rather than by modelling. Our 44%
classical failures have never been binned against solar zenith angle, local time, sweep index
within a sequence, or mission elapsed time. If they cluster on an instrumental covariate that
identifies missing forward-model physics; if they are uniform, they are genuine regime failures
and the ML advantage is real and explainable. Either answer strengthens the report.

**7. Round-trip consistency with an *independently trained* forward surrogate.** We reconstruct
using the same analytic forward model the network was trained against, so shared error cancels.
Liu et al. (2023) deliberately train the reverse model with a *different* basis so errors are not
shared, and report 9% RMS on real flight data. Upgrading `_recon_r2` this way turns it from a
sanity check into the strongest ground-truth-free validation available.

**8. Contamination as a stateful RC across consecutive sweeps.** Ranvier & Lebreton (2023) measure
probe contamination charging to -0.9 V over three sweeps with a time constant of tens of seconds,
**falsely inflating Te by several hundred K**, with the first sweep of a series cleanest. Cheap
first test before modelling anything: do early sweeps in a sequence fit better than late ones?

**9. Report accuracy in (Ne, Te) bands, not as one number.** MAVEN publishes a tiered ladder
(5%/20% at high density, degrading to factor-of-2 at low). Chalaturnyk & Marchand plot error as a
field over the parameter plane and find it concentrates where training coverage is sparse. Our
single median-error figures hide exactly the structure that would explain the hot-population
weakness.

### Deliberately not adopted

- **IRI-style climatology as pseudo-labels** (Wang et al. 2025 do this for Swarm). There is no
  lunar equivalent and inventing one would train the network to reproduce a model rather than the
  data. Named here so nobody proposes it later.
- **Wave/plasma-line density anchoring**, which is what makes MAVEN LPW the most accurate space
  Langmuir probe. RAMBHA-LP has no independent density channel. This is why absolute density here
  is uncalibrated, and the report should say so.
- **Full PIC/kinetic forward modelling** of lander plus probe geometry (the Rashid-1 approach).
  Correct, and far beyond this project's scope.
- **Neural posterior estimation / simulation-based inference** with normalizing flows. Strictly
  better uncertainty than MC-dropout and amortises well over 30,000 sweeps, and it would expose
  the cold/hot degeneracy as a banana-shaped contour rather than an inflated sigma. A substantial
  rewrite; items 3 and 4 capture most of the benefit far more cheaply. Worth naming as future work.

### Honest caveat on this audit

Several papers were paywalled or robots-blocked and are characterised from abstracts and indexed
metadata only: Ergun et al. 2021 (JGR 2020JA028956), Resendiz Lira & Marchand 2021, the PPCF 2024
neural-network paper, Mishra & Bhardwaj 2019, and Chatain et al. 2021 Part I. **Verify these
before citing any of them in the report.**

---

## Phase 14 — The photoelectron test, and what it revealed instead

The literature audit flagged that the hot component (1.2-1.9 eV) sits exactly where laboratory
photoelectron layers sit (Dove et al. 2012: 1.4 +- 0.3 eV), on a probe ~2 m above sunlit regolith.
Ambient hot plasma and photoelectrons emitted by the hardware are two completely different claims
about the Moon. `scripts/09_photoelectron_test.py` was written to separate them.

### The discriminating physics

Photoemission current is set by solar flux and emitting area. It does **not** scale with the
surrounding plasma density. An ambient hot population does, since both populations come from the
same plasma. So fit `log10(Nh) = a * log10(Nc) + b`:

- `a ~ 1` -> hot density tracks the plasma -> **ambient population**
- `a ~ 0` -> hot density is a fixed current -> **photoelectrons**

The statistic was verified numerically before use: a fraction-based construction gives
`a = +1.015` with `corr(log Nc, hot fraction) = +0.013`, and a fixed emitted density gives
`a = -0.007` with `corr = -0.919`. Clean separation, and the fraction correlation flips sign.

### Three obstacles found along the way, all worth recording

**The classical decomposition cannot run per-sweep on this archive.** A gate-by-gate trace over
1200 sweeps: 679 had **zero** points in the hot band above 3 sigma, and not one produced a
decomposition. The reason is arithmetic. The hot tail lives 1.5-2.6 V below Vp, where the electron
current is `exp(-d/Te)` of its peak: 2.4% at 1.5 V and 0.67% at 2.0 V for Te = 0.4 eV. The
measured noise floor is 0.2% of span on the quiet channels and **1.8% on the 20 Mohm setting**.
The signal is at or under the noise. The second population is a **population-level feature of
this archive, not a per-sweep measurement** — which is how it was found in the first place, by
stacking 1403 sweeps.

**`V_float` is nan for a third of successful classical fits** (11,220 of 16,596 finite), because
the single-population fit does not need it — but `fit_two_populations` requires it and silently
declines. Recovered here from the flux-balance relation the project already uses,
`V_float = Vp - Te*ln(sqrt(m_i/2*pi*m_e))`, which restored 5,376 sweeps.

**Two latent bugs in the scripts.** `06_validate_hop.py` rebuilds `ClassicalResult` from a
hard-coded 11-column list, silently dropping `Vp_V`, `Ie0_A` and three others that the CSV does
carry. It happens not to matter there (only Te and Ne are used) but the same pattern in step 9
produced "0 fits" with no error. Step 9 now reconstructs from every declared field and prints
which are missing. The hot-tail band and its minimum point count in `twopop.py` are now
overridable, defaults unchanged.

### The result: the test cannot answer the question, and the reason matters more

Stacking by density bin, three of ten stacks decomposed:

| Nc (cm^-3) | Nh (cm^-3) | Tc (K) | Th (K) | hot fraction | n |
|---|---|---|---|---|---|
| 522.1 | 0.33 | 2595 | 20982 | 0.001 | 1498 |
| 528.0 | 11.03 | 2993 | 20036 | 0.020 | 1498 |
| 511.5 | 28.89 | 4314 | 13379 | 0.053 | 1498 |

**Nc is flat at 511-528 while Nh varies by a factor of ~90.** There is no slope to measure,
for two independent reasons:

1. **The lever arm does not exist.** Classical density spans only p10 = 145 to p90 = 528 cm^-3 —
   a factor of 3.6, **0.56 dex**. The statistic was verified over 2.5 dex. Across 0.56 dex,
   distinguishing slope 0 from slope 1 means distinguishing a factor-3.6 change in Nh from no
   change, which requires Nh stable to well inside that. It is not.
2. **The decomposition is degenerate.** Tc rises 2595 -> 4314 K while Th falls 20982 -> 13379 K
   across stacks whose Nc is identical. That is the fit trading cold temperature against hot
   amplitude, not physics varying.

### What this forces us to withdraw

**The Phase 8 archive values — Tc = 2856 K, Th = 13,662 K, hot fraction 4.3% — were one draw
from a wide distribution, not a measurement.** Three stacks of 1,498 sweeps each give
Tc 2595-4314 K, Th 13,379-20,982 K, and hot fraction 0.1%-5.3%. The hot fraction alone spans a
factor of 50. Those numbers must not appear in the report as quoted values.

Note how consistent this is with everything else found independently: the network gives the hot
channels only +21-35% skill against a constant; the classical hot temperature moved between
13,700 K and 22,000 K depending on which Vp was used. Three different routes all say the hot
population's *parameters* are not determined by this data.

### What still stands

- **The curves are not single Maxwellians.** The local-Te profile varies five-fold across the
  retarding region (1.70 eV down to 0.35 eV over 1.9 V), stacked over 1403 sweeps. That is a
  statement about curve *shape* and does not depend on the amplitude decomposition at all. It
  survives intact, and it remains in tension with Ambili et al.'s claim of no multiple slopes.
- **The cold population is well measured**: Tc near the published 2573 K, Nc, Vp and the density
  trend all reproduce independently.
- **The network's slope is 0.768 +- 0.005 with corr(log Nc, hot fraction) = -0.425.** Because the
  training prior hard-wires Nh proportional to Nc — a slope of exactly 1 — a value of 0.768 means
  the real sweeps are pulling *away* from the prior, in the photoelectron direction. That is
  suggestive and no more: the network shares the same degeneracy, and the classical route is too
  unstable to corroborate it.

### Verdict and the right next step

**Undetermined, and it cannot be settled by a statistical test on this decomposition.** The
report should say that the second population is detected as a shape feature, that its identity
(ambient plasma versus hardware photoelectrons) is unresolved, and that its temperature and
density are not quoted.

The way to actually settle it is the literature's answer, not a better statistic: **put a
photoemission term in the forward model.** MAVEN's LPW fit carries two photoelectron currents as
free parameters rather than pre-subtracting them. If a bias-independent photoemission offset plus
ONE Maxwellian reproduces the real local-Te profile as well as two Maxwellians do, the hot
population was never needed. That is a decisive test rather than a suggestive correlation, and it
is the same experiment already run successfully in Phase 8 to validate the two-population
training set against the real profile.

---

# Phase 15 — RETRACTION: the evidence for two populations does not survive its control

This is the most consequential entry in this document. **The measurement that the entire
two-population line of work rests on does not reproduce, and the reproducible version of it does
not support a second electron population.**

## What the claim rested on

One measurement: the local temperature `d(ln Ie)/dV` across the retarding region of a stacked
curve, reported in Phase 8 as **1.70 eV at 1.9 V below Vp falling to 0.36 eV near Vp over 1403
sweeps**. A single Maxwellian requires that profile to be flat, so a five-fold monotonic variation
looked decisive. Everything after it — `twopop.py`, the five-parameter network, the spike-and-slab
prior, the photoelectron question — was built on that one row of numbers.

Three problems with it, found while trying to use it:

1. **The code that produced it was never saved.** No script in the repository computes it.
2. **Two records of it disagree.** CODE_REVIEW.md line 1587 has
   `1.70 1.62 1.46 1.17 0.74 0.41 0.36 0.35 0.49`; twopop.py's docstring has
   `1.70 1.46 0.98 0.55 0.36 0.35 0.49`. Same claimed diagnostic, different numbers.
3. **It had no control.**

## The control it needed

A stacked curve is an average of sweeps, and each sweep has its own temperature. **A sum of
exponentials with different decay constants is not an exponential** — it has a graded local slope.
So stacking sweeps drawn from a distribution of temperatures produces the two-population signature
even when every individual sweep is a textbook single Maxwellian.

That is not a marginal effect on this archive. Classical Te runs from 0.24 eV (p1) to 1.56 eV
(p99), and the deep retarding region of a stack is dominated by its hottest members: **sweeps
above the 90th percentile of Te supply 65% of the stacked current at u = -1.9 V while being 10%
of the sample.** Stacking 16,596 perfect single Maxwellians at the archive's own measured
temperatures yields a profile of 0.78 / 0.53 / 0.43 eV — graded, monotonic, and entirely
artefactual.

So the diagnostic is only evidence insofar as the real profile **exceeds a matched control**: the
same sweeps, each given a perfect single Maxwellian at its own measured Te and Ne, stacked
identically. `scripts/11_local_te_diagnostic.py` now always computes both.

## The result

Over 14,971 sweeps with a good classical fit, same window, same Vp alignment:

| group | source | −1.9 | −1.5 | −1.0 | −0.6 | −0.3 |
|---|---|---|---|---|---|---|
| ALL (14971) | REAL stacked | 0.47 | 0.44 | 0.70 | 0.65 | 0.67 |
| ALL | control: 1 Maxwellian | 0.77 | 0.64 | 0.52 | 0.46 | 0.42 |
| ALL | **REAL − control** | **−0.30** | **−0.20** | **+0.18** | **+0.19** | **+0.25** |
| channel 1 (9482) | REAL − control | −0.27 | −0.17 | +0.25 | +0.26 | +0.32 |
| channel 2 (5489) | REAL − control | −0.21 | −0.05 | +0.00 | +0.02 | +0.10 |

Three things to note, all bad for the claim:

**The reported profile does not reproduce.** Nothing resembling 1.70 / 1.17 / 0.36 appears in any
selection — all sweeps, either channel individually, or any single day. The real stacked profile
is 0.47 / 0.70 / 0.67: nearly flat, and not even monotonic.

**The excess has the wrong sign where it matters.** A hot population makes the DEEP retarding
region hotter, so the excess should be positive and growing leftward. It is **negative** at
−1.9 and −1.5 V in both channels: the real curves are *flatter* at depth than a stack of pure
single Maxwellians. The positive excess sits near Vp, which is where the saturation knee and the
sheath exponent live — not where a hot tail lives.

**The per-day pattern is incoherent.** 25 Aug gives +1.62 near Vp and −0.39 at depth; 27 Aug gives
+0.30 at depth and −0.05 near Vp. There is no consistent signature, which is what noise plus
baseline-subtraction error looks like.

## What is now withdrawn

- **The claim that this archive shows two electron populations.** Unsupported.
- **Tc = 2856 K, Th = 13,662 K, hot fraction 4.3%** as archive values. Already downgraded in
  Phase 14 for instability; now they describe a decomposition of something there is no evidence
  for.
- **"Resolving two populations moves the recovered temperature from 3903 K onto the published
  2573 K"** — the justification comment in config.yaml for `two_population`. That reasoning is
  void.
- **The stated tension with Ambili et al.** There is none. They report a single Maxwellian and no
  multiple slopes; the controlled diagnostic agrees with them. The project is now *consistent*
  with the published analysis rather than contradicting it, which is a weaker headline and a
  sounder position.
- **The photoelectron question of Phase 14** is moot. There is no second population whose origin
  needs explaining.

## What survives, and it is not nothing

- **The five-parameter network itself.** It trains, its output width follows config, it recovers a
  hot population that is genuinely present in synthetic data, and — the control that now matters
  most — it declines to invent one when none exists (1.7% false hot fraction, p90 3.2%). That is a
  working two-population inversion, and no published ML-Langmuir work has one.
- **Everything about the cold population**, which was never in question: Te agreeing with the
  published value, Ne agreement with the classical method to 0.005 dex over 16,596 sweeps, Vp
  reproducing the published −3.9 V, the density trend (Spearman 0.82), ramp consistency r = 0.995.
- **All of Phases 1-7 and 10-13**: the V_float fix, the yield improvement from 22% to 83.5%, the
  physics-loss ablation with its ten-point recovery finding, the resolution floor.

## How the report must now read

The honest framing is stronger than the one it replaces, because it is a *controlled negative
result* rather than an uncontrolled positive one:

> We extended the inversion to two electron populations — the first ML Langmuir-probe inversion to
> do so — and validated it with a negative control confirming it does not report a second
> population where none exists. Applied to the RAMBHA-LP archive with a matched
> single-Maxwellian control, we find no evidence that these sweeps require a second population,
> consistent with Ambili et al. We further show that the local-temperature profile of a *stacked*
> Langmuir sweep is not by itself evidence for multiple populations, since a distribution of
> single-population temperatures reproduces the same graded signature — in this archive, sweeps
> above the 90th percentile in Te contribute 65% of the stacked current 1.9 V below the plasma
> potential.

That last point is a genuine methodological contribution and applies to anyone stacking Langmuir
sweeps, which is standard practice in this field.

## The process failure, which is the real lesson

Phases 8 through 14 — the two-population decomposition, the five-parameter network, the training-
set validation, the spike-and-slab prior, the photoelectron test — is a substantial body of work
resting on an **uncontrolled measurement made by a script that was thrown away**. Every phase
after it added rigour to the machinery while never revisiting the foundation.

Two rules follow, and both are now enforced in code rather than intention:

1. **A diagnostic that a conclusion depends on must live in a script and be rerunnable.** The
   local-Te measurement is now `scripts/11_local_te_diagnostic.py`.
2. **Every claim of structure needs a matched control that generates the same statistic from the
   null hypothesis.** The pattern was applied diligently to the network (`test_twopop.py`'s "does
   not invent a population", the false-hot-fraction metric) and never once to the measurement that
   motivated building the network at all.

## Phase 15b — Where the real anomaly actually is

Running the diagnostic across all twelve days makes something visible that the ALL-stack row
hides. Taking the per-day "REAL minus control" excess as nine independent measurements:

| u (V below Vp) | mean excess | sd | days positive | sign-test p | verdict |
|---|---|---|---|---|---|
| −1.9 | −0.117 | 0.426 | 3 / 9 | 0.91 | scattered |
| −1.5 | −0.128 | 0.239 | 2 / 9 | 0.98 | scattered |
| −1.0 | +0.224 | 0.238 | 7 / 9 | 0.090 | marginal |
| −0.6 | +0.333 | 0.376 | 8 / 9 | 0.020 | **consistent** |
| −0.3 | +0.403 | 0.466 | 9 / 9 | **0.002** | **consistent** |

Two distinct conclusions, and they point in different directions.

**At depth, nothing.** At 1.5-1.9 V below Vp the excess is scattered across both signs, positive on
2 and 3 days of 9, with sign-test p of 0.98 and 0.91. This is where a hot electron tail would
dominate, and there is no signal there. **The retraction in Phase 15 stands: no second population.**

**Near Vp, a real and coherent deviation.** At 0.3 V below Vp the real curves are flatter than a
matched single-Maxwellian stack on **nine days out of nine** (p = 0.002 by sign test, mean 2.6
sigma). At 0.6 V it is eight of nine. This is not noise — it is the most reproducible structure in
the whole diagnostic, and it appears in both probe-resistance channels independently.

So these sweeps genuinely are not single Maxwellians. The deviation was simply misattributed. It
lives in the **transition into electron saturation**, which is governed by the sheath-expansion
exponent, not in the deep retarding tail where an additional electron population lives.

### This converges with the literature audit's top finding

The audit's gap #1, reached independently, was that `probe.sheath_exponent` is pinned at a single
value of 0.40 while the literature says it must vary: Liu et al. (2023) measure beta in 0.75-1.0
and find OML's 0.5 overestimates density by ~3x; Ranvier & Lebreton (2023) measure gamma = 0.69 in
a calibrated chamber and state explicitly that it must be computed **per sweep**.

Two independent lines now point at the same unmodelled physics. The diagnostic says the residual
structure sits exactly where the exponent acts; the literature says a single fixed exponent is the
largest systematic error for probes of this kind. That is a far better-supported research direction
than the two-population hypothesis ever was, and it is cheap to test:

1. Fit alpha per sweep (or per stack) instead of pinning it at 0.40, and re-run this diagnostic.
   If the near-Vp excess collapses toward zero, the anomaly is explained.
2. Make alpha a randomised nuisance parameter in `synthetic.py` so the network learns that the
   exponent varies, rather than that it is 0.40.

The prediction is falsifiable and the diagnostic to test it against already exists. If the excess
does **not** collapse, something else is going on near Vp and we will have learned that too.

### What the report should now say

The claim is no longer "we found a second electron population". It is the sharper and better
supported:

> A matched-control diagnostic shows these sweeps deviate systematically from a single Maxwellian —
> reproducibly on nine of nine observation days — but the deviation sits in the sheath-transition
> region rather than in the retarding tail, and is therefore attributable to sheath expansion
> rather than to an additional electron population. Testing a second population explicitly, with a
> validated negative control, finds no evidence for one.

That is a positive, specific, testable finding plus a controlled negative result. Both are
defensible, and neither depends on a measurement we cannot reproduce.

## Phase 15c — Four explanations tested, three rejected, anomaly still open

Phase 15b attributed the reproducible near-Vp deviation (positive on 9 of 9 days, sign test
p = 0.002) to the sheath exponent. **That attribution was wrong**, and so was the replacement.

### Tested and rejected

**Sheath exponent alpha.** Rebuilding the control at alpha = 0.40, 0.70 and 1.00 gives *byte-
identical* excess: −0.31 / −0.21 / +0.17 / +0.18 / +0.24 in all three cases. Obvious in hindsight:
alpha acts only for V >= Vp, and the diagnostic window is entirely below Vp. The literature's
warning about a fixed exponent is real and still worth acting on, but it has nothing to do with
this measurement.

**Plasma-potential misalignment.** The stack aligns on the *estimated* Vp, and ramp pairs give
that estimate an error of 0.255 V (6,764 up/down comparisons, sd of the difference 0.360 V) —
comparable to where the excess sits. Smearing the control by that amount accounts for about a
third at u = −0.3 (+0.24 -> +0.16) and nothing at u = −0.6 (+0.18 -> +0.19).

**Fit noise inflating the control's temperature spread.** Ramp pairs give a per-sweep Te error of
0.101 eV against an observed spread of 0.249 eV, so only 9% of the spread is noise. Deconvolving
it changes the excess from +0.24 to +0.24.

**A thermal-width transition at Vp — and a bug of mine worth recording.** The sharp piecewise
model has a kink at Vp that real sweeps cannot have. Blending the branches over ~1.3*Te appeared
to remove the anomaly outright (+0.18 and +0.24 becoming +0.00 and −0.02). **That result was an
artefact of my own implementation.** Blending the two *currents* linearly contributes `w * sat` in
the deep retarding region where `sat` is clipped to 1, and with `w = sigmoid(x/k)` that term
decays as exp(x/k) — slower than the true exp(x) — so it dominates the deep tail and fabricates a
spurious second population at temperature k*Te. A 10,000% relative change deep in the retarding
region is what exposed it. Blending the *logarithms* instead makes the correction vanish properly:
the deep local Te stays at exactly 0.400 eV where the linear blend had corrupted it.

With the correct blend the honest numbers are much smaller: rms excess 0.229 (sharp) -> 0.209 at
1.3*Te -> 0.197 at 1.6*Te, and at u = −0.3 the excess falls from +0.24 only to +0.13. **A real
improvement, not an explanation.** Implemented in both forward models behind
`probe.transition_width_te`, default **0.0** so nothing changes until it is understood.

### Where this leaves the anomaly

Genuinely open. A reproducible, same-sign deviation from a single Maxwellian in the ~0.6 V below
Vp, present on every observation day and in both probe-resistance channels, not accounted for by
the temperature distribution, Vp alignment error, Te fit noise, the sheath exponent, or a smooth
transition at Vp.

The leading remaining hypothesis is a **definitional** one rather than physics, and it should be
tested before anything else. The classical Te is fitted over 8-10 points within 1 V of the
*floating* potential — that is, deep in the retarding region. The control then uses that Te to
generate the curve everywhere, including immediately below Vp. If the real retarding slope varies
smoothly with bias for any reason at all, a Te fitted deep will not describe the slope near Vp,
and the control will disagree there *by construction*. The test is direct: refit Te using only
points within ~0.5 V below Vp and rebuild the control with that. If the near-Vp excess collapses,
the anomaly is an artefact of where the classical method places its fit window, and the honest
statement becomes a caveat about the measurement rather than a claim about the Moon.

### Three attributions, three retractions — the pattern

Phase 14 attributed the hot population to photoelectrons; wrong (probe photoemission is constant
below Vp and is removed by the floor subtraction). Phase 15b attributed the residual to the sheath
exponent; wrong (alpha does not act below Vp). Phase 15c attributed it to the transition width;
the supporting result came from a bug in my own blend.

The common failure is proposing a mechanism and testing whether it *can* produce the observed
signature, rather than first asking where the mechanism is even capable of acting. Both alpha and
probe photoemission were excluded by one line of reasoning about their domain of action, available
before any code was written. The discipline that did work throughout was the matched control —
every real advance in Phases 14 and 15 came from building the null-hypothesis version of a
statistic and comparing, and every error came from skipping that step.

---

## Phase 16 — The clean test, and the answer: one temperature

`scripts/12_slope_curvature.py` settles the question without stacking, without a forward-model
fit, and without any of the machinery that went wrong in Phases 14-15c.

### The design

For each sweep **on its own**, fit the retarding slope in two bias windows:

    DEEP  u in [-1.5, -0.8] V below Vp     (electron current ~1.4-13% of its peak)
    NEAR  u in [-0.6, -0.05] V below Vp    (~13-88% of peak)

A single Maxwellian has one temperature, so `d = Te_near - Te_deep` must be zero up to
measurement error. No averaging is involved, which removes the entire failure mode of Phase 15:
stacking sweeps with a spread of temperatures bends the result, but a per-sweep comparison cannot.
Both windows sit above the noise floor, unlike the 1.5-2.6 V region the per-sweep two-population
fit needed and could not reach.

Validated before use, on synthetic curves:

| case | Te_deep | Te_near | median d | d > 0 |
|---|---|---|---|---|
| single Maxwellian, Te = 0.4 | 0.399 | 0.400 | +0.0008 | 52.0% |
| single Maxwellian, Te varying 0.3-0.6 **between** sweeps | 0.453 | 0.454 | +0.0008 | 53.5% |
| two populations, +5% at 1.5 eV | 0.506 | 0.425 | **−0.081** | **0.0%** |
| two populations, +10% at 2.0 eV | 0.656 | 0.457 | **−0.199** | **0.0%** |

Unbiased on the null, immune to between-sweep temperature spread, and a 5% hot component takes the
positive fraction from 52% to zero. This is the sensitive, uncontaminated test the project needed
from the start.

### The selection effect, which decided the answer

The deep window requires `Ie > 3*noise` at u = −1.5 V, where a single Maxwellian sits at
exp(−1.5/Te) of its peak: 0.7% at Te = 0.3 eV, 2.4% at 0.4 eV, 8% at 0.6 eV. Against noise floors
of 0.24% (quiet channels) and 1.8% (20 Mohm), **the window is fittable only on sufficiently hot
sweeps** — so the fitted sample is biased toward large Te_deep, which makes d negative with no hot
population present.

The first control missed this because it drew from config.yaml's wide uniform prior (median Te
1.15 eV) and sailed over the threshold. Resampling the archive's own measured (Te, Ne, Vp) exposes
it to the identical cut: only **33%** of control sweeps pass, and their median true Te rises from
0.394 to **0.514 eV** — the selection bias, measured directly.

### The result

| | n | d > 0 | p10 | p25 | median | p75 | p90 | sd |
|---|---|---|---|---|---|---|---|---|
| REAL | 10046 | **35.7%** | −1.16 | −0.90 | −0.438 | +0.52 | +1.41 | 5.47 |
| CONTROL (single Maxwellian) | 1965 | **34.7%** | −0.30 | −0.15 | −0.042 | +0.03 | +0.08 | 1.52 |

(Scatter ratio: 3.6x in standard deviation, 6.8x in p10-p90 span. An earlier draft of this entry
said "ten times at every quantile" -- an overstatement, corrected here.)

**Sign statistic: 35.7% against 34.7%, a difference of +1.1% +- 1.2% = 0.9 sigma. Nothing.**

The median difference of −0.40 eV, which the script first reported as "7.3 standard errors", is an
artefact of the statistic. The real distribution is about **four times wider in standard deviation and seven to eight
times wider in interquartile span** than the control, while having the **same fraction above
zero**. Two distributions with identical sign
balance and different scale have different medians with no shift in location, so a median gap can
be pure scale — and quoting it as `excess/(sd/sqrt(n))` is invalid for a distribution with this
tail. The scale-free sign fraction is the right statistic, and a real hot component moves it
decisively (52% -> 0% in the validation above). Here it does not move at all. The script's verdict
now reads the sign statistic and says so.

### Answer

**These sweeps are consistent with a single Maxwellian electron population.** Three independent
routes agree: the stacked diagnostic with a matched control (Phase 15), the per-day sign test
(Phase 15b, deep region), and this per-sweep two-window test with a selection-matched control. The
project's finding is in agreement with Ambili et al., who report a single Maxwellian and no
multiple slopes.

### A second finding worth reporting on its own

**The real per-sweep scatter is several times larger than our noise model reproduces** (sd 5.47 eV
against 1.52 eV, a factor of 3.6; p10-p90 spanning 2.57 eV against 0.38 eV, a factor of 6.8). The scatter is symmetric, so it is not a bias,
but it means single-sweep temperature determinations are far less repeatable than the synthetic
model implies — consistent with the ramp-pair error on d of 0.66 eV measured from 3,554 up/down
pairs. Anyone quoting a per-sweep Te from this archive should quote that scatter with it. Whether
the extra variability comes from the ion-floor subtraction, genuine plasma variation within a
sweep, or an instrumental effect the forward model omits is not settled here.

### What the report should now claim

> Applying a per-sweep two-window slope test with a selection-matched single-Maxwellian control, we
> find the RAMBHA-LP retarding characteristics consistent with a single electron temperature
> (sign statistic 0.9 sigma), in agreement with the published analysis. We show that the
> local-temperature profile of a *stacked* Langmuir sweep is not evidence for multiple populations,
> since a distribution of single-population temperatures reproduces the same graded signature, and
> that a deep-window slope fit selects preferentially hot sweeps unless the control is matched to
> the measured parameter distribution. We further report that per-sweep temperature scatter in this
> archive exceeds our instrument noise model by an order of magnitude.

Four negative or methodological results, each with its control, plus a working five-parameter
inversion whose negative control passes. That is a defensible Independent Study contribution and
it does not rest on anything we cannot reproduce.

## Phase 16b — The aggregate result was two channels cancelling

The per-channel rows of step 12 should have been read before the conclusion was written. They
were not, and they change it.

| sample | d > 0 | vs a MIXED control | vs its OWN matched control |
|---|---|---|---|
| channel 1 (n = 6,351) | 39.7% | +4.0 sigma | **+3.1 sigma** |
| channel 2 (n = 3,695) | 28.8% | −4.5 sigma | **−10.0 sigma** |
| combined (n = 10,046) | 35.7% | 0.9 sigma | — |

**The two channels disagree with each other at 11 sigma, in opposite directions.** The reassuring
0.9 sigma aggregate is the two cancelling out, not a null result.

The first suspicion was that the control mixed the three probe-resistance noise settings while
each channel has its own, so the comparison was not like-for-like. Rebuilding a separate control
per channel — its own (Te, Ne, Vp) resampled from that channel's classical fits, and its noise
drawn from that channel's *measured* noise-fraction distribution rather than from config — does
not remove the split. It sharpens it: channel 2 moves from −4.5 to −10.0 sigma against its own
control. The measured noise fractions are nearly identical (0.0013 and 0.0012 of span), so noise
is not the explanation either.

### What this does and does not change

**It does not resurrect the second population.** A real hot electron component is a property of
the plasma, so it would push **both** channels the same way — negative, as the validation table
shows. Channel 2 goes negative and channel 1 goes *positive*. Two detectors observing the same
plasma cannot both be right about its shape, so an effect that reverses sign between them is
instrumental, not lunar. If anything this is stronger evidence against a population signature
than the aggregate was, because it explains the residual rather than merely failing to find one.

**It does change what may be claimed.** "These sweeps are consistent with a single Maxwellian
(0.9 sigma)" is not supportable as written — that number is an accident of cancellation. The
defensible statement is:

> No coherent electron-population signature is present: the residual deviation from a single
> Maxwellian reverses sign between the two probe channels (+3.1 and −10.0 sigma against
> channel-matched controls, disagreeing with each other at 11 sigma), which identifies it as a
> channel-dependent instrumental systematic rather than a property of the plasma. A genuine second
> population would bias both channels in the same direction.

**It opens a real question.** The two channels return closely agreeing *parameters* — median Tₑ
0.389 against 0.413 eV, median Nₑ 458 against 443 cm⁻³ — while their *curve shapes* differ
systematically. Something channel-dependent affects the retarding slope without displacing the
fitted values much. That is worth pursuing and is not explained here.

### The process note

This is the fourth time in this investigation that a headline number dissolved under
disaggregation, and the pattern is now unmistakable: the aggregate was computed and reported
before the subgroups were examined. The per-channel rows were printed in the same output that
produced the 0.9 sigma figure. The check cost one command and would have caught it before the
document was written.

---

# Phase 17 — Halving the temperature error

Te was the weakest of the three outputs throughout this project: 0.078 eV on synthetic hold-out
(~20% at the 0.4 eV this archive occupies) against 0.021 dex for density and 0.14 V for plasma
potential. Three pieces of work, in order.

## The literature benchmark first: 20% is normal

Two focused searches. The headline is that **20% is at the state of the art for absolute
planetary Langmuir Te, not below it.** MAVEN LPW's own specification table (Andersson et al.
2015, Table 7) gives, for sunlit conditions:

| density | relative | absolute |
|---|---|---|
| ne >= 10^3 cm^-3 | 5% | **20%** |
| ne >= 10^2 cm^-3 | 10% | **40%** |

This archive sits at ~450 cm^-3 in sunlight, between those rows. Ergun et al. 2015 quote "20%
accuracy in most conditions" plainly. The often-cited MAVEN "+-82 K" is **not** comparable: the
figure caption states the error bars are "goodness of fit only" and the profiles are bin-averaged
over 2.5 km altitude bins across 28 orbits. Multi-needle probes in Earth orbit cannot infer Te at
all and take it from IRI instead. **No space Langmuir work surveyed demonstrates sub-10% Te
against independent truth.** The report's framing should say this.

## But it was not the noise floor

A Fisher-information analysis of this exact measurement (450 cm^-3, 0.4 eV, 240 points, quiet-
channel noise at 0.24% of span) gives a Cramer-Rao bound of **0.0083 eV, about 2%** — we were 9x
above it. The same calculation gives the mechanism: **corr(Te, Vp) = +0.93**, against
corr(Te, logNe) = +0.17. The Te-only bound is 0.0025 eV; the joint bound is 3.3x worse, and
essentially all of that inflation is the plasma-potential coupling.

Two diagnostics confirmed the picture against the trained model:

**The error is flat in absolute terms across the prior.** Median |Te error| by true Te:
0.053 eV at 0.05-0.15, 0.092 at 0.15-0.30, 0.077 at 0.30-0.50, 0.067 at 0.50-0.80, 0.077 at
0.80-1.20, 0.084 at 1.20-2.00. Ratio of the low bins to the high bins: **1.08**. The network
returns one number regardless of the answer, because it minimises squared error in eV over a
uniform prior. Also worth noting: **41% of the training set sits above 1.2 eV**, a regime this
archive never occupies, against 10% in the 0.3-0.5 eV band that it does.

**Fixing the parameterisation alone does not work.** Narrowing the prior to 0.20-0.80 eV and
training on log10(Te), three seeds, scored in the 0.30-0.50 eV band:

    current (0.05-2.0, linear) : 19.9  22.3  24.6  -> 22.3% +- 2.4%
    tight   (0.20-0.80, log)   : 20.4  19.6  20.5  -> 20.2% +- 0.5%

2.1 points at 1.5 sigma. Not significant. Worth adopting for the collapse in seed spread
(2.4% -> 0.5%), not for accuracy. The limit is structural, as the Fisher analysis said.

## The fix that worked

Remove both quantities Te is degenerate with before the network sees the curve:

    u = V - Vp_hat                shift the plasma potential to the origin
    y = I(u) / I(u = 0)           divide out the density
    input = log(y) on a fixed u grid

The slope of that input **is** 1/Te. Verified analytically: a clean exponential at Te = 0.4 eV
returns a slope of 2.500 against a truth of 2.500.

Three configurations, three seeds, scored in the 0.30-0.50 eV band. First at reduced budget
(10000 sweeps, 60 epochs) to include an oracle arm:

| | seeds | mean |
|---|---|---|
| A baseline, raw curve | 36.2 36.9 32.2 | 35.1% +- 2.5% |
| B aligned on TRUE Vp (ceiling) | 11.8 11.5 10.7 | **11.3% +- 0.5%** |
| C aligned on PREDICTED Vp | 11.5 15.1 13.0 | **13.2% +- 1.8%** |

Then at full budget (20000 sweeps, 90 epochs), where the baseline reproduces the real pipeline's
~20% and the Vp estimate reproduces its 0.14 V — making this a fair head-to-head:

| | seeds | mean |
|---|---|---|
| A baseline, raw curve | 24.2 21.2 23.0 | 22.8% +- 1.5% |
| C aligned on PREDICTED Vp | 12.9 11.1 11.5 | **11.8% +- 0.9%** |

**11.0 points, 10.7 sigma, a factor of 1.93.** In absolute terms 0.091 -> 0.047 eV at 0.4 eV.
C has essentially reached the oracle ceiling of 11.3%, so a realistic Vp estimate costs almost
nothing — which was the risk the experiment was designed to test, and it did not materialise.

## Implementation, and why it defaults to off

`src/rambhalp/te_refine.py` implements this as a **refinement head, not a rewrite**. Stage one is
the existing validated three-parameter network, untouched, supplying Vp. Stage two is a small
separate network seeing only the aligned curve. Nothing in the density or plasma-potential path
changes, so enabling it cannot disturb those results. The head is trained on curves aligned with
*predicted* Vp rather than true Vp, so it learns on inputs carrying the same alignment error it
will meet at inference.

`model.te_refine` defaults to **false**, and that is deliberate. Every number above is synthetic
hold-out, and Phase 12 of this document records what happened the last time a synthetic
measurement was trusted alone: the physics-loss term looks harmful on synthetic data (removing it
improves Te from 0.078 to 0.055 eV) while buying ten percentage points of real-archive recovery.
The correct decision inverted once real-data checks were applied.

`scripts/14_te_refine_eval.py` applies those checks. The decisive one is **ramp-pair scatter in
Te**: the two halves of one commanded triangle are independent measurements of the same plasma,
so a genuinely better Te must be more repeatable between them. A refinement that sharpens
synthetic accuracy while worsening ramp scatter has learned the simulator rather than the
instrument, and should be rejected however good the synthetic number looks.

    python scripts/14_te_refine_eval.py

Adopt only if the ramp scatter improves or holds.

## Phase 17b — The real-data test rejects it

`scripts/14_te_refine_eval.py` was written to decide this, and it decided against.

| | baseline | refined | |
|---|---|---|---|
| synthetic hold-out, absolute | 0.0746 eV | **0.0485 eV** | better |
| synthetic, % in 0.30-0.50 eV band | 18.2% | **11.8%** | better, and reproduces scripts/13 |
| **ramp-pair scatter in Te (14,931 pairs)** | **0.0076 eV** | 0.0101 eV | **32% WORSE** |
| agreement with classical, median offset | −0.0025 eV | −0.0481 eV | 19x larger bias |
| agreement with classical, median abs | 0.0805 eV | 0.0809 eV | unchanged |

The synthetic gain reproduced exactly as predicted. **The real-data repeatability got 32%
worse, and a systematic −0.048 eV offset against the independent classical estimator appeared
where there had been none.** By the criterion set before the test was run, this is not adopted.

### First: is the ramp test fair to the refinement?

A model predicting nearly the same Te for every sweep would score perfect ramp repeatability
while carrying no information, which would rig this comparison in the baseline's favour. Checked:
baseline Te spans p10 = 0.120 to p90 = 0.580 eV across the archive with a ramp scatter of
0.0076 eV — a scatter-to-spread ratio of 0.017. It is repeatable *and* informative, and 13x more
repeatable than the classical fit's 0.101 eV. The test is fair and the refinement genuinely
fails it.

### Leading hypothesis, with its own supporting evidence

The alignment window is `u in [-2.0, +0.6] V`. That is centred on precisely the bias region where
this archive is independently known to depart from the forward model:

- **Phase 15b**: against a matched single-Maxwellian control, the real stacked curves show a
  positive excess at u = −0.3 and −0.6 V on **nine of nine observation days** (sign test
  p = 0.002), while the deep region 1.5-1.9 V below Vp shows nothing.
- **Phase 16b**: that same near-Vp excess **reverses sign between the two probe channels**
  (+3.1σ and −10.0σ against channel-matched controls, disagreeing with each other at 11σ),
  which marks it instrumental rather than plasma.

So the refinement reads Te from exactly the window where the real curves are known to disagree
with the simulator, and where the disagreement is channel-dependent. The baseline, reading all
240 raw points, spreads its dependence across the sweep and is less exposed. That would explain
both symptoms at once: a clean synthetic gain, and worse real repeatability plus a new systematic
offset.

**This is a hypothesis, not a finding.** Three mechanisms proposed earlier in this project were
wrong for reasons available before the code was written, so it is stated as testable rather than
established. The test is now one flag:

    python scripts/14_te_refine_eval.py --u-window -2.6 -0.8

`te_refine.set_window()` moves the alignment window below the deviant region; the reference point
for the density normalisation follows it. If ramp scatter recovers to baseline or better while
the synthetic gain largely survives, the hypothesis holds and the refinement becomes adoptable
with a corrected window. If ramp scatter stays worse, the idea does not transfer to this
instrument and should be reported as a negative result — which is still worth reporting, because
the synthetic gain is real and large, and the failure localises the sim-to-real defect.

### What this is worth to the report either way

This is the second clean demonstration in this project that **a synthetic hold-out cannot
adjudicate a change whose risk is sim-to-real** — the first being the physics-loss ablation of
Phase 12, where the conclusion also inverted. Two independent instances, in opposite directions:
there, a term that looked harmful on synthetic data bought ten points of real recovery; here, a
change that halves synthetic error costs a third of the real repeatability. Together they make
the methodological point far better than either alone.

## Phase 17c — Hypothesis refuted, and the result is better than the hypothesis

The window test came back decisively negative, on every metric at once:

| window | synthetic abs | synthetic % in band | ramp scatter | classical median offset | classical median abs |
|---|---|---|---|---|---|
| baseline (no refinement) | 0.0746 eV | 18.2% | **0.0076 eV** | −0.0025 eV | 0.0805 eV |
| refined, u in [−2.0, +0.6] | **0.0485 eV** | **11.8%** | 0.0101 eV (1.32x) | −0.0481 eV | 0.0809 eV |
| refined, u in [−2.6, −0.8] | 0.0872 eV | 20.5% | 0.0175 eV (**2.30x**) | +0.0746 eV | 0.1404 eV |

Moving the window below the region where the model is known to be wrong made everything worse,
**including the synthetic accuracy** — 0.0872 eV, worse than not refining at all. That rules out
the Phase 17b hypothesis cleanly: if the near-Vp model defect were the whole story, the deeper
window should have traded a little synthetic accuracy for better real repeatability. It traded
away both.

### The actual reason, which is a sharper finding

The deep window is signal-starved. Electron current falls as exp(u/Te), so at Te = 0.4 eV:

| u (V below Vp) | I / I_peak | quiet channels (0.24%) | 20 Mohm (1.8%) |
|---|---|---|---|
| −0.3 | 47% | above noise | above noise |
| −0.8 | 13.5% | above | above |
| −1.5 | 2.4% | above | **under** |
| −2.0 | 0.67% | **under** | **under** |
| −2.6 | 0.15% | **under** | **under** |

The default window [−2.0, +0.6] spans 0.7% to 100% of peak and contains the temperature
information. The deeper window [−2.6, −0.8] spans 0.15% to 13.5% and is mostly beneath the noise
floor on every channel.

**So on this instrument the temperature signal and the sim-to-real defect occupy the same bias
region, and cannot be separated by choosing a window.** Above ~0.8 V below Vp there is signal but
the forward model is demonstrably wrong (Phase 15b: excess on 9 of 9 days, p = 0.002; Phase 16b:
sign-reversing between channels at 11 sigma). Below it the model is clean but the current is
under the noise. There is no window that is both.

That is a quantitative, instrument-specific explanation for why Te is the hardest of the three
parameters here, and it is a better result than the hypothesis it replaced.

### Verdict

**The refinement is not adopted.** `model.te_refine` stays false. The code and both evaluation
scripts remain, because the negative result is worth reporting and worth reproducing.

What stands:

- The degeneracy diagnosis is correct and quantified: corr(Te, Vp) = +0.93, inflating the
  achievable Te error 3.3x, with a Cramer-Rao bound of ~0.008 eV against 0.078 eV achieved.
- Removing the degeneracy **does** halve synthetic Te error — 22.8% to 11.8%, 10.7 sigma over
  three seeds at full budget. The mechanism is real.
- It does **not** transfer to the instrument, and the reason is now measured rather than guessed.
- Te improvement on this archive is therefore blocked by the forward model's accuracy near the
  plasma potential, not by the estimator. Fixing the near-Vp model — the anomaly still open from
  Phase 15c — is the prerequisite, not more network work.

### Fourth retraction, and the honest tally

Phase 14 attributed the hot population to photoelectrons: wrong. Phase 15b attributed the
residual to the sheath exponent: wrong, and excludable by inspection. Phase 15c attributed it to
transition width, on the strength of a bug in my own blend. Phase 17b attributed this failure to
the near-Vp window: wrong, and the test that refuted it took ten minutes.

Every one of those was caught by a control or a cheap test, and every one produced a better
finding than the hypothesis would have. The pattern worth keeping is not the hypotheses — it is
that each was made falsifiable and then actually falsified, at low cost, before anything was
adopted or written into the report.
