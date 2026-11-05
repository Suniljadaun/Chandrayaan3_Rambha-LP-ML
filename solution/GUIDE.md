# GUIDE: run the RAMBHA-LP solution from zero to the reported results

This is the step-by-step walkthrough. It assumes nothing is set up. Every command says *what* it
does, *when* you run it and *why* it matters. Read it top to bottom once; after that, §3 is all you
need to reproduce the main results.

There are two ways to run the project, and you can mix them:

- **Laptop only (CPU).** Everything runs on a normal laptop. Training is the slow step.
- **Laptop + GPU (Kaggle / Colab).** Only *training* benefits from a GPU. Train there, download one
  checkpoint file, and run the rest locally (§7, §8).

> **Mental model.** Steps 1 to 7 are the main pipeline: steps 1 and 2 prepare data, step 3 trains
> the network, steps 5 to 7 apply it and produce the results. Step 4 (Mars pre-training) was tried
> and abandoned. Steps 8 to 18 are the experiments behind individual results in the report (§12).
> Every step writes to `outputs/` and the next step reads from there, so you can stop, inspect and
> resume anywhere.

---

## Table of contents

1. What you are building
2. Prerequisites and installation
3. The one-command run (start here)
4. What each step does, and what to check
5. Reading the outputs (the numbers in the report)
6. Design decisions you must be able to defend
7. Training on **Kaggle** (GPU)
8. Training on **Google Colab** (GPU)
9. The MAVEN (Mars) transfer stage, and why it was abandoned
10. Tuning, troubleshooting and common mistakes
11. Assembling the submission
12. The experiment scripts (steps 8 to 18)

---

## 1. What you are building

Chandrayaan-3's RAMBHA-LP is a spherical Langmuir probe on the Vikram lander. It sweeps its bias
voltage from -12 V to +12 V and back, and measures the current collected from the surrounding
lunar plasma. The *shape* of that current-voltage (I-V) curve encodes three numbers: electron
density **Ne**, electron temperature **Te** and plasma potential **Vp**.

The classical way to get them (fit a straight line to ln(I) below the plasma potential, then use
the saturation branch) works on clean sweeps and **fails** on noisy, faint or oddly shaped ones. On
this archive it fails on 13,266 of 29,862 sweeps (44%). This project trains a neural network on
simulated sweeps, with a loss that also checks the physics, to read the *whole* curve into
(Ne, Te, Vp). It recovers 5,115 of those failures (39%; 91% of the ones the instrument did not
clip), raising the usable part of the archive from 51% to 68%, with error bars, because there is no
ground truth for lunar plasma.

The thesis in one line: **the classical fit discards many sweeps; a physics-informed network
recovers most of them, and agrees with the classical fit where both work.**

---

## 2. Prerequisites and installation

You need Python 3.10 or newer:

```bash
python --version        # should print 3.10.x or higher
```

Create an isolated environment so this project's packages do not collide with anything else:

```bash
cd solution                       # the folder this GUIDE lives in
python -m venv .venv               # make a virtual environment named .venv
source .venv/bin/activate          # activate it   (Windows: .venv\Scripts\activate)
pip install -r requirements.txt    # numpy, pandas, matplotlib, torch, ...
```

(`environment.yml` does the same with conda.) The only large package is **PyTorch**; the CPU build
pip installs by default is fine on a laptop.

**Get the data.** Download the RAMBHA-LP archive from ISRO's PRADAN portal
(https://pradan.issdc.gov.in/ch3/), release `LTA_RAMBHA-LP_August2024_final`, and unzip it so that
this path exists next to `solution/`:

```
../Chandrayaan-3/ch3_rambha/LTA_RAMBHA-LP_August2024_final
```

That is the preset value of `data.archive_root` in `config.yaml`. If you put the archive elsewhere,
change that one line; no code changes are needed. The pipeline uses only the **rawB** level
(photoemission removed), so the `raw` and `rawA` day folders can be deleted to save space.

> **Why one config file?** Every path and every number the analysis depends on lives in
> `config.yaml`. When someone asks "what density range did you assume?", you point at the file, not
> at a number buried in a script. It is also what makes the runs reproducible.

Check the environment before a full run:

```bash
pytest -q tests/        # 27 pass, 1 skipped, 1 expected failure
```

---

## 3. The one-command run (start here)

```bash
python scripts/run_all.py --no-maven
```

This runs steps 1, 2, 3, 5, 6 and 7 in order and fills `outputs/`. It is the run behind every
number in the report: lunar-only model, checkpoint `pinn.pt`.

Run the smoke test first, the very first time:

```bash
python scripts/run_all.py --quick      # 3 files, few epochs: proves the setup works
```

If `--quick` finishes without error, launch the full run. **Always smoke-test before the long
run**: you find problems in seconds instead of after a long training.

> Do not drop `--no-maven`. Without it the pipeline also runs step 4, and step 5 then uses the Mars
> pre-trained model, which was about 15 times worse (§9).

To run the steps one at a time (useful while learning, so you can inspect each output), use the
`Makefile`:

```bash
make baseline   # step 1
make synth      # step 2
make train      # step 3
make apply      # step 5
make validate   # step 6
make figures    # step 7
make test       # the tests
make clean      # delete outputs/ to start fresh (also deletes pinn.pt: see §11)
```

(`make maven` and `make all` include step 4; avoid them for the reported results.)

---

## 4. What each step does, and what to check

**Step 1: `01_build_classical_baseline.py` (classical baseline).**
Reads every science file in the archive, uses the matching ops file to cut it into commanded
sweeps, splits each triangle (-12 to +12 to -12 V) into an up half and a down half, averages the
repeated samples at each 0.1 V step, and runs the classical fit on each of the 29,862 sweeps.
Writes `outputs/tables/classical_baseline.csv` (one row per sweep) and caches the sweeps in
`outputs/cache/sweeps.pkl` so later steps do not re-read the archive.
*Check:* it prints the archive-wide failure rate: **16,596 pass (55.6%), 13,266 fail (44.4%)**.
The failing set is what the network targets. If it reports 0 sweeps, `archive_root` is wrong.

**Step 2: `02_generate_synthetic.py` (synthetic data).**
Builds 40,000 training and 4,000 test sweeps from the OML forward model, with known (Ne, Te, Vp):
Ne log-uniform 2 to 1000 cm^-3, Te uniform 0.05 to 2 eV, Vp uniform -12 to +6 V. Each curve is
evaluated on the same 240-point voltage grid as the real sweeps, then given a small baseline tilt
and noise measured per probe channel. Saved to `outputs/cache/synth_train.npz` and
`synth_val.npz` (arrays X of curves and Y of true values).
*Check:* it prints the label ranges. The network can only invert curves inside these ranges, so
they must cover the real values; `tests/test_domain_overlap.py` checks this.

**Step 3: `03_train_pinn.py` (train).**
Trains the network (240 -> 256 -> 256 -> 128 -> 3, dropout 0.10) for 250 epochs with the
**physics-informed loss**: supervised error against the synthetic labels, plus 0.5 times a term
that pushes the predicted parameters back through the OML model and requires the rebuilt curve to
match the input. Writes `outputs/checkpoints/pinn.pt` and `outputs/tables/train_history.json`.
*Check:* the training and validation losses should fall together and level off.

**Step 4: `04_maven_pretrain.py` (MAVEN transfer). Not used.**
Pre-trains on Mars-like sweeps, then fine-tunes on the lunar set, saving `maven_transfer.pt`. It
made the model about 15 times worse and was abandoned (§9). `--no-maven` skips it.

**Step 5: `05_apply_and_recover.py` (apply and recover).**
Applies `pinn.pt` to every real sweep with 50 Monte-Carlo dropout passes (mean = answer, spread =
error bar), rebuilds each sweep through the OML model, and marks a sweep **recovered** if Ne is
1 to 100,000 cm^-3, Te is 0.03 to 5 eV, the rebuilt curve has R^2 of at least 0.90, and the density
error bar is below 0.5 dex. Writes `outputs/tables/ml_predictions.csv` and `recovery_summary.json`.
*Check:* `recovery_summary.json` holds the headline: **5,115 of 13,266 failures recovered
(38.6%); combined usable sweeps 20,262 (67.9%)**. A sweep the instrument clipped (9,074 of them,
30%) is unusable for both methods; see `config.yaml`, `recovery`.

**Step 6: `06_validate_hop.py` (validate).**
Four checks, since there is no ground truth:
(a) error on the 4,000 simulated test sweeps, where the truth is known;
(b) agreement with the classical fit on the 16,596 sweeps it handles (no systematic density
offset; individual sweeps differ by 4.4% in density and 25% in temperature; clipped sweeps left out);
(c) the **density trend** across the mission: the network is never told the date, yet its daily
medians rise smoothly (Spearman 0.78, p = 0.008, over 10 well-sampled days). The classical fit
shows the same rise, so this is a consistency check, not proof against an instrumental effect;
(d) **ramp consistency**: the up and down halves of each sweep measure the same plasma seconds
apart, so their disagreement (2.4% in density over 9,895 pairs) is an error estimate taken from
the data, not from the model. Writes `outputs/tables/validation.json`.

> The single-day "hop" test this replaced could not be answered from this archive: the hop on
> 2 Sep 2023 was recorded in a fixed-voltage mode with no sweeps, and the 3 Sep terminator session
> has 408 sweeps (3 recovered) against 2,596 on 2 Sep. The pre-hop and terminator medians are
> still reported, with their counts, as context only.

**Step 7: `07_make_figures.py` (figures).**
Renders F1 to F6 and `Fig_yield_ab.png` into `outputs/figures/` at 300 dpi.

---

## 5. Reading the outputs (the numbers in the report)

```
outputs/
├── tables/
│   ├── classical_baseline.csv       per-sweep classical Ne, Te, Vp, R^2, pass/fail + reason
│   ├── ml_predictions.csv           per-sweep network Ne, Te, Vp, error bars, rebuild R^2, recovered
│   ├── recovery_summary.json        * headline: recovered count and combined yield
│   ├── validation.json              * simulated error, classical agreement, density trend,
│   │                                   ramp consistency
│   ├── train_history.json           loss curves (figure F3)
│   ├── uncertainty_calibration.json are the error bars honest? (step 15)
│   ├── physics_ablation.json        effect of the physics loss weight (step 8)
│   ├── te_refine_eval.json          the rejected temperature refinement (step 14)
│   └── fisher_bound.json            theoretical limit on Te accuracy (step 16)
├── checkpoints/
│   └── pinn.pt                      the trained network behind the report
├── cache/                           sweeps.pkl, synth_train.npz, synth_val.npz
└── figures/
    ├── F1_example_sweep.png         a sweep with its classical fit
    ├── F2_failure_map.png           classical failure rate per day
    ├── F3_training_curves.png       training and validation loss
    ├── F4_synthetic_scatter.png     predicted vs true on the 4,000 test sweeps
    ├── F5_recovery.png              classical alone vs classical + network
    ├── F6_density_trend.png         * per-day recovered density; thin days drawn hollow
    └── Fig_yield_ab.png             * F2 and F5 side by side, as used in the report
```

Experiments 9 to 12 write to `outputs_2pop/` instead (§12).

**Quote numbers straight from the files** and never round a claim beyond what the file says. The
simulated-sweep error in `validation.json` is the *best case*; real accuracy is looser, so always
pair a recovered value with its error-bar column from `ml_predictions.csv`.

---

## 6. Design decisions you must be able to defend

Each is stated as *"I chose X over Y because ___; the cost is ___."*

**Forward model = OML for a sphere.** I chose Orbital-Motion-Limited theory because the Debye
length (about 22 cm at 450 cm^-3 and 0.4 eV) is about nine times the 2.5 cm probe radius, which
is the OML regime. Above Vp I use a sheath-expansion exponent of 0.40 fitted to real sweeps rather
than the textbook 1.0, because it fits them much better (median R^2 0.998 against 0.972). The cost:
one fixed exponent for every sweep, which Liu et al. (2023) show can bias density; it is the largest
known systematic and the top item of future work.

**Train on simulated, not real, labels.** There are no real labels for lunar plasma, so the network
learns from OML sweeps with known parameters, with ranges and noise chosen by measuring the real
archive. The cost: the model is only as right as the forward model. I mitigate this with the
physics term and with four checks on real data (§4, step 6).

**MLP, not a CNN.** The curve is short (240 points), smooth and on a common grid, so a fully
connected network trains quickly on CPU and is easy to explain. The cost: a CNN could exploit
local structure, but the gain here does not justify the added complexity.

**Physics-informed loss (supervised + rebuild).** Pure supervised learning can predict physically
inconsistent parameters on real curves unlike the training set. The rebuild term forces
self-consistency. Step 8 measured the trade over three runs per weight: it costs some accuracy on
simulated sweeps (Te 0.056 -> 0.075 eV) but cuts the disagreement between sweep halves by 30% and
removes a 5% density offset against the classical fit, so it stays. It does not change how many
sweeps are recovered.

**Recovery defined conservatively.** A sweep counts as recovered only if the parameters are
plausible, the rebuilt curve clears the same R^2 of 0.90 as the classical fit, and the density
error bar is bounded. The cost: borderline sweeps are under-counted, which is the honest side to
err on.

**MC dropout for uncertainty.** Dropout stays on at prediction time and 50 passes are averaged; the
spread is the error bar (Gal & Ghahramani 2016). Step 15 checked it: density and Vp bars are about
three times wider than needed (conservative), Te bars are about right. The cost: it measures the
network's own uncertainty, not errors in the physics model.

**MAVEN transfer frozen as future work.** Per the project's own Day-32 rule, the Mars stage was
tested, made the model about 15 times worse, and was dropped. The headline model is lunar-only.

---

## 7. Training on Kaggle (GPU)

Use this only if you want to retrain on a GPU. **Only training moves to Kaggle**; everything else
stays local.

1. **Generate the synthetic set locally once**, so you do not upload the archive:
   ```bash
   make synth        # writes outputs/cache/synth_train.npz and synth_val.npz
   ```
2. **Zip the code, config and synthetic cache** (not the raw archive):
   ```bash
   cd ..
   zip -r rambha_solution.zip solution/src solution/config.yaml \
       solution/outputs/cache/synth_train.npz solution/outputs/cache/synth_val.npz
   ```
3. Go to **kaggle.com > Create > New Notebook**.
4. In the right sidebar: **Notebook > Session options > Accelerator > GPU T4 x1**.
5. **Add Input > Upload > Dataset**, upload `rambha_solution.zip`. Kaggle mounts it under
   `/kaggle/input/<your-dataset-name>/`.
6. **File > Import Notebook**, choose `notebooks/kaggle_train_pinn.ipynb`.
7. **Run all**. The notebook finds the upload, confirms the GPU, trains and writes `pinn.pt` to
   `/kaggle/working/`.
   *Note:* the notebook sets `epochs = 200`. The reported model used 250 (`config.yaml`); change
   that line to 250 to reproduce it.
8. From the **Output** tab, download `pinn.pt` and `train_history.json`.
9. Put them in place locally and finish the pipeline:
   ```bash
   cp ~/Downloads/pinn.pt outputs/checkpoints/
   cp ~/Downloads/train_history.json outputs/tables/
   make apply && make validate && make figures
   ```

> **Why upload only the synthetic cache?** Training never touches the real archive; it learns from
> the simulated curves. The 1.6 GB archive is only needed locally, in steps 1 and 5.

---

## 8. Training on Google Colab (GPU)

1. Upload the `solution/` folder to Google Drive (e.g. `MyDrive/RAMBHA/solution`). Include
   `outputs/cache/synth_*.npz` if you generated it; otherwise the notebook regenerates it.
2. Open **colab.research.google.com > File > Upload notebook**, pick
   `notebooks/colab_train_pinn.ipynb`.
3. **Runtime > Change runtime type > Hardware accelerator > GPU > Save.**
4. Run the first cell to **mount Drive** and approve the popup.
5. Edit the `SOL = ...` line to your Drive path. As on Kaggle, change `epochs = 200` to 250 to
   match the reported model. Then **Runtime > Run all**.
6. The notebook saves `pinn.pt` into the Drive `solution/outputs/checkpoints/`.
7. Copy that file to your local `outputs/checkpoints/` and run
   `make apply && make validate && make figures`.

> **Colab vs Kaggle.** Colab is simplest if you already use Drive; its sessions time out and lose
> files not saved to Drive, which is why the notebook writes back there. Kaggle keeps uploaded
> datasets and gives a fixed weekly GPU budget. The code is identical either way.

### 8.3 Wiring in a *real* MAVEN granule (optional, advanced)
The MAVEN stage (§9) runs on a Mars-like simulated distribution. To use an actual MAVEN LPW L2
Langmuir sweep file instead, download one `.cdf` from the LASP MAVEN SDC public archive, read it
with `cdflib` (`pip install cdflib`), extract the voltage and current arrays, scale them with
`physics.to_input_np`, and pass them as `Xpt` in `notebooks/maven_pretrain.ipynb`. Given the
result in §9, this is not recommended without a new reason to expect it to help.

---

## 9. The MAVEN (Mars) transfer stage, and why it was abandoned

**Idea.** A Langmuir sweep at Mars (MAVEN's LPW instrument) is the same kind of measurement in a
denser, hotter plasma. Pre-training on Mars-like sweeps, then fine-tuning on the lunar set, is
transfer learning and was expected to sharpen the lunar model.

**What the code does.** `scripts/04_maven_pretrain.py` (and `notebooks/maven_pretrain.ipynb` for
GPU) checks whether the LASP MAVEN archive is reachable, then pre-trains on a simulated Mars-like
distribution (Ne 500 to 100,000 cm^-3, Te 0.1 to 5 eV), fine-tunes on the lunar set and saves
`maven_transfer.pt`. The data mode used is recorded in `outputs/tables/maven_mode.json`.

**Result.** It made the model about **15 times worse**. The likely reason: both datasets come from
the same forward model, so pre-training adds no new physics, only a density prior two to three
orders of magnitude away from the Moon's. Following the project's own gate ("a complete lunar
project beats a half-finished two-planet one"), it was frozen as future work. The report states
this as a negative result about this particular attempt, not a general claim about transfer
learning. Keep using `--no-maven`.

---

## 10. Tuning, troubleshooting and common mistakes

**Everything is in `config.yaml`.** The settings you are most likely to touch:
- `data.max_files`: set to a small number to iterate fast; `null` for the whole archive.
- `classical.min_r2`: how strict the classical pass bar is. It moves the failure count a lot, so
  always state it with the failure rate: 0.80 -> 71.2% of sweeps pass, 0.85 -> 69.6%,
  **0.90 -> 55.6% (used)**, 0.95 -> 38.7%.
- `synthetic.n_train`, `train.epochs`, `train.batch_size`: training budget. Bigger is slower and
  usually better.
- `model.w_physics`: weight of the physics term (0.5). See step 8 before changing it.
- `probe.sheath_exponent`: 0.40, fitted to real sweeps. Changing it changes every classical and
  network result.

**Common mistakes:**
1. **Wrong `archive_root`.** Symptom: "No sweeps parsed." Fix the one path in `config.yaml`.
2. **Skipping the smoke test.** Always run `--quick` first.
3. **Forgetting `--no-maven`.** Without it, step 5 uses the abandoned Mars model.
4. **Uploading the raw archive to Kaggle or Colab.** Training needs only the synthetic cache.
5. **Quoting a number the files do not support.** Only report what the JSON and CSV files say, and
   pair every recovered value with its error bar.
6. **Forgetting the sign convention.** Real sweeps arrive in either current polarity; the code
   flips them so electron current rises with voltage. Keep that step if you add a new data source.
7. **Changing code instead of config.** If you are editing a number in a `.py` file, it probably
   belongs in `config.yaml`.
8. **Treating the simulated-sweep error as the real accuracy.** It is the best case. Real accuracy
   is looser; say so.
9. **Running experiment scripts before the main run.** Steps 8 to 18 read the outputs of steps 1
   to 7, so run `run_all.py --no-maven` first.

---

## 11. Assembling the submission

1. **Freeze a clean run only if you have changed something**:
   `make clean && python scripts/run_all.py --no-maven`, so the tables and figures agree.
   Careful: `make clean` deletes `outputs/`, including the `pinn.pt` behind the report, and a
   retrained network will not reproduce every number exactly. Copy `outputs/` somewhere first. If
   nothing has changed, keep the existing outputs.
2. **Deliverables:**
   - the report and supplementary material in `../report/`
     (`RAMBHA-LP_Independent_Study_Report_SunilKumar.pdf/.docx`,
     `RAMBHA-LP_Supplementary_Material_SunilKumar.pdf/.docx`);
   - the `solution/` code with `config.yaml`, this GUIDE and the README;
   - `outputs/tables/`, `outputs/figures/` and `outputs/checkpoints/pinn.pt`;
   - a link to the raw data (`ch3_rambha.zip`, 1.6 GB), which is too large to attach.
   The packaged version of all this is `../submission/RAMBHA-LP_Submission_SunilKumar.zip`.
3. **Check the story holds:** the classical fit fails on 44% (F2), 57% of those because the
   instrument clipped the sweep -> the network recovers 91% of the unclipped failures (F5) ->
   checked four ways: simulated test sweeps, agreement with the classical fit,
   the mission density trend (F6) and up/down consistency (`validation.json`) -> stated with error
   bars, and with `classical.min_r2 = 0.90` given alongside the failure rate.
4. **Commit** so there is a timestamped record.

---

## 12. The experiment scripts (steps 8 to 18)

Each script answers one question in the report. All of them need a completed main run (§3).

| Script | Question | Writes | Report |
|---|---|---|---|
| `08_ablate_physics.py` | Does the physics loss term help? Retrains at weights 0, 0.1, 0.25, 0.5 | `tables/physics_ablation.json` | §3.3.1, Table 8 |
| `09_photoelectron_test.py` | Is the second population photoelectrons? **Superseded** | `outputs_2pop/` | none |
| `10_model_comparison.py` | Second population or baseline artefact? **Superseded** | `outputs_2pop/` | none |
| `11_local_te_diagnostic.py` | Local-Te profile against a matched single-temperature control | `outputs_2pop/tables/local_te_diagnostic.json` | §3.3.2 |
| `12_slope_curvature.py` | Does the slope change within a single sweep? | `outputs_2pop/tables/slope_curvature.json` | §3.3.2 |
| `13_vp_decoupling.py` | Does aligning on Vp halve the Te error on simulated data? | prints a table | Table 10 |
| `14_te_refine_eval.py` | Does that gain survive on real sweeps? (no) | `tables/te_refine_eval.json` | Table 11 |
| `15_uncertainty_calibration.py` | Are the MC-dropout error bars honest? | `tables/uncertainty_calibration.json` | §3.1.3 |
| `16_fisher_bound.py` | Best Te accuracy the noise allows (Cramér-Rao bound) | `tables/fisher_bound.json` | §3.3.3 |
| `17_sheath_exponent.py` | Does the data alone choose the sheath exponent 0.40? Is the 1.0 V fit window a fair choice? | `tables/sheath_exponent.json` | §1.1, §2.3 |
| `18_recovered_validation.py` | Are the recovered sweeps reliable? Twin halves, the classical test, clipping; draws F7 | `tables/recovered_validation.json`, `figures/F7_recovered_example.png` | §3.2.3, Figure 5 |

Exact commands for the report's numbers:

```bash
python scripts/13_vp_decoupling.py --n-synth 20000 --epochs 90   # Table 10 (slow: 3 seeds)
python scripts/14_te_refine_eval.py                              # Table 11 (default window)
python scripts/15_uncertainty_calibration.py
python scripts/16_fisher_bound.py
python scripts/17_sheath_exponent.py                             # §2.3 (about 3 minutes)
python scripts/18_recovered_validation.py                        # §3.2.3, Figure 5
```

Scripts 09 and 10 assumed a second electron population exists. Scripts 11 and 12 then showed that
the evidence for one came from averaging sweeps of different temperatures, so 09 and 10 are kept
only as a record. The supplementary material (Appendix A) lists what was withdrawn, and
`CODE_REVIEW.md` holds the full engineering log.
