# GUIDE — Run the RAMBHA-LP solution from zero to submission

This is the hand-held walkthrough. It assumes **nothing** is set up. Every command is explained:
*what* it does, *when* you run it, and *why* it matters. Read top to bottom the first time; after
that, §3 (the one-command run) is all you need day to day.

There are two ways to run this project, and you can mix them:

- **Laptop-only (CPU).** The whole pipeline runs on a normal laptop in a few minutes with the
  default settings. This is the simplest path and is enough to submit.
- **Laptop + GPU (Kaggle / Colab).** Only the *training* steps benefit from a GPU. You train on
  Kaggle or Colab, download one checkpoint file, and run the rest locally. Use this when you want
  the bigger, longer training runs for the final report.

> **Mental model.** The project is a 7-step pipeline. Steps 1–2 prepare data, steps 3–4 train the
> model, steps 5–7 apply it and make the results. The only heavy step is training (3, 4). Every
> step writes its output to `outputs/` and the next step reads it — so you can stop, inspect, and
> resume anywhere.

---

## Table of contents

1. What you are building (the 60-second version)
2. Prerequisites and installation
3. The one-command run (start here)
4. What each step does, and what to check
5. Reading the outputs (the numbers that go in the report)
6. Design decisions you must be able to defend
7. Training on **Kaggle** (GPU) — exact clicks
8. Training on **Google Colab** (GPU) — exact clicks
9. The MAVEN (Mars) transfer stage
10. Tuning, troubleshooting, and common mistakes
11. Assembling the submission
12. A realistic 2-week schedule

---

## 1. What you are building (the 60-second version)

Chandrayaan-3's RAMBHA-LP is a Langmuir probe: it sweeps a bias voltage from −12 V to +12 V and
measures the current the surrounding lunar plasma pushes onto it. The *shape* of that current-vs-
voltage (I–V) curve encodes three numbers: electron density **Ne**, electron temperature **Te**,
and plasma potential **Vp**.

The classical way to get those numbers (fit an exponential to part of the curve) works on clean
sweeps and **fails** on noisy, low-density, or oddly-shaped ones. This project trains a small
neural network to invert the *whole* curve into (Ne, Te, Vp), constrained by the actual Langmuir
physics, so it can recover sweeps the classical method throws away — with honest uncertainty,
because there is no ground truth for lunar plasma.

That is the entire thesis: **classical fails on some sweeps; a physics-informed network recovers a
defensible fraction of them.** Everything in `solution/` exists to produce that claim with receipts.

---

## 2. Prerequisites and installation

You need Python 3.10 or newer. Check:

```bash
python --version        # should print 3.10.x or higher
```

Create an isolated environment so this project's packages don't collide with anything else
(this is standard hygiene — a fresh sandbox per project):

```bash
cd solution                       # the folder this GUIDE lives in
python -m venv .venv               # make a virtual environment named .venv
source .venv/bin/activate          # activate it   (Windows: .venv\Scripts\activate)
pip install -r requirements.txt    # install numpy, pandas, matplotlib, torch, ...
```

`requirements.txt` pins the libraries. The only large one is **PyTorch** (`torch`); on a laptop the
CPU build is fine and is what pip installs by default. If the install is slow, that's the torch
download — let it finish once; it is cached afterwards.

**Point the code at your data.** Open `config.yaml` and check `data.archive_root`. It is preset to

```
archive_root: "../Chandrayaan-3/ch3_rambha/LTA_RAMBHA-LP_August2024_final"
```

which is correct if `solution/` sits next to your `Chandrayaan-3/` folder (it does). If you ever
move the archive, update this one line — no code changes needed.

> **Why one config file?** So that every path and every hyperparameter lives in exactly one place.
> When a reviewer asks "what density range did you assume?" you point at `config.yaml`, not at a
> number buried in a script. This is also what makes runs reproducible.

Optional: confirm the environment is healthy before a full run:

```bash
pytest -q tests/        # physics + IO sanity checks; the IO test skips if data is missing
```

---

## 3. The one-command run (start here)

```bash
python scripts/run_all.py
```

That runs steps 1→7 in order and fills `outputs/`. On a laptop it takes a few minutes. When it
finishes you'll have tables, a trained model, validation numbers, and six figures.

Two useful variants:

```bash
python scripts/run_all.py --quick      # tiny smoke test (3 files, few epochs) — proves it works
python scripts/run_all.py --no-maven   # skip the Mars-transfer step (lunar-only model)
```

Run `--quick` first the very first time. If it completes without error, your setup is correct and
you can launch the full run with confidence. **Always smoke-test before the long run** — you find
problems in 30 seconds instead of 5 minutes.

If you'd rather run steps one at a time (recommended while you're learning, so you can inspect each
output), use the `Makefile`:

```bash
make baseline   # step 1
make synth      # step 2
make train      # step 3
make maven      # step 4  (optional)
make apply      # step 5
make validate   # step 6
make figures    # step 7
# or simply:
make all
```

---

## 4. What each step does, and what to check

**Step 1 — `01_build_classical_baseline.py` (classical baseline).**
Parses every science file in the archive, slices each into individual sweeps using the ops
metadata, bins the repeated samples into a clean curve, and runs the classical Langmuir fit on
each. Writes `outputs/tables/classical_baseline.csv` (one row per sweep) and caches the parsed
sweeps to `outputs/cache/sweeps.pkl` so later steps don't re-parse.
*Check:* it prints the archive-wide **failure rate**. Expect the classical method to succeed on
roughly a third of sweeps (29% on the current archive) and fail on the rest — that failing set is
precisely what the ML model
will target. If it says 0 sweeps, your `archive_root` is wrong.

**Step 2 — `02_generate_synthetic.py` (synthetic data).**
Builds tens of thousands of *labelled* sweeps from the OML forward model — curves for which we
know the true (Ne, Te, Vp) because we chose them. Saved to `outputs/cache/synth_*.npz`.
*Check:* it prints the label ranges. This is the network's textbook: it can only learn to invert
curves whose parameters fall in these ranges, so they must bracket the real lunar values.

**Step 3 — `03_train_pinn.py` (train).**
Trains the network with the **physics-informed loss**: part of the loss is ordinary supervised
error against the synthetic labels; the other part pushes the network's predicted parameters back
through the OML forward model and demands the reconstructed curve match the input. Writes
`outputs/checkpoints/pinn.pt` and `outputs/tables/train_history.json`.
*Check:* the loss should fall steadily and the validation loss should track it (not blow up). The
history feeds figure F3.

**Step 4 — `04_maven_pretrain.py` (MAVEN transfer, optional).**
Pretrains on Mars-like (denser, hotter) sweeps, then fine-tunes on the lunar synthetic set. Saves
`outputs/checkpoints/maven_transfer.pt`. If `maven.enabled` is false, it cleanly no-ops.
*Check:* it records which data mode it used (real-archive-reachable vs MAVEN-like synthetic) in
`outputs/tables/maven_mode.json`. See §9.

**Step 5 — `05_apply_and_recover.py` (apply + recover).**
Loads the best available checkpoint (transfer if present, else the plain PINN), applies it to every
*real* sweep, reconstructs each through the physics model to score the fit, and counts how many
previously-**failed** sweeps are now recovered (plausible params + good reconstruction + tight
uncertainty). Writes `outputs/tables/ml_predictions.csv` and `outputs/tables/recovery_summary.json`.
*Check:* `recovery_summary.json` holds your headline number — how many of the classical failures
the model recovered.

**Step 6 — `06_validate_hop.py` (validate).**
Three honest checks, since there's no ground truth: (a) error on a synthetic hold-out where the
truth is known; (b) agreement with the classical method on sweeps it *did* fit; (c) the **density
trend** across the mission — the model is never told the date, so a smooth, monotonic day-to-day
density curve is evidence it learned physics rather than a dataset artefact; (d) **ramp
consistency** — each commanded sweep is a triangle, so its rising and falling halves measure the
same plasma seconds apart, and their disagreement is a systematic error bar derived from the data
rather than from the model. Writes `outputs/tables/validation.json`.

> The single-day "hop" test this replaced was not answerable from this archive: the terminator
> session (3 Sep 2023) holds ~200 sweeps against ~1300 the day before, so a couple of dozen
> recovered sweeps were being asked to settle a physical prediction. Its pre-hop/terminator
> numbers are still reported, as descriptive context with their counts attached, never as a
> pass/fail.

**Step 7 — `07_make_figures.py` (figures).**
Renders F1–F6 into `outputs/figures/`. These are report-ready.

---

## 5. Reading the outputs (the numbers that go in the report)

After a full run, `outputs/` contains:

```
outputs/
├── tables/
│   ├── classical_baseline.csv     per-sweep classical params + failed + reason
│   ├── ml_predictions.csv         per-sweep ML params + uncertainty + reconstruction R^2
│   ├── recovery_summary.json      ★ headline: recovered count + combined yield
│   ├── validation.json            ★ synthetic error, classical agreement, density trend,
│   │                                 ramp consistency
│   ├── train_history.json         loss curves
│   └── maven_mode.json            which MAVEN data mode was used
├── checkpoints/
│   ├── pinn.pt                    lunar-only model
│   └── maven_transfer.pt          Mars-pretrained model (if step 4 ran)
└── figures/
    ├── F1_example_sweep.png       a sweep + classical fit
    ├── F2_failure_map.png         classical failure rate per day
    ├── F3_training_curves.png     physics-informed training
    ├── F4_synthetic_scatter.png   predicted vs true (best-case accuracy)
    ├── F5_recovery.png            ★ classical vs classical+ML yield
    └── F6_density_trend.png         ★ per-day recovered density; thin days drawn hollow
```

The two starred JSONs and the two starred figures ARE your results section. Quote numbers from the
JSONs directly — never round a claim beyond what the file says.

**How to talk about the numbers honestly.** "Recovered N of M failed sweeps" is only true if
`recovery_summary.json` says so. The synthetic hold-out error in `validation.json` is your *best
case* (physics matching physics); real recoveries will be looser. Always pair a recovered value
with its uncertainty column from `ml_predictions.csv`.

---

## 6. Design decisions you must be able to defend

These are the choices a supervisor or reviewer will probe. Each is stated as *"I chose X over Y
because ___; the cost is ___."*

**Forward model = OML for a sphere.** I chose Orbit-Motion-Limited theory over thin-sheath models
because RAMBHA-LP is a small spherical probe in a tenuous plasma where the Debye length is large
compared to the probe — the OML regime. The cost: OML overestimates current when that assumption
weakens (denser plasma), which is one reason I report uncertainty rather than point values.

**Train on synthetic, not on real labels.** There are no real labels for lunar plasma, so I teach
the network on physics I trust (OML sweeps with known parameters). The cost: the model is only as
right as the forward model; I mitigate this with the physics-reconstruction loss (the prediction
must rebuild the *observed* curve) and by validating on the mission-wide density trend.

**MLP, not a CNN.** The curve is short (240 points), smooth, and already on a common grid, so a
fully-connected network trains in minutes on CPU and is trivial to explain. The cost: a CNN could
exploit local structure, but the gain is marginal here and the added complexity is not worth
defending.

**Physics-informed loss (supervised + reconstruction).** Pure supervised learning would fit the
labels but could predict physically inconsistent parameters on out-of-distribution real curves. The
reconstruction term forces self-consistency and gives the loss *teeth on unlabelled real data*. The
cost: a second loss weight (`w_physics`) to tune.

**Recovery defined conservatively.** A sweep counts as "recovered" only if the predicted params are
in physically plausible bands, the reconstruction R² clears the same threshold as the classical
fit, and the MC-dropout uncertainty is bounded. The cost: I under-count borderline recoveries — but
an under-count is the honest side to err on.

**MC-dropout for uncertainty.** I keep dropout active at inference and average many passes; the
spread is the uncertainty. It's cheap and defensible. The cost: it is an approximation to a full
Bayesian treatment, which I note as future work.

**MAVEN transfer is optional and gated.** Per the project's own Day-32 rule, if the Mars transfer
fights the pipeline it is frozen as future work rather than forced. The cost: the headline model may
be lunar-only — which the project explicitly accepts ("a complete lunar project beats a half-
finished two-planet one").

---

## 7. Training on Kaggle (GPU) — exact clicks

Use this when you want a longer/bigger training run than the laptop default. **Only training moves
to Kaggle**; everything else stays local.

1. **Generate the synthetic set locally once** so you don't upload the big archive:
   ```bash
   make synth        # writes outputs/cache/synth_train.npz and synth_val.npz
   ```
2. **Zip the project** (code + config + the synthetic cache — not the raw archive):
   ```bash
   cd ..
   zip -r rambha_solution.zip solution/src solution/config.yaml \
       solution/outputs/cache/synth_train.npz solution/outputs/cache/synth_val.npz
   ```
3. Go to **kaggle.com ▸ Create ▸ New Notebook**.
4. In the right sidebar: **Notebook ▸ Session options ▸ Accelerator ▸ GPU T4 x1**. (Free, ~30 h/wk.)
5. **Add Input ▸ Upload ▸ Dataset**, upload `rambha_solution.zip`. Kaggle mounts it under
   `/kaggle/input/<your-dataset-name>/`.
6. **File ▸ Import Notebook**, choose `notebooks/kaggle_train_pinn.ipynb` (or paste its cells).
7. **Run all**. The notebook finds your uploaded folder, confirms the GPU, loads the synthetic
   cache, trains 200 epochs, and writes `pinn.pt` to `/kaggle/working/`.
8. Open the **Output** tab, **download `pinn.pt`** and `train_history.json`.
9. Locally, put them in place and finish the pipeline:
   ```bash
   cp ~/Downloads/pinn.pt outputs/checkpoints/
   cp ~/Downloads/train_history.json outputs/tables/
   make apply && make validate && make figures
   ```

> **Why upload only the synthetic cache?** Training never touches the raw archive — it learns from
> the synthetic curves. Keeping the ~800 MB archive off Kaggle makes the upload trivial and the run
> reproducible. The real archive is only needed locally in steps 1 and 5.

---

## 8. Training on Google Colab (GPU) — exact clicks

1. Upload the `solution/` folder to your Google Drive (e.g. `MyDrive/RAMBHA/solution`). Include
   `outputs/cache/synth_*.npz` if you generated it (`make synth`); otherwise the notebook
   regenerates it on the Colab box.
2. Open **colab.research.google.com ▸ File ▸ Upload notebook**, pick
   `notebooks/colab_train_pinn.ipynb`.
3. **Runtime ▸ Change runtime type ▸ Hardware accelerator ▸ GPU ▸ Save.**
4. Run the first cell to **mount Drive**; approve the popup.
5. Edit the `SOL = ...` line to your Drive path, then **Runtime ▸ Run all**.
6. The notebook saves `pinn.pt` straight back into the Drive `solution/outputs/checkpoints/`.
7. Sync/download that file to your local `outputs/checkpoints/`, then locally run
   `make apply && make validate && make figures`.

> **Colab vs Kaggle.** Colab is simplest if you already live in Drive; its sessions time out and
> lose files not saved to Drive, which is why the notebook writes back to Drive. Kaggle keeps
> uploaded datasets persistent and gives a fixed weekly GPU budget. Either is fine; the code is
> identical — only the "where are my files" plumbing differs.

### 8.3 Wiring in a *real* MAVEN granule (optional, advanced)
The MAVEN stage (see §9) is fully runnable today using a physics-faithful Mars-like distribution and
a live reachability probe of the LASP archive. If you want to train on an actual MAVEN LPW L2
Langmuir I–V granule, download one `.cdf` from the LASP MAVEN SDC public tree, read it with `cdflib`
(`pip install cdflib`), extract the sweep voltage/current arrays, scale them with
`physics.to_input_np`, and pass them as `Xpt` in `notebooks/maven_pretrain.ipynb`. The rest of the
transfer code is unchanged. This is deliberately isolated so the core project never depends on an
external download succeeding.

---

## 9. The MAVEN (Mars) transfer stage

**Idea.** A Langmuir sweep on Mars (MAVEN's LPW instrument) is the same *kind* of measurement as on
the Moon, just in a denser, hotter plasma. Learning the general curve→parameter map on abundant
Mars-like data first, then fine-tuning on the lunar synthetic set, is transfer learning — it can
sharpen the lunar model.

**How this solution handles it.** `scripts/04_maven_pretrain.py` (and `notebooks/maven_pretrain.
ipynb` for GPU) pretrains on Mars-range sweeps and fine-tunes on lunar ones, saving
`maven_transfer.pt`. It first *probes* the LASP MAVEN archive for reachability, then trains on the
physics-faithful MAVEN-like distribution; §8.3 shows how to swap in a literal granule. The mode used
is recorded in `outputs/tables/maven_mode.json` so the report states exactly what was done.

**The gate (from your own plan).** If MAVEN transfer does not clearly help — check whether
`maven_transfer.pt` beats `pinn.pt` on the synthetic hold-out and the density trend — **freeze it as
future work** by setting `maven.enabled: false` in `config.yaml` and shipping the lunar-only model.
A complete lunar project beats a half-finished two-planet one. This is a legitimate, pre-planned
outcome, not a failure.

---

## 10. Tuning, troubleshooting, and common mistakes

**Everything is in `config.yaml`.** The knobs you'll actually touch:
- `data.level` — `rawB` (photoemission removed, science-ready; the default and correct choice),
  `rawA` (photoemission present), or `raw` (needs the gain-table conversion, handled automatically).
- `data.max_files` — set to a small number to iterate fast; `null` for the whole archive.
- `classical.min_r2` — the strictness of the classical success bar. Raising it flags more sweeps as
  failures (more ML targets); lowering it credits classical with more. Report the value you used.
- `synthetic.n_train`, `train.epochs`, `train.batch_size` — training budget. Bigger = better and
  slower; the GPU notebooks already bump these up.
- `model.w_physics` — weight of the physics-reconstruction loss. Higher = more physically
  self-consistent, but too high starves the supervised term.

**Common mistakes (learn from these):**
1. **Wrong `archive_root`.** Symptom: "No sweeps parsed." Fix the one path in `config.yaml`.
2. **Running the full pipeline before the smoke test.** Always `--quick` first.
3. **Uploading the raw archive to Kaggle/Colab.** Unnecessary — training only needs the synthetic
   cache. Upload that, not 800 MB of CSVs.
4. **Quoting a recovery number the reconstruction doesn't support.** Only report what
   `recovery_summary.json` says; pair every recovered value with its uncertainty column.
5. **Forgetting the sign convention.** Real sweeps arrive in either current polarity; the code
   auto-orients them. If you add a new data source, keep that orientation step.
6. **Changing code instead of config.** If you find yourself editing a number in a `.py` file, it
   probably belongs in `config.yaml`. Keep the code fixed and the config the record of your choices.
7. **Comparing `pinn.pt` and `maven_transfer.pt` by eye.** Use the synthetic hold-out error and the
   density-trend and ramp-consistency numbers in `validation.json` — numbers, not impressions.
8. **Treating the synthetic hold-out error as the real accuracy.** It's the best case (physics vs
   physics). Real recoveries are looser; say so.

---

## 11. Assembling the submission

A complete submission is: the code, the results it produces, and the report that explains them.

1. **Freeze a clean run.** `make clean && python scripts/run_all.py`. This regenerates `outputs/`
   from scratch so the tables and figures are mutually consistent.
2. **Collect the deliverables:**
   - the `solution/` folder (code + `config.yaml` + this GUIDE + README),
   - `outputs/figures/F1..F6.png`,
   - `outputs/tables/recovery_summary.json` and `validation.json` (your results section),
   - your written report (`IS_Project_Report_RAMBHA-LP.docx/.pdf` in the parent folder) updated with
     the figures and the numbers from the JSONs.
3. **Sanity-check the story:** classical fails on X% (F2) → the network recovers N of them (F5) →
   validated four ways: synthetic hold-out, agreement with classical where it works, the mission
   density trend (F6), and up/down ramp consistency (`validation.json`) → stated with uncertainty.
   Quote `classical.min_r2` alongside the failure rate: it moves the denominator a lot (0.80 →
   42% classical success, 0.90 → 29%, 0.95 → 15%), so the threshold must be stated, not buried.
4. **Commit it** so there's a timestamped record:
   ```bash
   cd ..
   git add solution && git commit -m "Complete RAMBHA-LP solution: run_all pipeline + guide"
   ```
5. If the department wants a single archive, zip `solution/` plus the figures and the report.

You can submit the **laptop-only** run as-is. The GPU path only makes the training longer/bigger for
a stronger final model; it does not change what you submit or how you defend it.

---

## 12. A realistic 2-week schedule

This is the safety-net timeline — it gets you to a submittable state without waiting on the live
YouTube learning track.

- **Day 1.** Install (§2), run `--quick`, then a full laptop run. Read the numbers in
  `recovery_summary.json` and `validation.json`. You now have a complete draft result.
- **Days 2–3.** Read §4 and §6 until you can explain each step and each design choice in your own
  words. Open `classical_baseline.csv` and `ml_predictions.csv` and look at real rows.
- **Days 4–5.** Do one GPU training run (Kaggle §7 or Colab §8) with the bumped-up epochs; swap in
  the better checkpoint. Compare `pinn.pt` vs the GPU model on the hold-out.
- **Days 6–7.** Run the MAVEN stage (§9). Decide FINALIZE or FREEZE from the numbers; set
  `maven.enabled` accordingly.
- **Days 8–10.** Fold F1–F6 and the JSON numbers into the report. Write the honesty/uncertainty
  paragraph. Tune `classical.min_r2` if you want the failure/recovery framing to be cleaner, and
  re-run.
- **Days 11–12.** Mock-defend using §6 as your question bank. For each design choice, say the
  "I chose X over Y because…; the cost is…" sentence aloud.
- **Days 13–14.** Freeze a clean run (§11), assemble the submission, commit, and stop.

That is the whole project, start to finish. The code is the safety net; your live-taught
understanding is the goal. Both are here.
