# RAMBHA-LP Physics-Informed ML Inversion — Complete Solution

**Author:** Sunil (M.Tech Independent Study, IIIT Hyderabad)
**Supervisor:** Dr. Rama Chandra Pillutla
**What this is:** A complete, runnable, submission-ready implementation of a physics-informed
machine-learning inversion of Chandrayaan-3 RAMBHA-LP Langmuir-probe I–V sweeps. It recovers
electron density (Ne), electron temperature (Te) and plasma potential (Vp) from sweeps —
including sweeps the classical OML fit fails on.

> This folder is the **safety-net deliverable**. It runs end-to-end and produces every number,
> table and figure the report needs. Your 50 day-scripts remain the *learning* track; this is
> the *submittable* track that does not depend on your live progress.

---

## The pipeline in one picture

```
RAMBHA-LP PDS4 archive (real lunar data, local)
        │
        ▼
[1] io_pds + preprocess ── parse raw/rawA/rawB, segment sweeps, bin, convert to current
        │
        ▼
[2] classical ──────────── OML baseline fit on every sweep → "where classical FAILS" map
        │
        ▼
[3] synthetic + physics ── generate labelled OML sweeps (Ne,Te,Vp known)
        │
        ▼
[4] model + train ──────── PINN inversion net trained with supervised + physics loss
        │                    (optionally MAVEN-pretrained — [4b] maven)
        ▼
[5] infer ──────────────── apply the net to the sweeps classical could NOT fit → recovered count
        │
        ▼
[6] validate ───────────── hop blind-test + uncertainty (honest, no lunar ground truth)
        │
        ▼
[7] figures ────────────── all report figures + tables
```

## Quick start (laptop, CPU is fine)

```bash
cd solution
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# edit config.yaml -> data.archive_root to your RAMBHA path (default already set)
python scripts/run_all.py            # runs the whole pipeline, writes outputs/
```

Everything lands in `outputs/` (tables, model checkpoint, figures). Nothing outside `solution/`
is modified.

## Where GPU (Kaggle / Colab) helps

Only steps **[4] PINN training** and **[4b] MAVEN pretrain** benefit from a GPU, and even those
run on a laptop in a few minutes for the default config. Use `notebooks/` when you want the
bigger/longer training runs. See `GUIDE.md` §7–§8 for the exact upload-train-download loop.

## Read this first

**`GUIDE.md`** is the hand-held, step-by-step walkthrough from a blank machine to a submitted
project — every command explained, every decision justified, with the Kaggle/Colab path spelled
out. If you read one file, read that one.

## Layout

```
solution/
├── README.md              you are here
├── GUIDE.md               ★ the detailed start-to-finish guide
├── requirements.txt       pip dependencies
├── environment.yml        conda alternative
├── config.yaml            all paths + hyperparameters (edit this, not the code)
├── Makefile               shortcuts (make baseline / synth / train / apply / figures / all)
├── src/rambhalp/          the library (importable package)
│   ├── config.py          loads config.yaml
│   ├── io_pds.py          PDS4 raw/rawA/rawB + ops + gain-table reader
│   ├── preprocess.py      sweep segmentation, binning, current conversion
│   ├── classical.py       classical OML baseline fit + failure classification
│   ├── physics.py         OML forward model + physics-informed loss
│   ├── synthetic.py       labelled synthetic sweep generator
│   ├── model.py           PyTorch inversion network (+ MC-dropout uncertainty)
│   ├── train.py           training loop
│   ├── maven.py           MAVEN download + pretrain + transfer (with fallback)
│   ├── infer.py           apply model to real sweeps, recover failing ones
│   ├── validate.py        hop blind-test + uncertainty audit
│   └── figures.py         all report figures + tables
├── scripts/               thin CLI wrappers you actually run
│   ├── 01_build_classical_baseline.py
│   ├── 02_generate_synthetic.py
│   ├── 03_train_pinn.py
│   ├── 04_maven_pretrain.py
│   ├── 05_apply_and_recover.py
│   ├── 06_validate_hop.py
│   ├── 07_make_figures.py
│   └── run_all.py         orchestrates 01→07
├── notebooks/             GPU training on Kaggle / Colab
│   ├── kaggle_train_pinn.ipynb
│   ├── colab_train_pinn.ipynb
│   └── maven_pretrain.ipynb
├── tests/                 sanity tests (pytest)
└── outputs/               created at runtime (git-ignored)
```

## Honesty note (carried from the project's own rules)

There is **no ground truth** for lunar near-surface plasma. Every recovered number is framed
with uncertainty; the model is validated against physics self-consistency, the classical fit
where it *does* work, and the Aug-26 lander "hop" as a natural blind test — not against a truth
we do not have. The code never claims a recovery it cannot show a number for.
