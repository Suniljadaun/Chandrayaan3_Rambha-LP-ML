#!/usr/bin/env python3
"""
run_all.py — run the whole pipeline end-to-end (steps 1..7).

Usage:
  python scripts/run_all.py                 # full run incl. MAVEN transfer
  python scripts/run_all.py --no-maven      # skip step 4 (lunar-only model)
  python scripts/run_all.py --quick         # tiny config for a smoke test

Each step is a separate module run in-process; a failure stops the run with a clear message.
"""
import _bootstrap as B
import argparse
import importlib
import time


STEPS = [
    ("01_build_classical_baseline", "classical baseline"),
    ("02_generate_synthetic", "synthetic data"),
    ("03_train_pinn", "train PINN"),
    ("04_maven_pretrain", "MAVEN transfer"),
    ("05_apply_and_recover", "apply + recover"),
    ("06_validate_hop", "validate"),
    ("07_make_figures", "figures"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-maven", action="store_true", help="skip MAVEN transfer step")
    ap.add_argument("--two-population", action="store_true",
                    help="predict a cold and a hot electron population (5 parameters instead "
                         "of 3) and write to outputs_2pop/, leaving outputs/ untouched")
    ap.add_argument("--quick", action="store_true",
                    help="patch config for a fast smoke test (few files/epochs)")
    args = ap.parse_args()

    if args.quick or args.no_maven or args.two_population:
        # Monkey-patch the config loader so EVERY step (each calls load_config() itself)
        # sees the same adjusted config -- shrinking the workload for --quick, and, just as
        # important, recording that MAVEN did not run for --no-maven. Without this, skipping
        # step 04's import (below) has no effect on cfg["maven"]["enabled"], and downstream
        # steps that check that flag for checkpoint selection would never see it flip.
        from rambhalp import config as cfgmod
        _orig = cfgmod.load_config

        def _patched(path=None):
            c = _orig(path)
            if args.quick:
                c["data"]["max_files"] = 3
                c["synthetic"]["n_train"] = 3000
                c["synthetic"]["n_val"] = 500
                c["train"]["epochs"] = 8
                c["maven"]["pretrain_epochs"] = 5
                c["validate"]["mc_samples"] = 10
            if args.no_maven:
                c["maven"]["enabled"] = False
            if args.two_population:
                # Separate output tree: the three-parameter results in outputs/ are validated
                # and should not be overwritten by an experimental run.
                c["synthetic"]["two_population"] = True
                from pathlib import Path as _P
                base = _P(c["_solution_dir"]) / "outputs_2pop"
                for key, sub in (("outputs", ""), ("figures", "figures"),
                                 ("tables", "tables"), ("checkpoints", "checkpoints")):
                    q = base / sub if sub else base
                    q.mkdir(parents=True, exist_ok=True)
                    c["paths"]["_" + key + "_abs"] = q
            return c
        cfgmod.load_config = _patched
        if args.quick:
            print(">>> QUICK MODE: reduced files/epochs for a smoke test\n")
        if args.two_population:
            print(">>> TWO-POPULATION MODE: 5 parameters (Nc, Tc, Nh, Th, Vp), "
                  "writing to outputs_2pop/\n")
        if args.no_maven:
            print(">>> --no-maven: cfg['maven']['enabled']=False -- "
                  "checkpoint selection will use pinn.pt only\n")

    t0 = time.time()
    for mod_name, label in STEPS:
        if args.no_maven and mod_name.startswith("04"):
            print(f"\n### skipping {label} (--no-maven)\n")
            continue
        print(f"\n########## {label} ({mod_name}) ##########")
        mod = importlib.import_module(mod_name)
        mod.main()
    print(f"\nDONE in {time.time()-t0:.0f}s. See outputs/ for tables, checkpoints, figures.")


if __name__ == "__main__":
    main()
