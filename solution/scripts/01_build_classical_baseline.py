#!/usr/bin/env python3
"""
Step 1 — Parse the whole archive and run the classical OML baseline on every sweep.

Outputs:
  outputs/tables/classical_baseline.csv   one row per sweep (params + failed + reason)
  outputs/cache/sweeps.pkl                 segmented sweeps, cached for later steps
  prints the archive-wide failure rate — the number Phase 1's deliverable is built on.
"""
import _bootstrap as B
import pandas as pd
from rambhalp.config import load_config, out
from rambhalp import preprocess, classical


def main():
    cfg = load_config()
    print("=" * 70)
    print("STEP 1: classical baseline over the RAMBHA-LP archive")
    print("=" * 70)

    sweeps = preprocess.load_all_sweeps(cfg, verbose=True)
    if not sweeps:
        raise SystemExit("No sweeps parsed — check config.yaml data.archive_root / level.")

    B.save_pickle(sweeps, B.cache_dir(cfg) / "sweeps.pkl")

    rows = classical.fit_all(sweeps, cfg)
    df = pd.DataFrame([r.as_row() for r in rows])
    df.to_csv(out(cfg, "tables", "classical_baseline.csv"), index=False)

    n = len(df)
    n_fail = int(df["failed"].sum())
    print("-" * 70)
    print(f"sweeps total          : {n}")
    print(f"classical SUCCEEDED   : {n - n_fail}  ({(n-n_fail)/n*100:.1f}%)")
    print(f"classical FAILED      : {n_fail}  ({n_fail/n*100:.1f}%)  <- ML target set")
    ok = df[~df["failed"]]
    if len(ok):
        print(f"median Te (ok)        : {ok['Te_eV'].median():.3f} eV")
        print(f"median Ne (ok)        : {ok['Ne_cc'].median():.0f} cm^-3")
    print("top failure reasons:")
    print(df[df["failed"]]["fail_reason"].str.split(";").str[0].value_counts().head(6)
          .to_string())
    print(f"\nwrote {out(cfg, 'tables', 'classical_baseline.csv')}")


if __name__ == "__main__":
    main()
