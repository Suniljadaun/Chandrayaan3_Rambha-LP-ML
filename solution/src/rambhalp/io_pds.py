"""
io_pds.py — read the RAMBHA-LP PDS4 archive.

The archive layout (per the mission readme) is:

    <archive_root>/data/raw/
        20230830/        level-0   science: (Bias_voltage, Output_voltage)  + ops
        20230830_rawA/   level-0A  science: (V, I) amps, photoemission present
        20230830_rawB/   level-0B  science: (V, I) amps, photoemission removed  <- best
        ...

Each science file `*_sci_*.csv` has a matching `*_ops_*.csv` (same timestamp) that lists,
per commanded sweep, how many samples it contains plus the gain/timing metadata. We use
those sample counts to slice the flat science file back into individual sweeps.

For the level-0 `raw` product the science column is a *raw output voltage* that must be
turned into a current using the Offset_Resistor_Gain table. rawA/rawB already give current
in amps, so most of the pipeline just uses rawB.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd

# science-file timestamp, e.g. ch3_rlp_nrB_sci_20230826T145357736.csv -> 20230826T145357736
_TS_RE = re.compile(r"_sci_(\d{8}T\d{9})")


@dataclass
class SweepFile:
    """One (science, ops) pair on disk, tagged with its date and level."""
    date: str            # yyyymmdd
    level: str           # raw | rawA | rawB
    timestamp: str       # yyyymmddThhmmssSSS
    sci_path: Path
    ops_path: Path


def _level_suffix(level: str) -> str:
    """Directory suffix for a processing level: '' for raw, '_rawA', '_rawB'."""
    return {"raw": "", "rawA": "_rawA", "rawB": "_rawB"}[level]


def discover_files(archive_root: Path, level: str = "rawB", max_files: int | None = None
                   ) -> list[SweepFile]:
    """Walk the archive and return all (sci, ops) pairs for the requested level.

    We match a science file to its ops file by identical timestamp, which is robust to the
    'nr' / 'nrA' / 'nrB' infix differing between products.
    """
    raw_dir = Path(archive_root) / "data" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Could not find {raw_dir}. Check data.archive_root in config.yaml — it must "
            f"point at the folder that contains data/raw/."
        )
    suffix = _level_suffix(level)
    out: list[SweepFile] = []
    # Day folders for this level look like '20230826' (raw) or '20230826_rawB'.
    for day_dir in sorted(raw_dir.glob("[0-9]" * 8 + suffix)):
        # Guard: the bare 8-digit glob also catches rawA/rawB dirs; skip mismatches.
        name = day_dir.name
        if suffix == "" and ("_rawA" in name or "_rawB" in name):
            continue
        if suffix and not name.endswith(suffix):
            continue
        date = name[:8]
        # index ops files by timestamp for quick pairing
        ops_by_ts = {}
        for ops in day_dir.glob("*_ops_*.csv"):
            m = re.search(r"_ops_(\d{8}T\d{9})", ops.name)
            if m:
                ops_by_ts[m.group(1)] = ops
        for sci in sorted(day_dir.glob("*_sci_*.csv")):
            m = _TS_RE.search(sci.name)
            if not m:
                continue
            ts = m.group(1)
            ops = ops_by_ts.get(ts)
            if ops is None:
                continue  # no matching ops -> cannot segment; skip
            out.append(SweepFile(date=date, level=level, timestamp=ts,
                                  sci_path=sci, ops_path=ops))
    if max_files is not None:
        out = out[:max_files]
    return out


def read_ops(ops_path: Path) -> pd.DataFrame:
    """Read an ops (metadata) CSV; strip whitespace from headers and string cells."""
    ops = pd.read_csv(ops_path, skipinitialspace=True)
    ops.columns = [c.strip() for c in ops.columns]
    return ops


def read_science(sci_path: Path, level: str) -> pd.DataFrame:
    """Read a science CSV into columns V (volts) and I (amps).

    rawA / rawB store (V, I) directly. `raw` stores (Bias_voltage, Output_voltage); we read
    it as (V, Vout) and leave current conversion to `preprocess.convert_output_voltage`.
    """
    if level in ("rawA", "rawB"):
        df = pd.read_csv(sci_path, names=["V", "I"], skiprows=1)
        df["V"] = pd.to_numeric(df["V"], errors="coerce")
        df["I"] = pd.to_numeric(df["I"], errors="coerce")
    else:  # raw / level-0
        df = pd.read_csv(sci_path, names=["V", "Vout"], skiprows=1)
        df["V"] = pd.to_numeric(df["V"], errors="coerce")
        df["Vout"] = pd.to_numeric(df["Vout"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def load_gain_table(archive_root: Path) -> pd.DataFrame | None:
    """Load Miscellaneous/Offset_Resistor_Gain.txt (used only for level-0 `raw`).

    The file has interleaved (bias, value) column pairs for six resistor/gain channels.
    We parse it into a tidy DataFrame with columns: bias, 50kG1, 1MG1, 20MG1, 50kG2,
    1MG2, 20MG2. Returns None if the file is absent (rawA/rawB do not need it).
    """
    p = Path(archive_root) / "Miscellaneous" / "Offset_Resistor_Gain.txt"
    if not p.exists():
        return None
    chans = ["50kG1", "1MG1", "20MG1", "50kG2", "1MG2", "20MG2"]
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = re.split(r"\s+", line)
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            # layout: bias, v1, bias, v2, bias, v3, ... -> take bias once + the 6 values
            if len(vals) >= 12:
                bias = vals[0]
                channel_vals = [vals[1], vals[3], vals[5], vals[7], vals[9], vals[11]]
                rows.append([bias] + channel_vals)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["bias"] + chans)
