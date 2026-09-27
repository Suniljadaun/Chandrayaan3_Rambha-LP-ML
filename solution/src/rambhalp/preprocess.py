"""
preprocess.py — turn a raw science file into a list of clean, fixed-length I-V sweeps.

Steps:
  1. segment: use the ops sample-counts to slice the flat science array into sweeps
  2. bin:     average the many repeated samples at each bias voltage into a smooth curve
  3. resample: interpolate every sweep onto ONE common voltage grid (fixed length) so the
     network always sees the same-shaped input vector

The resampled current vector (grid_points long) is the model input. We keep the binned
curve too, because the classical fit is happier on the native binning.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

from . import io_pds


@dataclass
class Sweep:
    """One segmented I-V sweep with identity metadata."""
    date: str
    level: str
    timestamp: str
    idx: int                       # sweep number within the file
    V: np.ndarray                  # binned bias voltage (V)
    I: np.ndarray                  # binned current (A)
    adc_channel: int = -1
    probe_res: float = np.nan
    t_start: str = ""
    direction: str = ""            # "up" | "down" | "" (both ramps averaged together)
    segment: int = -1              # index of the ops row this ramp came from
    meta: dict = field(default_factory=dict)


def bin_sweep(V: np.ndarray, I: np.ndarray, step: float, min_per_bin: int):
    """Average repeated samples at each commanded voltage into a clean monotone-V curve."""
    lo, hi = np.floor(V.min()), np.ceil(V.max())
    edges = np.arange(lo, hi + step, step)
    idx = np.digitize(V, edges)
    Vb, Ib = [], []
    for b in np.unique(idx):
        m = idx == b
        if m.sum() >= min_per_bin:
            Vb.append(V[m].mean())
            Ib.append(I[m].mean())
    Vb, Ib = np.asarray(Vb), np.asarray(Ib)
    order = np.argsort(Vb)          # ensure ascending V for downstream interpolation
    return Vb[order], Ib[order]


def convert_output_voltage(V: np.ndarray, Vout: np.ndarray, gain_table: pd.DataFrame,
                           channel: str = "1MG1") -> np.ndarray:
    """Convert level-0 raw output voltage to current using the resistor/gain table.

    The table gives, per bias voltage, the transfer value for each resistor/gain channel.
    Current ≈ Vout / R_effective, where the table already folds in gain. We interpolate the
    channel curve onto the sweep's bias values and divide. (Only needed for level='raw';
    rawA/rawB deliver current directly, so most runs never call this.)
    """
    tv = np.interp(V, gain_table["bias"].to_numpy()[::-1],
                   gain_table[channel].to_numpy()[::-1])
    # Avoid divide-by-zero; the table values are transfer resistances in the used channel.
    tv = np.where(np.abs(tv) < 1e-30, np.nan, tv)
    return Vout / tv


def segment_file(sf: io_pds.SweepFile, cfg: dict, gain_table: pd.DataFrame | None = None
                 ) -> list[Sweep]:
    """Read one (sci, ops) pair and return its list of binned Sweeps."""
    pcfg = cfg["preprocess"]
    ops = io_pds.read_ops(sf.ops_path)
    sci = io_pds.read_science(sf.sci_path, sf.level)

    n_samps = ops["Number_of_samples"].astype(int).tolist()
    split = bool(pcfg.get("split_ramps", True))
    sweeps: list[Sweep] = []
    start = 0
    for i, n in enumerate(n_samps):
        if start + n > len(sci):
            break
        seg = sci.iloc[start:start + n]
        start += n
        V = seg["V"].to_numpy()
        if sf.level == "raw":
            if gain_table is None:
                continue                      # cannot convert without the table
            I = convert_output_voltage(V, seg["Vout"].to_numpy(), gain_table)
        else:
            I = seg["I"].to_numpy()
        ok = np.isfinite(V) & np.isfinite(I)
        V, I = V[ok], I[ok]
        if len(V) < 20:
            continue
        row = ops.iloc[i] if i < len(ops) else {}

        def _emit(Vp_, Ip_, sub: int, direction: str):
            """Bin one ramp and append it as a Sweep, if it survives the length guards."""
            if len(Vp_) < 20:
                return
            Vb, Ib = bin_sweep(Vp_, Ip_, pcfg["bias_step"], pcfg["min_samples_per_bin"])
            if len(Vb) < 20:
                return
            sweeps.append(Sweep(
                date=sf.date, level=sf.level, timestamp=sf.timestamp,
                idx=2 * i + sub, V=Vb, I=Ib, direction=direction, segment=i,
                adc_channel=int(row.get("ADC_channel", -1)) if hasattr(row, "get") else -1,
                probe_res=float(row.get("Probe_res(ohm)", np.nan)) if hasattr(row, "get") else np.nan,
                t_start=str(row.get("Start_time(UTC)", "")) if hasattr(row, "get") else "",
            ))

        # ------------------------------------------------------------------------------
        # Each ops row is one TRIANGLE ramp: the commanded bias runs -12 -> +12 -> -12
        # within a single segment (verified: argmin(V) = 0, argmax(V) mid-segment, exactly
        # one direction reversal, ~19,800 samples over 241 commanded levels). Because
        # bin_sweep averages by voltage, the previous code merged the rising and falling
        # ramps into ONE curve per segment, which (a) throws away half the measurements and
        # (b) silently averages away any up/down hysteresis instead of letting you measure
        # it. Splitting gives two independent sweeps per segment plus a free systematic
        # check -- validate.ramp_consistency compares the pair.
        #
        # idx = 2*i (rising) and 2*i + 1 (falling) so that (timestamp, idx) stays a unique
        # key for every join downstream. Set preprocess.split_ramps: false to restore the
        # old averaged behaviour.
        # ------------------------------------------------------------------------------
        if split:
            k = int(np.argmax(V))
            if k >= 20 and (len(V) - k) >= 20:
                _emit(V[:k + 1], I[:k + 1], 0, "up")
                _emit(V[k:], I[k:], 1, "down")
            else:
                _emit(V, I, 0, "single")      # not a triangle; keep as one sweep
        else:
            _emit(V, I, 0, "")
    return sweeps


def resample_to_grid(sweep: Sweep, cfg: dict) -> np.ndarray:
    """Interpolate a sweep's current onto the common fixed-length voltage grid.

    Returns a vector of length cfg['preprocess']['grid_points']. Out-of-range voltages are
    edge-clamped (the network learns to ignore flat tails). This vector is the model input.
    """
    pcfg = cfg["preprocess"]
    grid = np.linspace(pcfg["bias_min"], pcfg["bias_max"], pcfg["grid_points"])
    return np.interp(grid, sweep.V, sweep.I, left=sweep.I[0], right=sweep.I[-1])


def common_grid(cfg: dict) -> np.ndarray:
    """The single voltage grid used everywhere (model input axis, physics forward model)."""
    pcfg = cfg["preprocess"]
    return np.linspace(pcfg["bias_min"], pcfg["bias_max"], pcfg["grid_points"])


def load_all_sweeps(cfg: dict, verbose: bool = True) -> list[Sweep]:
    """Discover + segment every file at the configured level. The archive-wide sweep set."""
    root = cfg["data"]["_archive_abs"]
    level = cfg["data"]["level"]
    gain = io_pds.load_gain_table(root) if level == "raw" else None
    files = io_pds.discover_files(root, level=level, max_files=cfg["data"]["max_files"])
    if verbose:
        print(f"[preprocess] {len(files)} science files at level={level}")
    all_sweeps: list[Sweep] = []
    for k, sf in enumerate(files):
        try:
            all_sweeps.extend(segment_file(sf, cfg, gain))
        except Exception as e:                 # one bad file must not kill the archive run
            if verbose:
                print(f"  ! skip {sf.sci_path.name}: {e}")
        if verbose and (k + 1) % 20 == 0:
            print(f"  parsed {k+1}/{len(files)} files, {len(all_sweeps)} sweeps so far")
    if verbose:
        print(f"[preprocess] total sweeps: {len(all_sweeps)}")
    return all_sweeps
