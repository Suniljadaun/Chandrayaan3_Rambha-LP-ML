"""Shared bootstrap: make `import rambhalp` work and provide a tiny disk cache so scripts
can hand sweeps/arrays to each other without re-parsing the archive every time."""
import sys
import pickle
from pathlib import Path

# Put solution/src on the import path.
SOLUTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION / "src"))


def cache_dir(cfg):
    d = cfg["paths"]["_outputs_abs"] / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
