"""IO/preprocess sanity against the real archive (skips gracefully if data is absent)."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rambhalp.config import load_config
from rambhalp import io_pds, preprocess


def test_discover_and_segment():
    cfg = load_config()
    root = cfg["data"]["_archive_abs"]
    if not (root / "data" / "raw").exists():
        pytest.skip("archive not found — set data.archive_root in config.yaml")
    files = io_pds.discover_files(root, level=cfg["data"]["level"], max_files=1)
    assert len(files) >= 1
    sweeps = preprocess.segment_file(files[0], cfg)
    assert len(sweeps) >= 1
    sw = sweeps[0]
    assert sw.V.shape == sw.I.shape and sw.V.size > 10
