"""
Tests for eval_registry.py — regression_gate(), latest_val_data().

No YOLO / GPU required.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval_registry import regression_gate, latest_val_data, GATE_THRESHOLD


# ---------------------------------------------------------------------------
# regression_gate
# ---------------------------------------------------------------------------

def test_gate_passes_improvement():
    passes, delta = regression_gate(candidate_map50=0.80, deployed_map50=0.75)
    assert passes is True
    assert delta == pytest.approx(0.05)


def test_gate_passes_no_change():
    passes, delta = regression_gate(0.75, 0.75)
    assert passes is True
    assert delta == pytest.approx(0.0)


def test_gate_passes_small_drop():
    # Drop slightly less than GATE_THRESHOLD should pass.
    # Avoid exact threshold boundary — floating point subtraction is not exact.
    passes, delta = regression_gate(0.75 - GATE_THRESHOLD + 0.001, 0.75)
    assert passes is True


def test_gate_fails_drop_exceeds_threshold():
    passes, delta = regression_gate(0.75 - GATE_THRESHOLD - 0.001, 0.75)
    assert passes is False


def test_gate_fails_large_drop():
    passes, delta = regression_gate(0.50, 0.80)
    assert passes is False
    assert delta == pytest.approx(-0.30)


# ---------------------------------------------------------------------------
# latest_val_data
# ---------------------------------------------------------------------------

def test_latest_val_data_finds_highest(tmp_path, monkeypatch):
    import eval_registry
    ds = tmp_path / "datasets"
    (ds / "v1" / "train" / "images").mkdir(parents=True)
    (ds / "v1").joinpath("data.yaml").write_text("nc: 1\n")
    (ds / "v3" / "train" / "images").mkdir(parents=True)
    (ds / "v3").joinpath("data.yaml").write_text("nc: 1\n")
    (ds / "v2" / "train" / "images").mkdir(parents=True)
    (ds / "v2").joinpath("data.yaml").write_text("nc: 1\n")

    monkeypatch.chdir(tmp_path)
    result = latest_val_data()
    assert "v3" in result


def test_latest_val_data_skips_missing_yaml(tmp_path, monkeypatch):
    import eval_registry
    ds = tmp_path / "datasets"
    # v2 exists but has no data.yaml; v1 does
    (ds / "v2").mkdir(parents=True)
    (ds / "v1").mkdir()
    (ds / "v1" / "data.yaml").write_text("nc: 1\n")

    monkeypatch.chdir(tmp_path)
    result = latest_val_data()
    assert "v1" in result


def test_latest_val_data_falls_back_to_merged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    merged = tmp_path / "merged_dataset"
    merged.mkdir()
    (merged / "data.yaml").write_text("nc: 1\n")
    result = latest_val_data()
    assert "merged_dataset" in result


def test_latest_val_data_raises_when_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        latest_val_data()
