"""
Tests for add_data.py — validate_staging(), sha256_file(),
current_version_number(), next_dataset_dir().

No YOLO / GPU required. Uses tmp_path for isolated filesystem state.
"""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image as PilImage

# Make repo root importable
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from add_data import (
    ValidationError,
    current_version_number,
    next_dataset_dir,
    sha256_file,
    validate_staging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_valid_staging(tmp_path: Path, n_images: int = 2) -> Path:
    """Create a minimal valid staging directory."""
    staging = tmp_path / "staging"
    (staging / "images").mkdir(parents=True)
    (staging / "labels").mkdir()
    (staging / "meta").mkdir()

    for i in range(n_images):
        stem = f"frame{i:04d}"
        # Create a tiny real JPEG so PIL can open it
        img = PilImage.new("RGB", (32, 32), color=(i * 40, 100, 200))
        img.save(staging / "images" / f"{stem}.jpg", "JPEG")

        (staging / "labels" / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n"
        )
        (staging / "meta" / f"{stem}.json").write_text(
            json.dumps({"auto_generated": True, "human_reviewed": False,
                        "human_corrected": False})
        )

    (staging / "batch_info.json").write_text(
        json.dumps({"date": "2026-04-15", "source": "test", "labeled_by": "pytest"})
    )
    return staging


# ---------------------------------------------------------------------------
# validate_staging — happy path
# ---------------------------------------------------------------------------

def test_validate_staging_passes(tmp_path):
    staging = make_valid_staging(tmp_path, n_images=3)
    result = validate_staging(staging)
    assert len(result["img_files"]) == 3
    assert len(result["new_hashes"]) == 3
    assert result["batch_info"]["source"] == "test"


# ---------------------------------------------------------------------------
# validate_staging — missing batch_info.json
# ---------------------------------------------------------------------------

def test_missing_batch_info(tmp_path):
    staging = make_valid_staging(tmp_path)
    (staging / "batch_info.json").unlink()
    with pytest.raises(ValidationError, match="batch_info.json"):
        validate_staging(staging)


def test_invalid_batch_info_json(tmp_path):
    staging = make_valid_staging(tmp_path)
    (staging / "batch_info.json").write_text("not json {{{")
    with pytest.raises(ValidationError, match="not valid JSON"):
        validate_staging(staging)


# ---------------------------------------------------------------------------
# validate_staging — missing directories
# ---------------------------------------------------------------------------

def test_missing_images_dir(tmp_path):
    staging = make_valid_staging(tmp_path)
    import shutil
    shutil.rmtree(staging / "images")
    with pytest.raises(ValidationError, match="images"):
        validate_staging(staging)


def test_missing_labels_dir(tmp_path):
    staging = make_valid_staging(tmp_path)
    import shutil
    shutil.rmtree(staging / "labels")
    with pytest.raises(ValidationError, match="labels"):
        validate_staging(staging)


def test_no_images_in_dir(tmp_path):
    staging = make_valid_staging(tmp_path)
    for f in (staging / "images").iterdir():
        f.unlink()
    with pytest.raises(ValidationError, match="No images"):
        validate_staging(staging)


# ---------------------------------------------------------------------------
# validate_staging — file pairing failures
# ---------------------------------------------------------------------------

def test_missing_label_for_image(tmp_path):
    staging = make_valid_staging(tmp_path, n_images=2)
    # Remove label for first image
    list((staging / "labels").iterdir())[0].unlink()
    with pytest.raises(ValidationError, match="Missing label"):
        validate_staging(staging)


def test_missing_meta_for_image(tmp_path):
    staging = make_valid_staging(tmp_path, n_images=2)
    list((staging / "meta").iterdir())[0].unlink()
    with pytest.raises(ValidationError, match="Missing meta"):
        validate_staging(staging)


def test_orphan_label(tmp_path):
    staging = make_valid_staging(tmp_path, n_images=1)
    (staging / "labels" / "ghost.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValidationError, match="Orphan label"):
        validate_staging(staging)


def test_orphan_meta(tmp_path):
    staging = make_valid_staging(tmp_path, n_images=1)
    (staging / "meta" / "ghost.json").write_text("{}")
    with pytest.raises(ValidationError, match="Orphan meta"):
        validate_staging(staging)


# ---------------------------------------------------------------------------
# validate_staging — label format errors
# ---------------------------------------------------------------------------

def test_label_wrong_field_count(tmp_path):
    staging = make_valid_staging(tmp_path, n_images=1)
    lbl = list((staging / "labels").iterdir())[0]
    lbl.write_text("0 0.5 0.5\n")  # only 3 values
    with pytest.raises(ValidationError, match="expected 5 values"):
        validate_staging(staging)


def test_label_wrong_class(tmp_path):
    staging = make_valid_staging(tmp_path, n_images=1)
    lbl = list((staging / "labels").iterdir())[0]
    lbl.write_text("1 0.5 0.5 0.1 0.1\n")  # class 1, not 0
    with pytest.raises(ValidationError, match="class id 1"):
        validate_staging(staging)


def test_label_cx_out_of_range(tmp_path):
    staging = make_valid_staging(tmp_path, n_images=1)
    lbl = list((staging / "labels").iterdir())[0]
    lbl.write_text("0 1.5 0.5 0.1 0.1\n")  # cx=1.5 > 1.0
    with pytest.raises(ValidationError, match="cx=1.5 out of"):
        validate_staging(staging)


def test_label_non_numeric(tmp_path):
    staging = make_valid_staging(tmp_path, n_images=1)
    lbl = list((staging / "labels").iterdir())[0]
    lbl.write_text("0 abc 0.5 0.1 0.1\n")
    with pytest.raises(ValidationError, match="non-numeric"):
        validate_staging(staging)


def test_empty_label_file_is_ok(tmp_path):
    """Empty label = no detections, perfectly valid."""
    staging = make_valid_staging(tmp_path, n_images=1)
    lbl = list((staging / "labels").iterdir())[0]
    lbl.write_text("")
    result = validate_staging(staging)
    assert len(result["img_files"]) == 1


# ---------------------------------------------------------------------------
# validate_staging — duplicate detection
# ---------------------------------------------------------------------------

def test_duplicate_within_batch(tmp_path):
    staging = make_valid_staging(tmp_path, n_images=1)
    # Copy the same image under a second stem
    src = list((staging / "images").iterdir())[0]
    import shutil
    dup = staging / "images" / "dup_frame.jpg"
    shutil.copy2(src, dup)
    (staging / "labels" / "dup_frame.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (staging / "meta" / "dup_frame.json").write_text("{}")
    with pytest.raises(ValidationError, match="duplicate within staging"):
        validate_staging(staging)


def test_duplicate_vs_existing_dataset(tmp_path):
    staging = make_valid_staging(tmp_path, n_images=1)
    img_path = list((staging / "images").iterdir())[0]
    h = sha256_file(img_path)

    # Plant the hash in a fake dataset
    ds_dir = tmp_path / "datasets" / "v1"
    ds_dir.mkdir(parents=True)
    (ds_dir / "hashes.txt").write_text(h + "\n")

    # Patch DATASETS path in add_data module
    import add_data
    original = add_data.DATASETS
    add_data.DATASETS = tmp_path / "datasets"
    try:
        with pytest.raises(ValidationError, match="duplicate"):
            validate_staging(staging)
    finally:
        add_data.DATASETS = original


# ---------------------------------------------------------------------------
# version numbering
# ---------------------------------------------------------------------------

def test_current_version_number_empty(tmp_path, monkeypatch):
    import add_data
    monkeypatch.setattr(add_data, "DATASETS", tmp_path / "datasets")
    assert add_data.current_version_number() == 0


def test_current_version_number(tmp_path, monkeypatch):
    import add_data
    ds = tmp_path / "datasets"
    (ds / "v1").mkdir(parents=True)
    (ds / "v3").mkdir()
    (ds / "v2").mkdir()
    monkeypatch.setattr(add_data, "DATASETS", ds)
    assert add_data.current_version_number() == 3


def test_next_dataset_dir(tmp_path, monkeypatch):
    import add_data
    ds = tmp_path / "datasets"
    (ds / "v2").mkdir(parents=True)
    monkeypatch.setattr(add_data, "DATASETS", ds)
    path, version = add_data.next_dataset_dir()
    assert version == "v3"
    assert path == ds / "v3"


# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------

def test_sha256_file_deterministic(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello world")
    h1 = sha256_file(f)
    h2 = sha256_file(f)
    assert h1 == h2
    assert len(h1) == 64


def test_sha256_file_differs(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbb")
    assert sha256_file(a) != sha256_file(b)
