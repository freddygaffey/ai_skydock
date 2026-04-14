"""
Validate staging/ and ingest into datasets/vN+1/.

Flow:
  rsync labels + images → staging/
  python add_data.py          → validate → datasets/vN+1/ → log to registry.db
  python add_data.py --dry-run → validate only, nothing moves

Validation checks (all must pass — staging untouched on any failure):
  Format:
    - Every .jpg has matching .txt and .json in meta/
    - No orphan labels or meta files
    - Each label line: exactly 5 values, all floats
    - cx, cy, w, h all in [0.0, 1.0]
    - Class id is 0 (Ball only)
    - batch_info.json present and parseable
  Content:
    - No duplicate images vs all existing dataset versions (SHA-256 hash)
    - Images not corrupt (readable by PIL)
"""

import argparse
import hashlib
import json
import shutil
import sqlite3
from datetime import date
from pathlib import Path

STAGING    = Path("staging")
DATASETS   = Path("datasets")
DB_PATH    = Path("registry.db")
HASH_FILE  = "hashes.txt"   # one SHA-256 per line per dataset version dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_all_hashes() -> set[str]:
    """Load hashes from all existing dataset versions."""
    hashes: set[str] = set()
    if not DATASETS.exists():
        return hashes
    for version_dir in DATASETS.iterdir():
        if not version_dir.is_dir():
            continue
        hf = version_dir / HASH_FILE
        if hf.exists():
            for line in hf.read_text().splitlines():
                h = line.strip()
                if h:
                    hashes.add(h)
    return hashes


def current_version_number() -> int:
    """Return highest existing vN number, or 0 if none."""
    if not DATASETS.exists():
        return 0
    nums = []
    for d in DATASETS.iterdir():
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit():
            nums.append(int(d.name[1:]))
    return max(nums) if nums else 0


def next_dataset_dir() -> tuple[Path, str]:
    n = current_version_number() + 1
    version = f"v{n}"
    return DATASETS / version, version


def count_images_in_dataset(version_dir: Path) -> int:
    total = 0
    for split in ("train", "valid", "test"):
        img_dir = version_dir / split / "images"
        if img_dir.is_dir():
            total += sum(1 for p in img_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return total


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    pass


def validate_staging(staging: Path) -> dict:
    """
    Full validation of staging/. Returns info dict on success.
    Raises ValidationError with descriptive message on any failure.
    Staging directory is never modified.
    """
    errors = []

    # 1. batch_info.json
    batch_info_path = staging / "batch_info.json"
    if not batch_info_path.exists():
        raise ValidationError("Missing staging/batch_info.json")
    try:
        batch_info = json.loads(batch_info_path.read_text())
    except Exception as e:
        raise ValidationError(f"batch_info.json not valid JSON: {e}")

    img_dir  = staging / "images"
    lbl_dir  = staging / "labels"
    meta_dir = staging / "meta"

    if not img_dir.is_dir():
        raise ValidationError("staging/images/ directory missing")
    if not lbl_dir.is_dir():
        raise ValidationError("staging/labels/ directory missing")
    if not meta_dir.is_dir():
        raise ValidationError("staging/meta/ directory missing")

    img_files = sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    if not img_files:
        raise ValidationError("No images found in staging/images/")

    # 2. Every image has matching label and meta
    for img in img_files:
        lbl = lbl_dir / (img.stem + ".txt")
        meta = meta_dir / (img.stem + ".json")
        if not lbl.exists():
            errors.append(f"Missing label: {lbl.name}")
        if not meta.exists():
            errors.append(f"Missing meta:  {meta.name}")

    # 3. No orphan labels
    for lbl in lbl_dir.iterdir():
        if lbl.suffix != ".txt":
            continue
        img_candidates = [
            img_dir / (lbl.stem + ext)
            for ext in (".jpg", ".jpeg", ".png")
        ]
        if not any(p.exists() for p in img_candidates):
            errors.append(f"Orphan label (no image): {lbl.name}")

    # 4. No orphan meta
    for meta in meta_dir.iterdir():
        if meta.suffix != ".json":
            continue
        img_candidates = [
            img_dir / (meta.stem + ext)
            for ext in (".jpg", ".jpeg", ".png")
        ]
        if not any(p.exists() for p in img_candidates):
            errors.append(f"Orphan meta (no image): {meta.name}")

    if errors:
        raise ValidationError(
            "File pairing failures:\n" + "\n".join(f"  {e}" for e in errors)
        )

    # 5. Label format
    label_errors = []
    for img in img_files:
        lbl_path = lbl_dir / (img.stem + ".txt")
        if not lbl_path.exists():
            continue
        for lineno, line in enumerate(lbl_path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                label_errors.append(
                    f"{lbl_path.name}:{lineno}: expected 5 values, got {len(parts)}"
                )
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                label_errors.append(f"{lbl_path.name}:{lineno}: non-numeric values")
                continue
            if cls_id != 0:
                label_errors.append(
                    f"{lbl_path.name}:{lineno}: class id {cls_id} (expected 0 = Ball)"
                )
            for val, name in [(cx, "cx"), (cy, "cy"), (w, "w"), (h, "h")]:
                if not (0.0 <= val <= 1.0):
                    label_errors.append(
                        f"{lbl_path.name}:{lineno}: {name}={val} out of [0,1]"
                    )

    if label_errors:
        raise ValidationError(
            "Label format failures:\n" + "\n".join(f"  {e}" for e in label_errors[:50])
        )

    # 6. Image integrity + hash dedup
    existing_hashes = load_all_hashes()
    new_hashes: dict[Path, str] = {}
    hash_errors = []
    corrupt_errors = []

    try:
        from PIL import Image as PilImage
    except ImportError:
        raise ValidationError("Pillow not installed — run: pip install pillow")

    for img in img_files:
        # Corruption check
        try:
            pil_img = PilImage.open(img)
            pil_img.verify()
        except Exception as e:
            corrupt_errors.append(f"{img.name}: {e}")
            continue

        # Hash dedup
        h = sha256_file(img)
        if h in existing_hashes:
            hash_errors.append(f"{img.name}: duplicate (hash already in dataset)")
        elif h in new_hashes.values():
            hash_errors.append(f"{img.name}: duplicate within staging batch")
        else:
            new_hashes[img] = h

    all_errors = corrupt_errors + hash_errors
    if all_errors:
        raise ValidationError(
            "Content failures:\n" + "\n".join(f"  {e}" for e in all_errors)
        )

    return {
        "img_files":   img_files,
        "new_hashes":  new_hashes,
        "batch_info":  batch_info,
    }


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest(staging: Path, validation: dict, dry_run: bool) -> tuple[Path, str]:
    """Move validated staging data into datasets/vN+1/. Returns (dest_dir, version)."""
    img_files  = validation["img_files"]
    new_hashes = validation["new_hashes"]
    batch_info = validation["batch_info"]

    dest_dir, version = next_dataset_dir()

    if dry_run:
        print(f"\n[DRY RUN] Would create {dest_dir}/ with {len(img_files)} images.")
        return dest_dir, version

    # Determine split from batch_info, default train
    split = batch_info.get("split", "train")
    if split not in ("train", "valid", "test"):
        split = "train"

    # Carry forward previous dataset content (copy, not move)
    prev_n = current_version_number()
    if prev_n > 0:
        prev_dir = DATASETS / f"v{prev_n}"
        if prev_dir.is_dir():
            print(f"  Copying {prev_dir.name}/ → {dest_dir.name}/ ...")
            shutil.copytree(str(prev_dir), str(dest_dir))

    # Create split dirs
    (dest_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (dest_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    (dest_dir / "meta").mkdir(exist_ok=True)

    lbl_dir  = staging / "labels"
    meta_dir = staging / "meta"

    print(f"  Adding {len(img_files)} images to {dest_dir}/{split}/")
    for img in img_files:
        dest_img = dest_dir / split / "images" / img.name
        dest_lbl = dest_dir / split / "labels" / (img.stem + ".txt")
        dest_meta = dest_dir / "meta" / (img.stem + ".json")

        shutil.copy2(img, dest_img)
        src_lbl = lbl_dir / (img.stem + ".txt")
        if src_lbl.exists():
            shutil.copy2(src_lbl, dest_lbl)
        meta_src = meta_dir / (img.stem + ".json")
        if meta_src.exists():
            shutil.copy2(meta_src, dest_meta)

    # Update hash file
    all_hashes = load_all_hashes() | set(new_hashes.values())
    (dest_dir / HASH_FILE).write_text("\n".join(sorted(all_hashes)) + "\n")

    # Copy batch_info
    shutil.copy2(staging / "batch_info.json", dest_dir / "batch_info.json")

    # Write data.yaml — point at the new dataset
    # Build paths from existing yaml if available, else from scratch
    train_path = str((dest_dir / "train" / "images").resolve())
    val_path   = str((dest_dir / "valid" / "images").resolve())
    test_path  = str((dest_dir / "test"  / "images").resolve())
    yaml_content = (
        f"names:\n- Ball\nnc: 1\n"
        f"train: {train_path}\n"
        f"val:   {val_path}\n"
        f"test:  {test_path}\n"
    )
    (dest_dir / "data.yaml").write_text(yaml_content)

    # Clear staging (but keep directory structure)
    for f in (staging / "images").iterdir():
        if f.is_file():
            f.unlink()
    for f in (staging / "labels").iterdir():
        if f.is_file():
            f.unlink()
    for f in (staging / "meta").iterdir():
        if f.is_file():
            f.unlink()
    (staging / "batch_info.json").unlink(missing_ok=True)

    return dest_dir, version


def register_dataset(version: str, dest_dir: Path, batch_info: dict,
                     images_added: int):
    if not DB_PATH.exists():
        print("  WARNING: registry.db not found — dataset not registered in DB.")
        return
    total = count_images_in_dataset(dest_dir)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        INSERT OR IGNORE INTO datasets
            (version, date, total_images, images_added, source, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        version,
        batch_info.get("date", date.today().isoformat()),
        total,
        images_added,
        batch_info.get("source", "manual"),
        batch_info.get("notes", ""),
    ))
    conn.commit()
    conn.close()
    print(f"  registry.db: {version} registered ({total} total images, {images_added} added)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate only, move nothing")
    parser.add_argument("--staging", default=str(STAGING),
                        help=f"Staging directory (default: {STAGING})")
    args = parser.parse_args()

    staging = Path(args.staging)
    print(f"Staging : {staging.resolve()}")

    print("\n[1/2] Validating ...")
    try:
        info = validate_staging(staging)
    except ValidationError as e:
        print(f"\nVALIDATION FAILED — nothing moved:\n{e}")
        import sys
        sys.exit(1)

    n_images = len(info["img_files"])
    print(f"  OK — {n_images} images, {len(info['new_hashes'])} unique.")

    if args.dry_run:
        dest_dir, version = ingest(staging, info, dry_run=True)
        print(f"\nDry run passed. Would create {version}.")
        return

    print("\n[2/2] Ingesting ...")
    dest_dir, version = ingest(staging, info, dry_run=False)
    register_dataset(version, dest_dir, info["batch_info"], n_images)

    print(f"\nDone. Created datasets/{version}/  ({n_images} new images)")
    print("Staging cleared. Run 2_train.py --data datasets/{version}/data.yaml next.")


if __name__ == "__main__":
    main()
