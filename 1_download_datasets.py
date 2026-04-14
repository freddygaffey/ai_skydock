"""
Download and merge multiple ball detection datasets from Roboflow.

Datasets sourced (all single-class, remapped to class 0 = "Ball"):
  - istt/balls-9q0lp v4      (BALLS-4, general balls)
  - roboflow-100/ball-detection-nnh2m v1  (sports balls)
  - davidproject24/soccer-ball-detection v2 (soccer/football)
  - basketball-detection datasets

Usage:
    source venv/bin/activate
    python 1_download_datasets.py --api-key YOUR_KEY
    # or set env var: export ROBOFLOW_API_KEY=YOUR_KEY
    python 1_download_datasets.py
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Datasets to download  (workspace, project, version)
# Add or remove rows to change what gets merged.
# ---------------------------------------------------------------------------
DATASETS = [
    # From https://universe.roboflow.com/search?q=balls (first 10 results)
    ("istt",                  "balls-9q0lp",           4),  # 7,902 images — confirmed
    ("leslie-nguyen",         "ball-dataset-k7f64",    1),  # 13,578 images
    ("new-workspace-va9vn",   "balls-detection",       1),  # 815 images
    ("viren-dhanwani",        "tennis-ball-detection", 1),  # 578 images
    ("kandidat-pgnqx",        "ball-detection-hwpfg",  1),  # 162 images
    ("8ballpool",             "ball-detection-qkvnj",  1),  # 95 images
    ("flynn-harrison",        "balls-detector",        5),  # 97 images
    ("jft",                   "balls-juxru",           1),  # misc balls
    ("draker-master-nybia",   "color-balls",           1),  # 1,800 images
    # User-specified datasets
    ("ayah-buisa",            "waterpolo-ball-detection-hkfpe", 1),  # water polo balls
    ("josias-werly-gmail-com","ball-0zqmb",            1),  # balls
]

MERGED_DIR = Path("datasets/v1")
SPLITS = ["train", "valid", "test"]


def download_dataset(rf, workspace, project, version, dest: Path):
    """Download a Roboflow dataset in YOLOv8 format to dest/. Returns None on failure."""
    if dest.exists():
        print(f"  Skipping (already downloaded): {dest}")
        return dest / "data.yaml"
    print(f"  Downloading {workspace}/{project} v{version} ...")
    try:
        rf.workspace(workspace).project(project).version(version).download(
            "yolov8", location=str(dest)
        )
        return dest / "data.yaml"
    except Exception as e:
        print(f"  FAILED ({workspace}/{project} v{version}): {e}")
        if dest.exists():
            shutil.rmtree(dest)
        return None


def remap_labels(label_file: Path, class_map: dict[int, int]):
    """Rewrite label file so old class IDs are replaced by mapped IDs."""
    lines = label_file.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        old_cls = int(parts[0])
        new_cls = class_map.get(old_cls)
        if new_cls is None:
            # Class not in our map — skip this detection
            continue
        new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
    label_file.write_text("\n".join(new_lines) + "\n" if new_lines else "")


def merge_split(split: str, src_dirs: list[tuple[Path, dict[int, int]]]):
    """Copy images + remapped labels from all source dirs into merged_dataset/split/."""
    img_out = MERGED_DIR / split / "images"
    lbl_out = MERGED_DIR / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total = 0
    for src_root, class_map in src_dirs:
        img_src = src_root / split / "images"
        lbl_src = src_root / split / "labels"
        if not img_src.exists():
            continue
        for img_path in img_src.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            lbl_path = lbl_src / (img_path.stem + ".txt")

            # Unique filename to avoid collisions between datasets
            prefix = src_root.name[:12].replace(" ", "_")
            new_stem = f"{prefix}_{img_path.name}"
            dest_img = img_out / new_stem
            dest_lbl = lbl_out / (new_stem.rsplit(".", 1)[0] + ".txt")

            shutil.copy2(img_path, dest_img)

            if lbl_path.exists():
                shutil.copy2(lbl_path, dest_lbl)
                remap_labels(dest_lbl, class_map)
            else:
                # Create empty label file (background image)
                dest_lbl.write_text("")
            total += 1
    print(f"  {split}: {total} images")
    return total


def build_data_yaml():
    yaml_content = f"""names:
- Ball
nc: 1
train: {(MERGED_DIR / 'train' / 'images').resolve()}
val:   {(MERGED_DIR / 'valid' / 'images').resolve()}
test:  {(MERGED_DIR / 'test'  / 'images').resolve()}
"""
    (MERGED_DIR / "data.yaml").write_text(yaml_content)
    print(f"Wrote {MERGED_DIR}/data.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Provide --api-key or set ROBOFLOW_API_KEY env var")
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=args.api_key)

    raw_dir = Path("raw_datasets")
    raw_dir.mkdir(exist_ok=True)

    src_dirs: list[tuple[Path, dict[int, int]]] = []

    for workspace, project, version in DATASETS:
        dest = raw_dir / f"{workspace}__{project}_v{version}"
        yaml_path = download_dataset(rf, workspace, project, version, dest)
        if yaml_path is None:
            continue  # skip this dataset

        # Parse the data.yaml to build a class_map: all classes → 0 ("Ball")
        import yaml
        try:
            with open(yaml_path) as f:
                meta = yaml.safe_load(f)
            names = meta.get("names", [])
            class_map = {i: 0 for i in range(len(names) if isinstance(names, list) else len(names))}
            print(f"  Classes in {project}: {names}  → all remapped to 'Ball'")
        except Exception as e:
            print(f"  WARNING: could not parse data.yaml for {project}: {e}")
            class_map = {0: 0}

        src_dirs.append((dest, class_map))

    print(f"\nMerging {len(src_dirs)} datasets into {MERGED_DIR}/")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        merge_split(split, src_dirs)

    build_data_yaml()

    # Count total images
    total = sum(
        1 for p in (MERGED_DIR / "train" / "images").iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ) if (MERGED_DIR / "train" / "images").exists() else 0

    # Register in registry.db if it exists
    db_path = Path("registry.db")
    if db_path.exists():
        import sqlite3
        from datetime import date
        conn = sqlite3.connect(str(db_path))
        existing = conn.execute(
            "SELECT id FROM datasets WHERE version='v1'"
        ).fetchone()
        if not existing:
            conn.execute("""
                INSERT INTO datasets (version, date, total_images, images_added, source, notes)
                VALUES ('v1', ?, ?, ?, 'roboflow', 'Initial Roboflow download — 11 datasets merged')
            """, (date.today().isoformat(), total, total))
            conn.commit()
            print(f"Registered datasets/v1 in registry.db  ({total} images)")
        conn.close()

    print(f"\nDone. Dataset written to {MERGED_DIR}/")
    print("Run 2_train.py next (use --data datasets/v1/data.yaml).")


if __name__ == "__main__":
    main()
