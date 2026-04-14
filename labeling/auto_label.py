"""
Auto-label a flight's raw_frames/ using the ground model.
Writes YOLO .txt labels and per-image meta .json to the flight directory,
then optionally stages them for add_data.py.

Usage:
    # First set model path (or use SKYDOCK_YOLO_MODEL env var):
    export SKYDOCK_YOLO_MODEL=~/ai_skydock/model_registry/ground_latest/best.pt

    source ~/ai_skydock/venv/bin/activate
    python labeling/auto_label.py --flight 2026-04-20_flight01
    python labeling/auto_label.py --flight 2026-04-20_flight01 --stage
    python labeling/auto_label.py --frames /path/to/frames/ --stage

The ground model is the YOLOv8x variant. Set via:
  SKYDOCK_YOLO_MODEL=~/ai_skydock/model_registry/ground_latest/best.pt
  or --model flag.

Output in flights/FLIGHT_ID/:
  labels_auto_vN/   ← YOLO .txt labels (class cx cy w h)
  meta/             ← per-image .json (auto_generated, human_reviewed, etc.)
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH   = REPO_ROOT / "registry.db"


def get_model_path(override: str | None) -> str:
    """Resolve ground model path. Priority: --model flag, SKYDOCK_YOLO_MODEL, ground_latest."""
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return str(p.resolve())
        raise FileNotFoundError(f"Model not found: {override}")

    env = os.environ.get("SKYDOCK_YOLO_MODEL", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return str(p.resolve())

    ground_latest = REPO_ROOT / "model_registry" / "ground_latest" / "best.pt"
    if ground_latest.exists():
        return str(ground_latest)

    raise FileNotFoundError(
        "No ground model found. Set SKYDOCK_YOLO_MODEL env var or pass --model."
    )


def get_model_version(model_path: str) -> str | None:
    """Try to find registered version for this .pt path."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT version FROM models WHERE pt_path=?", (model_path,)
    ).fetchone()
    conn.close()
    return row["version"] if row else None


def label_dir_name(model_version: str | None) -> str:
    """e.g. 'labels_auto_v003' or 'labels_auto_unknown'."""
    v = model_version or "unknown"
    return f"labels_auto_{v}"


def run_inference(model_path: str, frames_dir: Path, imgsz: int,
                  conf: float, device: str) -> dict[str, list[list[float]]]:
    """
    Run YOLO on all JPGs in frames_dir.
    Returns {stem: [[cx, cy, w, h], ...]} in normalised coords.
    """
    from ultralytics import YOLO
    model = YOLO(model_path)

    img_paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not img_paths:
        return {}

    results: dict[str, list[list[float]]] = {}
    for res in model.predict(
        source=[str(p) for p in img_paths],
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
        stream=True,
    ):
        stem = Path(res.path).stem
        boxes = []
        if res.boxes is not None:
            for box in res.boxes.xywhn.cpu().numpy():
                boxes.append([round(float(v), 6) for v in box[:4]])
        results[stem] = boxes
    return results


def write_yolo_label(label_path: Path, boxes: list[list[float]]):
    """Write YOLO format label file (class 0 = Ball)."""
    lines = [f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_frame_meta(meta_path: Path, model_version: str | None, n_dets: int):
    """Write per-image meta JSON for staging."""
    meta = {
        "label_type": "auto",
        "auto_generated": True,
        "human_reviewed": False,
        "human_corrected": False,
        "original_model": model_version or "unknown",
        "correction_by": None,
        "correction_date": None,
        "n_detections": n_dets,
        "notes": "",
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def stage_flight(flight_dir: Path, label_dir: Path, model_version: str | None):
    """Copy images + labels + meta to staging/ for add_data.py."""
    staging = REPO_ROOT / "staging"

    frames_dir = flight_dir / "raw_frames"
    meta_dir   = flight_dir / "meta"

    n_copied = 0
    for lbl_file in label_dir.iterdir():
        if lbl_file.suffix != ".txt":
            continue
        stem = lbl_file.stem
        # Find the image
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = frames_dir / (stem + ext)
            if candidate.exists():
                img = candidate
                break
        if img is None:
            continue

        import shutil
        shutil.copy2(img, staging / "images" / img.name)
        shutil.copy2(lbl_file, staging / "labels" / lbl_file.name)

        meta_src = meta_dir / (stem + ".json")
        if meta_src.exists():
            shutil.copy2(meta_src, staging / "meta" / (stem + ".json"))
        n_copied += 1

    # Write batch_info.json
    batch_info = {
        "date": date.today().isoformat(),
        "source": "auto_label",
        "labeled_by": model_version or "ground_model",
        "notes": f"Auto-labeled from {flight_dir.name}",
        "flight_id": flight_dir.name,
        "split": "train",
    }
    (staging / "batch_info.json").write_text(json.dumps(batch_info, indent=2))

    print(f"\nStaged {n_copied} frames to {staging}/")
    print("Run: python add_data.py")


def main():
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--flight",  metavar="FLIGHT_ID",
                     help="Flight ID under flights/ (e.g. 2026-04-20_flight01)")
    grp.add_argument("--frames",  metavar="FRAMES_DIR",
                     help="Direct path to frames directory")
    parser.add_argument("--model",   default=None,
                        help="Path to ground model .pt (default: SKYDOCK_YOLO_MODEL env)")
    parser.add_argument("--imgsz",   type=int, default=1280)
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--device",  default=None)
    parser.add_argument("--stage",   action="store_true",
                        help="Copy to staging/ after labeling (ready for add_data.py)")
    parser.add_argument("--stride",  type=int, default=1,
                        help="Label every Nth frame (1 = all)")
    args = parser.parse_args()

    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = get_model_path(args.model)
    model_version = get_model_version(model_path)

    print(f"Ground model : {model_path}")
    print(f"Version      : {model_version or 'not in registry'}")
    print(f"Device       : {args.device}  imgsz={args.imgsz}  conf={args.conf}")

    # Resolve flight dir
    if args.flight:
        flight_dir = REPO_ROOT / "flights" / args.flight
        if not flight_dir.is_dir():
            print(f"ERROR: {flight_dir} not found")
            return
        frames_dir = flight_dir / "raw_frames"
    else:
        flight_dir = None
        frames_dir = Path(args.frames)

    if not frames_dir.is_dir():
        print(f"ERROR: frames dir not found: {frames_dir}")
        return

    # Determine output label dir
    lbl_dir_name = label_dir_name(model_version)
    if flight_dir:
        out_label_dir = flight_dir / lbl_dir_name
        out_meta_dir  = flight_dir / "meta"
    else:
        out_label_dir = frames_dir.parent / lbl_dir_name
        out_meta_dir  = frames_dir.parent / "meta"

    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    # Collect frames (apply stride)
    img_paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.stride > 1:
        img_paths = img_paths[::args.stride]

    if not img_paths:
        print(f"No images found in {frames_dir}")
        return

    print(f"\nFrames dir   : {frames_dir}")
    print(f"Frame count  : {len(img_paths)}")
    print(f"Output labels: {out_label_dir}")

    print("\nRunning inference ...")
    preds = run_inference(model_path, frames_dir, args.imgsz, args.conf, args.device)

    # Write labels + meta
    n_with_det = 0
    for stem, boxes in preds.items():
        write_yolo_label(out_label_dir / (stem + ".txt"), boxes)
        write_frame_meta(out_meta_dir / (stem + ".json"), model_version, len(boxes))
        if boxes:
            n_with_det += 1

    total = len(preds)
    print(f"Wrote {total} label files  ({n_with_det} with detections, {total - n_with_det} empty)")

    if args.stage and flight_dir:
        stage_flight(flight_dir, out_label_dir, model_version)
    elif args.stage and not flight_dir:
        print("WARNING: --stage requires --flight (not --frames). Stage manually.")


if __name__ == "__main__":
    main()
