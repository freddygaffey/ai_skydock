"""
Train YOLO ball detection models and register results in registry.db.

Edge model (YOLOv8n/s) → compiled to HEF, deployed to RPi Hailo-8.
Ground model (YOLOv8x) → stays on laptop / deb.local, used for auto-labeling.

Usage:
    source venv/bin/activate
    python 2_train.py                              # edge model, 1280px, auto batch
    python 2_train.py --model yolov8n --imgsz 1280
    python 2_train.py --model yolov8x              # ground model
    python 2_train.py --finetune-from v003         # finetune from registry version
    python 2_train.py --batch 8                    # override batch size
    python 2_train.py --resume                     # resume interrupted training
"""

import argparse
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path


DB_PATH = Path("registry.db")


def get_device() -> tuple[float, str]:
    """Returns (available_memory_gb, device_string)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9, "cuda"
        if torch.backends.mps.is_available():
            import psutil
            total_ram = psutil.virtual_memory().total / 1e9
            return min(total_ram * 0.75, 24.0), "mps"
    except Exception:
        pass
    return 0.0, "cpu"


def next_version() -> str:
    """Return next vNNN version string, scanning model_registry/ dir."""
    registry = Path("model_registry")
    registry.mkdir(exist_ok=True)
    existing = sorted(
        int(p.name[1:]) for p in registry.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    )
    n = (existing[-1] + 1) if existing else 1
    return f"v{n:03d}"


def get_latest_dataset_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT id FROM datasets ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["id"] if row else None


def get_model_pt_path(version: str) -> Path:
    """Resolve --finetune-from version to a .pt path."""
    pt = Path("model_registry") / version / "best.pt"
    if not pt.exists():
        raise FileNotFoundError(f"No weights at {pt} for version {version!r}")
    return pt


def register_run(
    conn: sqlite3.Connection,
    version: str,
    model_arch: str,
    imgsz: int,
    epochs: int,
    duration_mins: float,
    results,
    pt_path: Path,
    parent_version: str | None,
    dataset_id: int | None,
):
    """Write training_run + model rows to registry.db."""
    parent_id = None
    if parent_version:
        row = conn.execute(
            "SELECT id FROM models WHERE version=?", (parent_version,)
        ).fetchone()
        if row:
            parent_id = row["id"]

    run_id = conn.execute("""
        INSERT INTO training_runs (date, model_arch, imgsz, epochs, duration_mins,
                                   dataset_id, parent_model_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        model_arch,
        imgsz,
        epochs,
        round(duration_mins, 2),
        dataset_id,
        parent_id,
    )).lastrowid

    # Extract metrics from Ultralytics results object
    mAP50 = mAP50_95 = prec = rec = None
    try:
        d = results.results_dict
        mAP50    = d.get("metrics/mAP50(B)")
        mAP50_95 = d.get("metrics/mAP50-95(B)")
        prec     = d.get("metrics/precision(B)")
        rec      = d.get("metrics/recall(B)")
    except Exception:
        pass

    conn.execute("""
        INSERT OR REPLACE INTO models
            (version, training_run_id, mAP50, mAP50_95, precision_val, recall_val,
             deployed, pt_path)
        VALUES (?, ?, ?, ?, ?, ?, 0, ?)
    """, (version, run_id, mAP50, mAP50_95, prec, rec, str(pt_path)))
    conn.commit()

    print(f"\n[Registry] Saved: {version}  mAP50={mAP50}  run_id={run_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",           default="merged_dataset/data.yaml")
    parser.add_argument("--model",          default="yolov8n.pt",
                        help="Model weights or hub name (yolov8n.pt, yolov8x.pt, ...)")
    parser.add_argument("--epochs",         type=int, default=100)
    parser.add_argument("--batch",          type=int, default=None)
    parser.add_argument("--imgsz",          type=int, default=1280,
                        help="Image size for training and ONNX export (default: 1280)")
    parser.add_argument("--name",           default=None,
                        help="Run name under runs/detect/. Defaults to model arch.")
    parser.add_argument("--resume",         action="store_true")
    parser.add_argument("--device",         default=None)
    parser.add_argument("--finetune-from",  dest="finetune_from", default=None,
                        metavar="VERSION",
                        help="Finetune from a registered version, e.g. --finetune-from v003")
    parser.add_argument("--version",        default=None,
                        help="Force output version string (default: auto-increment)")
    parser.add_argument("--no-register",    dest="no_register", action="store_true",
                        help="Skip writing to registry.db")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Resolve weights
    if args.finetune_from:
        weights = str(get_model_pt_path(args.finetune_from))
        print(f"Finetune from : {args.finetune_from} → {weights}")
    else:
        weights = args.model

    # Detect architecture name from weights filename for registry
    arch_name = Path(weights).stem.lower()  # e.g. "yolov8n", "best"
    if args.finetune_from:
        # Look up original arch in registry
        if DB_PATH.exists():
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT tr.model_arch FROM models m
                LEFT JOIN training_runs tr ON m.training_run_id = tr.id
                WHERE m.version=?
            """, (args.finetune_from,)).fetchone()
            conn.close()
            if row and row["model_arch"]:
                arch_name = row["model_arch"]

    run_name = args.name or f"ball_{arch_name}"

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        return

    vram, auto_device = get_device()
    if args.device is None:
        args.device = auto_device

    workers = 2 if auto_device == "mps" else 4

    print(f"Device   : {args.device}  Memory: {vram:.1f} GB")
    print(f"Image sz : {args.imgsz}px")

    from ultralytics import YOLO

    batch = args.batch if args.batch is not None else -1
    if args.batch is not None:
        print(f"Batch    : {batch} (manual override)")
    else:
        print(f"Batch    : auto (Ultralytics will find best fit for {vram:.1f} GB VRAM)")

    if args.resume:
        last = Path(f"runs/detect/{run_name}/weights/last.pt")
        model = YOLO(str(last) if last.exists() else weights)
        if last.exists():
            print(f"Resuming from {last}")
    else:
        model = YOLO(weights)

    print(f"\nTraining on : {data_path.resolve()}")
    print(f"Weights     : {weights}  Epochs: {args.epochs}")

    t0 = time.time()
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        name=run_name,
        device=args.device,
        workers=workers,
        exist_ok=args.resume,
        resume=args.resume,
        flipud=0.3,
        fliplr=0.5,
        degrees=15.0,
        scale=0.4,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        close_mosaic=10,
        patience=30,
    )
    duration_mins = (time.time() - t0) / 60.0

    best_pt_run = Path(f"runs/detect/{run_name}/weights/best.pt")
    print(f"\nTraining complete. Best weights: {best_pt_run}")
    try:
        print(f"mAP50    : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"mAP50-95 : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    except Exception:
        pass

    # Archive to model_registry/
    version = args.version or next_version()
    version_dir = Path("model_registry") / version
    version_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    dest_pt = version_dir / "best.pt"
    shutil.copy2(best_pt_run, dest_pt)
    print(f"\nArchived: {dest_pt}")

    # Copy training log
    log_src = Path(f"runs/detect/{run_name}/results.csv")
    if log_src.exists():
        shutil.copy2(log_src, version_dir / "training_log.csv")

    # ONNX export
    print("\nExporting to ONNX...")
    model = YOLO(str(dest_pt))
    model.export(format="onnx", imgsz=args.imgsz, opset=11, simplify=True, dynamic=False)
    onnx_src = dest_pt.with_suffix(".onnx")
    if onnx_src.exists():
        print(f"ONNX: {onnx_src}")
    else:
        # Ultralytics may export next to the original
        onnx_src = best_pt_run.with_suffix(".onnx")
        if onnx_src.exists():
            shutil.copy2(onnx_src, version_dir / "best.onnx")
            print(f"ONNX: {version_dir / 'best.onnx'}")

    # Register in DB
    if not args.no_register and DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        dataset_id = get_latest_dataset_id(conn)
        register_run(
            conn=conn,
            version=version,
            model_arch=arch_name,
            imgsz=args.imgsz,
            epochs=args.epochs,
            duration_mins=duration_mins,
            results=results,
            pt_path=dest_pt,
            parent_version=args.finetune_from,
            dataset_id=dataset_id,
        )
        conn.close()
    elif not DB_PATH.exists():
        print("WARNING: registry.db not found — run init_registry.py first.")

    print(f"\nVersion {version} archived. Next: ./3_compile_hailo8.sh {version_dir}/best.onnx")


if __name__ == "__main__":
    main()
