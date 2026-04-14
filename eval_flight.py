"""
Evaluate a registered model version over all raw frames in a flight directory,
comparing predictions against labels_truth/ (if present) or labels_auto_vN/.
Results logged to frame_results in registry.db.

Usage:
    source venv/bin/activate
    python eval_flight.py --model v005 --flight 2026-04-20_flight01
    python eval_flight.py --model v005 --flight 2026-04-20_flight01 --labels labels_truth
    python eval_flight.py --model v005 --flight 2026-04-20_flight01 --iou-thresh 0.5
"""

import argparse
import sqlite3
from pathlib import Path

DB_PATH  = Path("registry.db")
IOU_DEFAULT = 0.5


def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found — run init_registry.py first.")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def iou_box(b1: list[float], b2: list[float]) -> float:
    """Compute IoU between two YOLO-format boxes [cx, cy, w, h] (normalised)."""
    x1_min = b1[0] - b1[2] / 2
    x1_max = b1[0] + b1[2] / 2
    y1_min = b1[1] - b1[3] / 2
    y1_max = b1[1] + b1[3] / 2

    x2_min = b2[0] - b2[2] / 2
    x2_max = b2[0] + b2[2] / 2
    y2_min = b2[1] - b2[3] / 2
    y2_max = b2[1] + b2[3] / 2

    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter   = inter_w * inter_h
    union   = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def parse_yolo_label(label_path: Path) -> list[list[float]]:
    """Parse YOLO label file. Returns list of [cx, cy, w, h] for class-0 boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and int(parts[0]) == 0:
            boxes.append([float(p) for p in parts[1:5]])
    return boxes


def match_boxes(pred_boxes: list, gt_boxes: list, iou_thresh: float):
    """
    Greedy match predictions to ground truth at given IoU threshold.
    Returns (TP, FP, FN, avg_iou_of_matched).
    """
    if not gt_boxes:
        return 0, len(pred_boxes), 0, 0.0
    if not pred_boxes:
        return 0, 0, len(gt_boxes), 0.0

    used_gt = set()
    tp = fp = 0
    ious = []
    for pb in pred_boxes:
        best_iou = 0.0
        best_j   = -1
        for j, gb in enumerate(gt_boxes):
            if j in used_gt:
                continue
            v = iou_box(pb, gb)
            if v > best_iou:
                best_iou = v
                best_j   = j
        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            used_gt.add(best_j)
            ious.append(best_iou)
        else:
            fp += 1
    fn = len(gt_boxes) - len(used_gt)
    avg_iou = sum(ious) / len(ious) if ious else 0.0
    return tp, fp, fn, avg_iou


def resolve_label_dir(flight_dir: Path, label_hint: str | None) -> Path | None:
    """
    Priority:
      1. Explicit --labels argument
      2. labels_truth/ (if present)
      3. Highest-numbered labels_auto_vN/
    """
    if label_hint:
        p = flight_dir / label_hint
        return p if p.is_dir() else None

    truth = flight_dir / "labels_truth"
    if truth.is_dir():
        return truth

    candidates = sorted(
        [d for d in flight_dir.iterdir()
         if d.is_dir() and d.name.startswith("labels_auto_")],
        key=lambda d: d.name,
    )
    return candidates[-1] if candidates else None


def run_yolo_on_frames(pt_path: str, frames_dir: Path, imgsz: int, device: str,
                       conf: float = 0.25) -> dict[str, list[list[float]]]:
    """
    Run YOLO inference on all .jpg files in frames_dir.
    Returns dict {image_stem: [[cx,cy,w,h], ...]} (normalised coordinates).
    """
    from ultralytics import YOLO
    model = YOLO(pt_path)
    img_paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not img_paths:
        return {}

    results_map: dict[str, list[list[float]]] = {}
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
                boxes.append(list(map(float, box[:4])))
        results_map[stem] = boxes
    return results_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True,
                        help="Registered version (e.g. v005) or path to .pt")
    parser.add_argument("--flight",     required=True,
                        help="Flight ID (e.g. 2026-04-20_flight01) or full path")
    parser.add_argument("--labels",     default=None,
                        help="Label subdir name (default: labels_truth > labels_auto_vN)")
    parser.add_argument("--iou-thresh", type=float, default=IOU_DEFAULT)
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--no-save",    dest="no_save", action="store_true")
    args = parser.parse_args()

    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    conn = get_conn()

    # Resolve model
    if args.model.startswith("v") and args.model[1:].isdigit():
        row = conn.execute(
            "SELECT id, pt_path, training_run_id FROM models WHERE version=?",
            (args.model,)
        ).fetchone()
        if not row:
            print(f"ERROR: {args.model} not in registry.db")
            return
        model_id = row["id"]
        pt_path  = row["pt_path"]
        tr = conn.execute(
            "SELECT imgsz FROM training_runs WHERE id=?", (row["training_run_id"],)
        ).fetchone() if row["training_run_id"] else None
        imgsz = tr["imgsz"] if tr and tr["imgsz"] else 1280
    else:
        pt_path  = args.model
        model_id = None
        imgsz    = 1280

    if not pt_path or not Path(pt_path).exists():
        print(f"ERROR: weights not found at {pt_path!r}")
        return

    # Resolve flight dir
    if Path(args.flight).is_dir():
        flight_dir = Path(args.flight)
    else:
        flight_dir = Path("flights") / args.flight
    if not flight_dir.is_dir():
        print(f"ERROR: flight dir not found: {flight_dir}")
        return

    frames_dir = flight_dir / "raw_frames"
    if not frames_dir.is_dir():
        print(f"ERROR: raw_frames/ not found in {flight_dir}")
        return

    label_dir = resolve_label_dir(flight_dir, args.labels)
    if label_dir is None:
        print(f"WARNING: no label directory found in {flight_dir} — only FP counted (no GT).")

    print(f"Model    : {args.model}  ({pt_path})")
    print(f"Flight   : {flight_dir}")
    print(f"Frames   : {frames_dir}")
    print(f"Labels   : {label_dir or 'none'}")
    print(f"IoU thr  : {args.iou_thresh}   conf: {args.conf}")
    print(f"Device   : {args.device}")

    print("\nRunning inference ...")
    preds = run_yolo_on_frames(pt_path, frames_dir, imgsz, args.device, args.conf)

    # Eval per frame
    total_tp = total_fp = total_fn = 0
    frame_rows = []

    for stem, pred_boxes in preds.items():
        if label_dir:
            gt_boxes = parse_yolo_label(label_dir / f"{stem}.txt")
        else:
            gt_boxes = []

        # Check if frame was human-corrected (meta json)
        human_corrected = 0
        auto_generated  = 0
        meta_path = flight_dir / "meta" / f"{stem}.json"
        if meta_path.exists():
            import json
            try:
                meta = json.loads(meta_path.read_text())
                human_corrected = int(bool(meta.get("human_corrected")))
                auto_generated  = int(bool(meta.get("auto_generated")))
            except Exception:
                pass

        tp, fp, fn, avg_iou = match_boxes(pred_boxes, gt_boxes, args.iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        frame_path = str(frames_dir / f"{stem}.jpg")
        frame_rows.append((model_id, frame_path, tp, fp, fn, avg_iou,
                           auto_generated, human_corrected))

    # Summary
    prec   = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    print(f"\nResults over {len(preds)} frames:")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision={prec:.4f}  Recall={recall:.4f}")

    # Save to DB
    if not args.no_save and model_id is not None:
        # Remove old results for this (model, flight) combination
        conn.execute("""
            DELETE FROM frame_results
            WHERE model_id=? AND frame_path LIKE ?
        """, (model_id, str(frames_dir) + "%"))
        conn.executemany("""
            INSERT INTO frame_results
                (model_id, frame_path, TP, FP, FN, iou, auto_generated, human_corrected)
            VALUES (?,?,?,?,?,?,?,?)
        """, frame_rows)
        conn.commit()
        print(f"\n[Registry] {len(frame_rows)} frame_results saved for model_id={model_id}")

    conn.close()


if __name__ == "__main__":
    main()
