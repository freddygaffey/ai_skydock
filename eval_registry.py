"""
Evaluate all registered model versions on the fixed validation set and print
a comparison table. Optionally check regression gate vs last deployed model.

Usage:
    source venv/bin/activate
    python eval_registry.py                      # eval all, show table
    python eval_registry.py --version v003       # eval one version only
    python eval_registry.py --gate               # print pass/fail vs deployed
    python eval_registry.py --version v004 --gate
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH  = Path("registry.db")
GATE_THRESHOLD = 0.02   # 2% absolute mAP50 drop triggers block


def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found — run init_registry.py first.")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def latest_val_data() -> str:
    """Return path to the highest-numbered datasets/vN/data.yaml."""
    datasets = Path("datasets")
    if datasets.exists():
        versions = sorted(
            [d for d in datasets.iterdir() if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()],
            key=lambda d: int(d.name[1:]),
        )
        for v in reversed(versions):
            yaml = v / "data.yaml"
            if yaml.exists():
                return str(yaml)
    # Fallback to symlinked merged_dataset
    fallback = Path("merged_dataset/data.yaml")
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("No dataset found. Run 1_download_datasets.py first.")


def run_val(pt_path: str, data_yaml: str, imgsz: int, device: str) -> dict:
    """Run YOLO val on pt_path, return metrics dict."""
    from ultralytics import YOLO
    model = YOLO(pt_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False)
    return {
        "mAP50":    metrics.box.map50,
        "mAP50_95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall":    metrics.box.mr,
    }


def eval_version(conn: sqlite3.Connection, version: str, data_yaml: str,
                 device: str, save: bool = True) -> dict | None:
    row = conn.execute(
        "SELECT id, pt_path, mAP50 FROM models WHERE version=?", (version,)
    ).fetchone()
    if not row:
        print(f"  version {version!r} not found in registry")
        return None

    pt_path = row["pt_path"]
    if not pt_path or not Path(pt_path).exists():
        print(f"  {version}: weights not found at {pt_path!r} — skipping")
        return None

    # Determine imgsz from training_run if available
    tr = conn.execute("""
        SELECT tr.imgsz FROM models m
        LEFT JOIN training_runs tr ON m.training_run_id = tr.id
        WHERE m.version=?
    """, (version,)).fetchone()
    imgsz = tr["imgsz"] if tr and tr["imgsz"] else 1280

    print(f"  Evaluating {version}  ({pt_path}, imgsz={imgsz}) ...")
    try:
        metrics = run_val(pt_path, data_yaml, imgsz, device)
    except Exception as e:
        print(f"  ERROR during eval of {version}: {e}")
        return None

    if save:
        conn.execute("""
            UPDATE models SET mAP50=?, mAP50_95=?, precision_val=?, recall_val=?
            WHERE version=?
        """, (metrics["mAP50"], metrics["mAP50_95"], metrics["precision"],
              metrics["recall"], version))
        conn.commit()

    return {"version": version, **metrics}


def get_deployed_map50(conn: sqlite3.Connection) -> float | None:
    """mAP50 of the most recently deployed model."""
    row = conn.execute("""
        SELECT mAP50 FROM models WHERE deployed=1 ORDER BY deployed_at DESC LIMIT 1
    """).fetchone()
    return row["mAP50"] if row and row["mAP50"] is not None else None


def regression_gate(candidate_map50: float, deployed_map50: float) -> tuple[bool, float]:
    """Returns (passes, delta). Passes when delta >= -GATE_THRESHOLD."""
    delta = candidate_map50 - deployed_map50
    return (delta >= -GATE_THRESHOLD), delta


def print_table(rows: list[dict]):
    if not rows:
        print("No results.")
        return
    header = f"{'Version':<10} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        def fmt(v):
            return f"{v:.4f}" if v is not None else "  N/A  "
        print(f"{r['version']:<10} {fmt(r.get('mAP50')):>8} {fmt(r.get('mAP50_95')):>10} "
              f"{fmt(r.get('precision')):>10} {fmt(r.get('recall')):>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",  default=None, help="Eval one version only")
    parser.add_argument("--gate",     action="store_true",
                        help="Check regression gate vs last deployed model")
    parser.add_argument("--device",   default=None)
    parser.add_argument("--no-save",  dest="no_save", action="store_true",
                        help="Don't update mAP values in registry.db")
    args = parser.parse_args()

    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    conn = get_conn()
    data_yaml = latest_val_data()
    print(f"Val dataset : {data_yaml}")
    print(f"Device      : {args.device}")

    if args.version:
        versions = [args.version]
    else:
        rows_db = conn.execute("SELECT version FROM models ORDER BY id").fetchall()
        versions = [r["version"] for r in rows_db]

    results = []
    for v in versions:
        r = eval_version(conn, v, data_yaml, args.device, save=not args.no_save)
        if r:
            results.append(r)

    print_table(results)

    if args.gate and results:
        deployed_map50 = get_deployed_map50(conn)
        if deployed_map50 is None:
            print("\n[Gate] No deployed model found — gate skipped.")
        else:
            candidate = results[-1]
            passes, delta = regression_gate(candidate["mAP50"], deployed_map50)
            status = "PASS" if passes else "FAIL"
            print(f"\n[Gate] {candidate['version']} vs deployed: Δ mAP50 = {delta:+.4f} → {status}")
            if not passes:
                print(f"       Blocked: drop exceeds {GATE_THRESHOLD*100:.0f}% threshold.")
                import sys
                sys.exit(1)

    conn.close()


if __name__ == "__main__":
    main()
