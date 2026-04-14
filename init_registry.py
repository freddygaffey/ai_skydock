"""
Initialise registry.db with the full schema and import v001 baseline model.

Run once:
    python init_registry.py

Safe to re-run — creates tables only if they don't exist, imports v001 only if
the models table is empty.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("registry.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    version     TEXT    NOT NULL UNIQUE,   -- e.g. "v1", "v2"
    date        TEXT    NOT NULL,
    total_images INTEGER,
    images_added INTEGER,
    source      TEXT,                      -- "roboflow", "manual", "auto_label"
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS training_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL,
    model_arch      TEXT    NOT NULL,      -- "yolov8n", "yolov8x", etc.
    imgsz           INTEGER NOT NULL,
    epochs          INTEGER,
    duration_mins   REAL,
    dataset_id      INTEGER REFERENCES datasets(id),
    parent_model_id INTEGER REFERENCES models(id)
);

CREATE TABLE IF NOT EXISTS models (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    version         TEXT    NOT NULL UNIQUE,  -- "v001", "v002", etc.
    training_run_id INTEGER REFERENCES training_runs(id),
    mAP50           REAL,
    mAP50_95        REAL,
    precision_val   REAL,
    recall_val      REAL,
    fps_rpi_hailo8  REAL,
    deployed        INTEGER NOT NULL DEFAULT 0,  -- 0/1 boolean
    deployed_at     TEXT,
    retired_at      TEXT,
    pt_path         TEXT,
    hef_path        TEXT
);

CREATE TABLE IF NOT EXISTS deployments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    INTEGER NOT NULL REFERENCES models(id),
    deployed_at TEXT    NOT NULL,
    retired_at  TEXT,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS frame_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        INTEGER NOT NULL REFERENCES models(id),
    frame_path      TEXT    NOT NULL,
    TP              INTEGER NOT NULL DEFAULT 0,
    FP              INTEGER NOT NULL DEFAULT 0,
    FN              INTEGER NOT NULL DEFAULT 0,
    iou             REAL,
    auto_generated  INTEGER NOT NULL DEFAULT 0,
    human_corrected INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_frame_results_model    ON frame_results(model_id);
CREATE INDEX IF NOT EXISTS idx_frame_results_frame    ON frame_results(frame_path);
CREATE INDEX IF NOT EXISTS idx_frame_results_corrected ON frame_results(human_corrected);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    print(f"Schema created / verified: {db_path}")
    return conn


def import_v001(conn: sqlite3.Connection):
    existing = conn.execute("SELECT id FROM models WHERE version='v001'").fetchone()
    if existing:
        print("v001 already in registry — skipping import.")
        return

    pt_path = Path("model_registry/v001/best.pt")
    if not pt_path.exists():
        print(f"WARNING: {pt_path} not found — v001 not imported.")
        return

    # v001 was trained externally at 640px; no training_run row needed.
    conn.execute("""
        INSERT INTO models (version, mAP50, mAP50_95, deployed, pt_path)
        VALUES ('v001', NULL, NULL, 0, ?)
    """, (str(pt_path),))
    conn.commit()
    print(f"Imported v001 → {pt_path}  (mAP unknown — trained externally at 640px)")


if __name__ == "__main__":
    conn = init_db(DB_PATH)
    import_v001(conn)
    conn.close()
    print("\nDone. registry.db ready.")
