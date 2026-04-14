"""
AI Skydock Dashboard — Streamlit app.

Run:
    source venv/bin/activate
    streamlit run dashboard.py

Views:
  • Model history      — mAP50 + FPS over versions
  • Hard cases tracker — frames corrected most often
  • Version comparison — two versions side-by-side
  • Flight browser     — scrub frames, overlay labels
  • Dataset history    — what was added each version
  • Actions           — upload to staging, deploy, re-label
"""

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

# ─── Paths ───────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent
DB_PATH     = REPO / "registry.db"
DATASETS    = REPO / "datasets"
FLIGHTS     = REPO / "flights"
STAGING     = REPO / "staging"
MODEL_REG   = REPO / "model_registry"

# ─── DB helpers ──────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def db_models() -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = get_conn()
    rows = conn.execute("""
        SELECT m.*, tr.model_arch, tr.imgsz, tr.epochs, tr.duration_mins,
               tr.date as train_date
        FROM models m
        LEFT JOIN training_runs tr ON m.training_run_id = tr.id
        ORDER BY m.id
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_datasets() -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = get_conn()
    rows = conn.execute("SELECT * FROM datasets ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_frame_results(model_id: int) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM frame_results WHERE model_id=?
    """, (model_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_hard_cases(top_n: int = 50) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT frame_path, COUNT(*) as corrections,
               SUM(FN) as total_fn, SUM(FP) as total_fp
        FROM frame_results
        WHERE human_corrected=1
        GROUP BY frame_path
        ORDER BY corrections DESC
        LIMIT ?
    """, (top_n,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_frame_results_for_path(frame_path: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT fr.*, m.version
        FROM frame_results fr
        JOIN models m ON fr.model_id = m.id
        WHERE fr.frame_path=?
        ORDER BY m.id
    """, (frame_path,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Streamlit layout ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Skydock Dashboard",
    page_icon="🎯",
    layout="wide",
)

st.title("AI Skydock Dashboard")

if not DB_PATH.exists():
    st.error(f"registry.db not found at {DB_PATH}. Run `python init_registry.py` first.")
    st.stop()

# Sidebar navigation
page = st.sidebar.radio(
    "View",
    [
        "Model History",
        "Hard Cases",
        "Version Comparison",
        "Flight Browser",
        "Dataset History",
        "Actions",
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# Model History
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Model History":
    st.header("Model History")
    models = db_models()
    if not models:
        st.info("No models in registry yet.")
    else:
        try:
            import pandas as pd
            df = pd.DataFrame(models)
            display_cols = ["version", "model_arch", "imgsz", "mAP50", "mAP50_95",
                            "precision_val", "recall_val", "fps_rpi_hailo8",
                            "deployed", "deployed_at"]
            df_show = df[[c for c in display_cols if c in df.columns]]
            st.dataframe(df_show, use_container_width=True)

            # Line chart: mAP50 over versions
            chart_df = df[df["mAP50"].notna()][["version", "mAP50"]].copy()
            if not chart_df.empty:
                st.subheader("mAP50 over versions")
                st.line_chart(chart_df.set_index("version"))

            # FPS chart
            fps_df = df[df["fps_rpi_hailo8"].notna()][["version", "fps_rpi_hailo8"]].copy()
            if not fps_df.empty:
                st.subheader("RPi Hailo-8 FPS over versions")
                st.line_chart(fps_df.set_index("version"))
        except ImportError:
            st.write(models)

# ═══════════════════════════════════════════════════════════════════════════════
# Hard Cases
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Hard Cases":
    st.header("Hard Cases Tracker")
    st.caption("Frames that needed human correction most often across all batches.")

    top_n = st.slider("Show top N frames", 10, 200, 50)
    rows = db_hard_cases(top_n)
    if not rows:
        st.info("No human-corrected frames recorded yet.")
    else:
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
        except ImportError:
            st.write(rows)

        # Drill-down: pick a frame to see per-model breakdown
        st.subheader("Per-model breakdown")
        frame_paths = [r["frame_path"] for r in rows]
        sel = st.selectbox("Select frame", frame_paths)
        if sel:
            model_rows = db_frame_results_for_path(sel)
            if model_rows:
                try:
                    import pandas as pd
                    st.dataframe(
                        pd.DataFrame(model_rows)[
                            ["version", "TP", "FP", "FN", "iou",
                             "auto_generated", "human_corrected"]
                        ],
                        use_container_width=True,
                    )
                except ImportError:
                    st.write(model_rows)

            # Show image if file exists
            p = Path(sel)
            if p.exists():
                st.image(str(p), caption=p.name)

# ═══════════════════════════════════════════════════════════════════════════════
# Version Comparison
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Version Comparison":
    st.header("Version Comparison")
    models = db_models()
    versions = [m["version"] for m in models]
    if len(versions) < 2:
        st.info("Need at least 2 versions in registry.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            v_a = st.selectbox("Version A", versions, index=0)
        with col2:
            v_b = st.selectbox("Version B", versions, index=min(1, len(versions) - 1))

        m_a = next((m for m in models if m["version"] == v_a), None)
        m_b = next((m for m in models if m["version"] == v_b), None)

        if m_a and m_b:
            st.subheader("Metrics")
            try:
                import pandas as pd
                comp_data = {
                    "Metric": ["mAP50", "mAP50_95", "Precision", "Recall", "FPS (RPi)"],
                    v_a: [
                        m_a.get("mAP50"),    m_a.get("mAP50_95"),
                        m_a.get("precision_val"), m_a.get("recall_val"),
                        m_a.get("fps_rpi_hailo8"),
                    ],
                    v_b: [
                        m_b.get("mAP50"),    m_b.get("mAP50_95"),
                        m_b.get("precision_val"), m_b.get("recall_val"),
                        m_b.get("fps_rpi_hailo8"),
                    ],
                }

                def delta(a, b):
                    if a is None or b is None:
                        return None
                    return round(b - a, 4)

                comp_data["Δ (B−A)"] = [
                    delta(comp_data[v_a][i], comp_data[v_b][i])
                    for i in range(len(comp_data["Metric"]))
                ]
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
            except ImportError:
                st.write({"A": m_a, "B": m_b})

            # Per-frame comparison (if both have frame_results)
            if m_a.get("id") and m_b.get("id"):
                st.subheader("Frame-level comparison")
                conn = get_conn()
                rows_a = {
                    r["frame_path"]: dict(r)
                    for r in conn.execute(
                        "SELECT * FROM frame_results WHERE model_id=?", (m_a["id"],)
                    ).fetchall()
                }
                rows_b = {
                    r["frame_path"]: dict(r)
                    for r in conn.execute(
                        "SELECT * FROM frame_results WHERE model_id=?", (m_b["id"],)
                    ).fetchall()
                }
                conn.close()

                common = set(rows_a) & set(rows_b)
                if common:
                    fixed = []
                    regressed = []
                    for fp in common:
                        a_fn = rows_a[fp]["FN"]
                        b_fn = rows_b[fp]["FN"]
                        if a_fn > 0 and b_fn == 0:
                            fixed.append(fp)
                        elif a_fn == 0 and b_fn > 0:
                            regressed.append(fp)
                    st.metric("Frames fixed by B", len(fixed))
                    st.metric("Regressions in B", len(regressed))

                    if fixed:
                        st.caption(f"Sample fixed frames ({min(5, len(fixed))} shown):")
                        for p_str in fixed[:5]:
                            p = Path(p_str)
                            if p.exists():
                                st.image(str(p), caption=p.name, width=300)
                else:
                    st.info("No overlapping frame_results between these versions.")

# ═══════════════════════════════════════════════════════════════════════════════
# Flight Browser
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Flight Browser":
    st.header("Flight Browser")

    # List available flights
    flight_dirs = sorted(
        [d for d in FLIGHTS.iterdir() if d.is_dir() and not d.name.startswith(".")]
    ) if FLIGHTS.exists() else []

    if not flight_dirs:
        st.info("No flights yet. Run pull_flight.sh to pull from RPi.")
    else:
        flight_names = [d.name for d in flight_dirs]
        sel_flight = st.selectbox("Flight", flight_names, index=len(flight_names) - 1)
        flight_dir = FLIGHTS / sel_flight

        # Show flight metadata
        meta_path = flight_dir / "flight_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            st.caption(
                f"Mission: {meta.get('mission_id')}  |  "
                f"Frames: {meta.get('frame_count')}  |  "
                f"Pulled: {meta.get('pull_date', '')[:10]}"
            )

        # Discover available label sets
        label_dirs: dict[str, Path] = {}
        for d in flight_dir.iterdir():
            if d.is_dir() and (d.name.startswith("labels_") or d.name == "labels_truth"):
                label_dirs[d.name] = d

        label_overlay = st.selectbox(
            "Label overlay", ["none"] + list(label_dirs.keys())
        ) if label_dirs else "none"

        # Frame browser
        frames_dir = flight_dir / "raw_frames"
        if not frames_dir.is_dir():
            st.warning("No raw_frames/ directory in this flight.")
        else:
            frame_files = sorted(
                p for p in frames_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg"}
            )
            if not frame_files:
                st.warning("No frames found.")
            else:
                idx = st.slider("Frame index", 0, len(frame_files) - 1, 0)
                frame_path = frame_files[idx]

                # Show image
                st.image(str(frame_path), caption=frame_path.name, use_container_width=True)

                # Show labels
                if label_overlay != "none":
                    lbl_file = label_dirs[label_overlay] / (frame_path.stem + ".txt")
                    if lbl_file.exists():
                        content = lbl_file.read_text().strip()
                        if content:
                            st.code(content, language="text")
                            st.caption(
                                f"{content.count(chr(10)) + 1} detection(s) in {label_overlay}"
                            )
                        else:
                            st.caption(f"Empty label (no detections) in {label_overlay}")
                    else:
                        st.caption(f"No label file for this frame in {label_overlay}")

                # Per-frame meta
                meta_dir = flight_dir / "meta"
                meta_file = meta_dir / (frame_path.stem + ".json")
                if meta_file.exists():
                    with st.expander("Frame meta"):
                        st.json(json.loads(meta_file.read_text()))

        # Mission log viewer
        log_path = flight_dir / "mission.jsonl"
        if log_path.exists():
            with st.expander("Mission timeline"):
                sys.path.insert(0, str(REPO))
                from flights.analysis import build_timeline_payload, build_summary_payload
                summary = build_summary_payload(log_path)
                st.metric("Duration", f"{summary['duration_s']:.0f}s")
                st.metric("Path length", f"{summary['path_length_m']:.0f}m")

                timeline = build_timeline_payload(log_path)
                if timeline["summary"]:
                    try:
                        import pandas as pd
                        st.dataframe(
                            pd.DataFrame(timeline["summary"])[["state", "total_s", "visits"]],
                            use_container_width=True,
                        )
                    except ImportError:
                        st.write(timeline["summary"])

# ═══════════════════════════════════════════════════════════════════════════════
# Dataset History
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset History":
    st.header("Dataset History")
    datasets = db_datasets()
    if not datasets:
        st.info("No datasets registered yet. Run 1_download_datasets.py first.")
    else:
        try:
            import pandas as pd
            df = pd.DataFrame(datasets)
            st.dataframe(df, use_container_width=True)

            # Bar chart: images added per version
            add_df = df[["version", "images_added"]].dropna()
            if not add_df.empty:
                st.subheader("Images added per version")
                st.bar_chart(add_df.set_index("version"))
        except ImportError:
            st.write(datasets)

        # Show on-disk stats for each version
        st.subheader("On-disk image counts")
        for ver_dir in sorted(DATASETS.iterdir(), key=lambda d: d.name):
            if not ver_dir.is_dir():
                continue
            counts = {}
            for split in ("train", "valid", "test"):
                img_dir = ver_dir / split / "images"
                if img_dir.is_dir():
                    counts[split] = sum(
                        1 for p in img_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    )
            if counts:
                st.caption(
                    f"**{ver_dir.name}**: " +
                    "  ".join(f"{k}: {v}" for k, v in counts.items())
                )

# ═══════════════════════════════════════════════════════════════════════════════
# Actions
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Actions":
    st.header("Actions")

    # ── Upload to staging ───────────────────────────────────────────────────
    st.subheader("Upload frames to staging")
    st.caption(
        "Drop images + labels (+ optional meta JSON) here. "
        "Then fill in batch_info and run add_data.py."
    )

    uploaded_imgs   = st.file_uploader("Images (.jpg/.png)", accept_multiple_files=True,
                                        type=["jpg", "jpeg", "png"])
    uploaded_labels = st.file_uploader("Labels (.txt)", accept_multiple_files=True,
                                        type=["txt"])
    uploaded_meta   = st.file_uploader("Per-image meta (.json)", accept_multiple_files=True,
                                        type=["json"])

    with st.form("batch_info_form"):
        st.caption("batch_info.json")
        b_date   = st.text_input("Date (YYYY-MM-DD)", value=datetime.today().strftime("%Y-%m-%d"))
        b_source = st.selectbox("Source", ["manual", "auto_label", "roboflow"])
        b_by     = st.text_input("Labeled by", value="fred")
        b_notes  = st.text_area("Notes")
        b_flight = st.text_input("Flight ID (optional)")
        b_split  = st.selectbox("Split", ["train", "valid", "test"])
        submit   = st.form_submit_button("Copy to staging")

    if submit:
        if not uploaded_imgs:
            st.error("No images uploaded.")
        else:
            for uf in uploaded_imgs:
                dest = STAGING / "images" / uf.name
                dest.write_bytes(uf.read())
            for uf in uploaded_labels:
                dest = STAGING / "labels" / uf.name
                dest.write_bytes(uf.read())
            for uf in uploaded_meta:
                dest = STAGING / "meta" / uf.name
                dest.write_bytes(uf.read())

            batch_info = {
                "date":       b_date,
                "source":     b_source,
                "labeled_by": b_by,
                "notes":      b_notes,
                "flight_id":  b_flight or None,
                "split":      b_split,
            }
            (STAGING / "batch_info.json").write_text(json.dumps(batch_info, indent=2))

            n_imgs = len(uploaded_imgs)
            st.success(
                f"Staged {n_imgs} image(s). "
                "Now run: `python add_data.py --dry-run` to validate, then `python add_data.py`."
            )

    st.divider()

    # ── Deploy ──────────────────────────────────────────────────────────────
    st.subheader("Deploy to RPi")
    models = db_models()
    versions = [m["version"] for m in models]
    if versions:
        deploy_v = st.selectbox("Version to deploy", versions)
        rpi_host = st.text_input("RPi host", value="raspberrypi.local")
        if st.button("Deploy", type="primary"):
            hef = MODEL_REG / deploy_v / "best.hef"
            if not hef.exists():
                st.error(f"No HEF at {hef}. Run 3_compile_hailo8.sh first.")
            else:
                with st.spinner(f"Deploying {deploy_v} ..."):
                    result = subprocess.run(
                        ["bash", str(REPO / "5_deploy_to_rpi.sh"), deploy_v, rpi_host],
                        capture_output=True, text=True, cwd=str(REPO),
                    )
                if result.returncode == 0:
                    st.success(result.stdout)
                else:
                    st.error(result.stderr or result.stdout)
    else:
        st.info("No models in registry.")

    st.divider()

    # ── Send to deb.local for auto-labeling ─────────────────────────────────
    st.subheader("Send flight to deb.local for auto-labeling")
    st.caption(
        "Rsync raw_frames/ to deb.local, run ground model, rsync labels back."
    )
    flight_dirs = (
        sorted([d.name for d in FLIGHTS.iterdir() if d.is_dir()])
        if FLIGHTS.exists() else []
    )
    if flight_dirs:
        sel = st.selectbox("Flight", flight_dirs, index=len(flight_dirs) - 1, key="deb_flight")
        deb_host = st.text_input("deb.local host", value="deb.local")
        deb_user = st.text_input("deb.local user", value="fred")
        if st.button("Send + Label"):
            flight_dir = FLIGHTS / sel
            frames_dir = flight_dir / "raw_frames"
            deb_dest   = f"/home/{deb_user}/ai_skydock_staging/{sel}/"
            with st.spinner(f"Rsyncing {sel} to {deb_host} ..."):
                result = subprocess.run(
                    ["rsync", "-avz", str(frames_dir) + "/",
                     f"{deb_user}@{deb_host}:{deb_dest}"],
                    capture_output=True, text=True,
                )
            if result.returncode == 0:
                st.success(f"Frames rsynced to {deb_host}:{deb_dest}")
                st.info(
                    f"On deb.local, run:\n"
                    f"  cd ~/ai_skydock && "
                    f"  python labeling/auto_label.py --frames {deb_dest} --stage"
                )
            else:
                st.error(result.stderr or result.stdout)
    else:
        st.info("No flights yet.")
