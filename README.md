# ai_skydock

YOLOv8 ball detection — training, versioning, and deployment pipeline for the Skydock drone platform.

## Architecture

Two-model design:

| Model | Arch | Runs on | Purpose |
|-------|------|---------|---------|
| Edge model | YOLOv8n/s | RPi 5 + Hailo-8 (HEF) | Real-time detection during flight |
| Ground model | YOLOv8x | `deb.local` RTX 4070 | Post-flight auto-labeling |

Training hardware: Intel i7, 36 GB RAM, RTX 4070 Laptop (8 GB VRAM). Camera: 1280×720.

## Pipeline

```bash
# 1. Download + merge Roboflow datasets → datasets/v1/
python 1_download_datasets.py --api-key <KEY>

# 2. Train (auto-detects GPU, auto-tunes batch, exports ONNX on finish)
python 2_train.py
python 2_train.py --resume
python 2_train.py --finetune-from v003

# 3. Compile ONNX → HEF (Linux, requires Hailo Docker)
./3_compile_hailo8.sh model_registry/v001/best.onnx

# 4. Deploy to RPi
./5_deploy_to_rpi.sh v001
```

## Data flywheel

```
Drone flight (RPi + Hailo-8)
  → pull_flight.sh        pull raw_frames/ + mission.jsonl from RPi
  → auto_label.py         ground model labels frames on deb.local
  → add_data.py           validates + ingests to datasets/vN+1/
  → 2_train.py            finetune edge model
  → eval_registry.py      regression gate (block if mAP drops >2%)
  → 5_deploy_to_rpi.sh    compile + deploy new HEF
```

## Key scripts

| Script | Purpose |
|--------|---------|
| `1_download_datasets.py` | Pull 11 Roboflow sources, remap all classes → Ball (0), write datasets/v1/ |
| `2_train.py` | Train or finetune, auto-register in registry.db, export ONNX |
| `3_compile_hailo8.sh` | ONNX → HEF via Hailo Docker container |
| `5_deploy_to_rpi.sh` | SCP HEF to RPi, restart skydock2, mark deployed in registry.db |
| `add_data.py` | Validate staging/ + ingest to new dataset version |
| `eval_registry.py` | Eval all registered models on val set, regression gate |
| `eval_flight.py` | Run any model over any flight, log TP/FP/FN to registry.db |
| `pull_flight.sh` | Rsync raw_frames + mission.jsonl + droneDB.db from RPi |
| `labeling/auto_label.py` | Run ground model on flight frames, write YOLO labels + meta |
| `dashboard.py` | Streamlit dashboard (see below) |

## Dashboard

```bash
source venv/bin/activate
pip install streamlit   # first time only
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. Views:

- **Model History** — mAP50 + FPS over versions
- **Hard Cases** — frames corrected most often, per-model breakdown
- **Version Comparison** — metric delta, fixed vs regressed frames
- **Flight Browser** — scrub frames, overlay labels, mission timeline
- **Dataset History** — images added per version, source breakdown
- **Actions** — upload to staging, deploy to RPi, send frames to deb.local

## Registry

All metrics, history, and deployments tracked in `registry.db` (SQLite). Weights stay on disk.

```
models         — version, mAP50, mAP50_95, precision, recall, fps_rpi_hailo8, deployed
datasets       — version, total_images, images_added, source, date
training_runs  — arch, imgsz, epochs, dataset_id, parent_model_id
frame_results  — per-frame TP/FP/FN/IoU, auto_generated, human_corrected
deployments    — model_id, deployed_at, retired_at
```

## Staging schema

Drop labeled frames here, run `add_data.py` to ingest:

```
staging/
  images/frame001.jpg
  labels/frame001.txt        YOLO format: 0 cx cy w h (normalised)
  meta/frame001.json         label_type, auto_generated, human_reviewed, human_corrected
  batch_info.json            date, source, labeled_by, flight_id
```

Validation aborts (staging untouched) on: missing pairs, bad label format, corrupt images, hash duplicates.

## Model versioning

```
model_registry/
  v001/best.pt  best.onnx  best.hef  training_log.csv
  v002/...
  ground_latest → vNNN      symlink to current ground model
```

Finetune from any version:
```bash
python 2_train.py --finetune-from v003
```

## Flight data

```
flights/
  2026-04-20_flight01/
    raw_frames/          nanosecond-timestamped JPEGs from RPi
    labels_auto_v003/    ground model auto-labels
    labels_truth/        human-verified (authoritative)
    meta/                per-image JSON
    mission.jsonl        FSM events, telemetry, detections
    flight_meta.json     flight_id, mission_id, frame_count
```

## Constraints

- Always `hw_arch="hailo8"` (26 TOPS) — never `hailo8l`
- ONNX export: `opset=11, simplify=True, dynamic=False, imgsz=1280`
- Train at **1280px** — matches camera resolution (1280×720)
- Edge model → HEF only. Ground model never deployed to RPi
- Never edit `~/skydock2` — consume via CLI args only
- `labels_truth/` takes priority over `labels_auto_*/`
- Staging untouched if validation fails
