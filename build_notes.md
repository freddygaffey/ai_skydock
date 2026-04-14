# Build Notes — AI Skydock

## Decisions

### 2026-04-14 — Stage 1

**imgsz default: 640 → 1280**
Camera is 1280×720. Training at 640 means downscaling on ingest; training at 1280
keeps native resolution. ONNX export also at 1280 to match compile target.
ONNX export constraint from CLAUDE.md (`opset=11, simplify=True, dynamic=False`)
still applies — only imgsz changes.

**v001 baseline import**
`/home/fred/ai_train/best.pt` is YOLOv8x trained at 640px. Imported as v001 in
registry.db with `imgsz=640` to record true training conditions. All future runs
use 1280px from scratch. The .pt is NOT copied — pt_path points to the original
location so the file is not duplicated.

**merged_dataset: symlink not copy**
`/home/fred/ai_train/merged_dataset/` is ~GB of images. Symlinked as
`ai_skydock/merged_dataset` to avoid duplication. `1_download_datasets.py` still
writes new dataset versions to `datasets/v1/` etc.

**ONNX calibration images path**
`3_compile_hailo8.sh` now searches `datasets/v1/train/images` first (new layout),
then falls back to `merged_dataset/train/images` (symlink to ai_train), then the
old `BALLS-4` path. This keeps backward compat while supporting the new structure.

**model_registry/v001 layout**
v001 only has best.pt (no ONNX, no HEF) — it was trained at 640px and is for
reference/finetune-from only. Not compiled to HEF. Noted in registry.db with
`pt_path` set, `hef_path` NULL.

**eval_registry.py design**
Uses fixed val set from current active dataset (latest `datasets/vN/`). Runs
all registered models through YOLO val. Regression gate compares mAP50 vs last
deployed model — blocks if delta < -0.02 (2% absolute drop).

**add_data.py: hash-based dedup**
SHA-256 of image file bytes. Hash index stored in `datasets/vN/hashes.txt` (one
hash per line). On ingest, checks new images against all hashes in all existing
dataset versions. Aborts on any duplicate.

**dashboard.py: standalone, no skydock2 imports**
analysis.py and projection.py copied from skydock2 but stripped of imports that
reference skydock2-internal modules (ai_class, utils, etc.). Functions that need
those are re-implemented using only stdlib + numpy in the copies.

**Ground model symlink: ground_latest**
`model_registry/ground_latest` → symlink to whichever ground model version is
active. Set manually after training a new YOLOv8x ground model.
`SKYDOCK_YOLO_MODEL` env var takes precedence in training_data.py.

**pull_flight.sh flight ID format**
Format: `YYYY-MM-DD_flightNN` where NN is zero-padded 2 digits, auto-incremented
by scanning existing dirs in `flights/`. No user input required for the ID.

**Stage 2 regression gate**
Implemented as a function in `eval_registry.py` (not a separate script) so it
can be called from `2_train.py` post-training. Returns bool + delta. Blocks
deploy if delta < -0.02 on the fixed val set. Hard cases (human_corrected=1)
checked separately — same threshold.
