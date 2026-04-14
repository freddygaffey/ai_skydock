# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Train a YOLOv8x ball detection model (single class: `Ball`) and deploy it to a Raspberry Pi 5 with a **Hailo-8 (26 TOPS)** AI accelerator. The compiled HEF is consumed by `~/skydock2` via CLI args — do not edit skydock2 code.

## Training hardware

Dedicated laptop: Intel i7, 36 GB RAM, **NVIDIA RTX 4070 Laptop GPU (8 GB VRAM)**. Primary training platform for ALL models — both edge (YOLOv8n/s) and ground (YOLOv8x). CUDA available. Use batch 8–12 for YOLOv8x at 1280px.

## Pipeline (run in order)

```bash
# 1. Download and merge datasets from Roboflow
source venv/bin/activate
python 1_download_datasets.py --api-key 93VUb4KNdeuAHfQm6PqQ

# 2. Train locally (auto-detects GPU/MPS/CPU, auto-tunes batch size)
python 2_train.py
python 2_train.py --resume        # resume after interruption
python 2_train.py --batch 8       # override batch size

# 3. Compile ONNX → HEF (Linux only, needs Hailo Docker)
./3_compile_hailo8.sh runs/detect/ball_detection_yolov8x/weights/best.onnx

# 4. Deploy to Raspberry Pi
./5_deploy_to_rpi.sh runs/detect/ball_detection_yolov8x/weights/best.hef
```

## Cloud training (Kaggle)

Notebook: `kaggle_train.ipynb` — upload to Kaggle with the `ball-data-set` dataset attached.

- Dataset path on Kaggle: `/kaggle/input/datasets/fredgaffey/ball-data-set/merged_dataset/data.yaml`
- Checkpoints back up to `/kaggle/working/weights/` after every epoch and persist in the Output tab after session expiry
- Re-running Cell 3 auto-resumes from last checkpoint

Also available: `colab_train.ipynb` — same training but saves checkpoints to Google Drive.

## Architecture

### Training flow
`1_download_datasets.py` pulls 11 Roboflow datasets, remaps all classes → class 0 (`Ball`), and writes `merged_dataset/` with `train/valid/test` splits.

`2_train.py` auto-detects device (CUDA/MPS/CPU), benchmarks batch sizes, trains YOLOv8x, and exports to ONNX on completion. OOM retry loop halves batch automatically.

### Compilation flow
`3_compile_hailo8.sh` copies the ONNX + calibration images into `hailo8_ai_sw_suite_2025-10_docker/shared_with_docker/`, then runs `4_compile_inside_docker.py` inside the Hailo Docker container.

`4_compile_inside_docker.py` runs the Hailo DFC pipeline: ONNX → HAR → quantized HAR → HEF. End node is `/model.22/Concat_3` (raw detection maps, NMS handled by the RPi postprocess .so at runtime). Target is `hw_arch="hailo8"` — **not hailo8l**.

### Deployment
`5_deploy_to_rpi.sh` SCPs `best.hef` and `ball_labels.json` to the RPi. The app is launched with:
```bash
python3 main.py --hef-path models/ball_detection.hef --labels-json models/ball_labels.json
```

## RPi mission directory structure

Missions live at `~/skydock2/missions/NNNN/` on the RPi (`fred@rpi.local`).

```
missions/
  0240/
    frames/
      1775974450342394577.jpg   ← nanosecond-timestamp filenames
      1775974450370064503.jpg
      ...
    droneDB.db
    database_snapshot.json      ← mission log (NOT mission.jsonl — that name is not used)
```

`pull_flight.sh` handles this: tars `frames/` over SSH in one connection, pulls both
`database_snapshot.json` and `droneDB.db`. Frame filenames are nanosecond Unix timestamps.

## Key constraints

- **hailo8 vs hailo8l**: HEF files are not interchangeable. Always target `hailo8` (26 TOPS). Hailo compilation must happen on the Linux PC (Docker container).
- **ONNX export settings**: `opset=11, simplify=True, dynamic=False, imgsz=640` — required for Hailo DFC compatibility.
- **DDP batch splitting**: When training with `device='0,1'` on Kaggle, each GPU gets `batch/2`. Use `batch=20` for dual T4s (15 GB VRAM each).
- **workers**: Use `workers=2` on Apple MPS, `workers=4–8` on CUDA.
- **Do not edit skydock2**: The model is consumed by `~/skydock2` via CLI args only.
