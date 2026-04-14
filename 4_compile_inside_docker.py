"""
Runs INSIDE the Hailo AI SW Suite Docker container.
Called automatically by 3_compile_hailo8.sh.

Compiles a custom YOLOv8 ONNX → HEF targeting hailo8 (26 TOPS).

The HEF outputs raw detection tensors; NMS is handled by
libhailo_yolov8_postprocess.so on the Raspberry Pi (already in
the hailo-rpi5-examples pipeline).

End node for YOLOv8 all variants:  /model.22/Concat_3
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

SHARED = Path("/local/shared_with_docker")

# Read model name written by the shell script
model_name_file = SHARED / "model_name.txt"
if not model_name_file.exists():
    print("ERROR: model_name.txt not found in shared_with_docker/")
    sys.exit(1)
MODEL_NAME = model_name_file.read_text().strip()
ONNX_PATH  = SHARED / f"{MODEL_NAME}.onnx"
HAR_PATH   = SHARED / f"{MODEL_NAME}.har"
HAR_Q_PATH = SHARED / f"{MODEL_NAME}_quantized.har"
HEF_PATH   = SHARED / f"{MODEL_NAME}.hef"
CALIB_DIR  = SHARED / "calib_images"
IMGSZ = 1280

print(f"Model name : {MODEL_NAME}")
print(f"ONNX       : {ONNX_PATH}")
print(f"Target HW  : hailo8  (26 TOPS) — NOT hailo8l")

# -----------------------------------------------------------------------
# 1. Load calibration images
#    These are used by the quantizer to minimise accuracy loss from
#    float → int8 conversion.  100–200 representative images is plenty.
# -----------------------------------------------------------------------
def load_calib_images(calib_dir: Path, max_images: int = 150) -> np.ndarray:
    images = []
    if calib_dir.exists():
        for p in sorted(calib_dir.iterdir())[:max_images]:
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                img = Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ))
                arr = np.array(img, dtype=np.float32) / 255.0  # normalise [0,1]
                # Hailo SDK expects HWC layout for calibration data
                images.append(arr)
            except Exception as e:
                print(f"  Skipping {p.name}: {e}")

    if images:
        print(f"Loaded {len(images)} calibration images")
        return np.array(images)

    print("WARNING: No calibration images — using random noise (accuracy will be lower)")
    return (np.random.rand(64, IMGSZ, IMGSZ, 3)).astype(np.float32)


calib_data = load_calib_images(CALIB_DIR)

# -----------------------------------------------------------------------
# 2. Parse ONNX → HAR
# -----------------------------------------------------------------------
from hailo_sdk_client import ClientRunner

print("\n[1/3] Parsing ONNX model ...")
runner = ClientRunner(hw_arch="hailo8")  # hailo8 = 26 TOPS

hn, npz = runner.translate_onnx_model(
    str(ONNX_PATH),
    MODEL_NAME,
    start_node_names=["images"],
    # Use individual conv outputs (not Concat_3) so HailoRT postprocess .so
    # handles NMS on RPi — also allows compiler to fit YOLOv8x on hailo8.
    end_node_names=[
        "/model.22/cv2.0/cv2.0.2/Conv",
        "/model.22/cv3.0/cv3.0.2/Conv",
        "/model.22/cv2.1/cv2.1.2/Conv",
        "/model.22/cv3.1/cv3.1.2/Conv",
        "/model.22/cv2.2/cv2.2.2/Conv",
        "/model.22/cv3.2/cv3.2.2/Conv",
    ],
    net_input_shapes={"images": [1, 3, IMGSZ, IMGSZ]},
)

runner.save_har(str(HAR_PATH))
print(f"HAR saved: {HAR_PATH}  ({HAR_PATH.stat().st_size / 1e6:.1f} MB)")

# -----------------------------------------------------------------------
# 3. Quantize (float32 → int8)
# -----------------------------------------------------------------------
print("\n[2/3] Quantizing (int8 calibration) ...")
runner.load_model_script("performance_param(compiler_optimization_level=max)\n")
runner.optimize(calib_data)
runner.save_har(str(HAR_Q_PATH))
print(f"Quantized HAR saved: {HAR_Q_PATH}  ({HAR_Q_PATH.stat().st_size / 1e6:.1f} MB)")

# -----------------------------------------------------------------------
# 4. Compile → HEF
# -----------------------------------------------------------------------
print("\n[3/3] Compiling to HEF ...")
hef = runner.compile()
with open(str(HEF_PATH), "wb") as f:
    f.write(hef)

print(f"\n=== Compilation complete ===")
print(f"HEF: {HEF_PATH}")
print(f"Size: {HEF_PATH.stat().st_size / 1e6:.1f} MB")
