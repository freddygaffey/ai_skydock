#!/bin/bash
# ============================================================
# Compile a custom YOLOv8 ONNX → HEF for Hailo-8 (26 TOPS)
#
# IMPORTANT: targets hailo8, NOT hailo8l.
#
# Usage:
#   ./3_compile_hailo8.sh PATH/TO/best.onnx
#
# Example:
#   ./3_compile_hailo8.sh runs/detect/ball_detection_yolov8x/weights/best.onnx
#
# The compiled HEF will be placed next to the ONNX file.
# ============================================================

set -e

ONNX_PATH="${1}"
if [ -z "$ONNX_PATH" ]; then
    echo "Usage: $0 PATH/TO/best.onnx"
    exit 1
fi

ONNX_ABS=$(realpath "$ONNX_PATH")
ONNX_DIR=$(dirname "$ONNX_ABS")
ONNX_NAME=$(basename "$ONNX_ABS" .onnx)
SHARED_DIR="$(realpath hailo8_ai_sw_suite_2025-10_docker/shared_with_docker)"

echo "==> ONNX source : $ONNX_ABS"
echo "==> Output dir  : $ONNX_DIR"
echo "==> Docker share: $SHARED_DIR"

# Copy ONNX and compile script into the shared Docker volume
cp "$ONNX_ABS" "$SHARED_DIR/${ONNX_NAME}.onnx"
cp "$(dirname "$0")/4_compile_inside_docker.py" "$SHARED_DIR/compile_inside_docker.py"

# Copy calibration images — search datasets/vN/ first (new layout), then fallbacks
CALIB_SRC=""
# Search datasets/v*/train/images — pick the highest-numbered version
for candidate in $(ls -d datasets/*/train/images 2>/dev/null | sort -V | tail -1) \
                 merged_dataset/train/images \
                 BALLS-4/train/images; do
    if [ -d "$candidate" ]; then
        CALIB_SRC="$candidate"
        break
    fi
done

if [ -n "$CALIB_SRC" ]; then
    echo "==> Copying calibration images from $CALIB_SRC ..."
    rm -rf "$SHARED_DIR/calib_images"
    mkdir -p "$SHARED_DIR/calib_images"
    # Copy up to 200 images for calibration
    ls "$CALIB_SRC"/*.jpg "$CALIB_SRC"/*.jpeg "$CALIB_SRC"/*.png 2>/dev/null \
        | head -200 \
        | xargs -I{} cp {} "$SHARED_DIR/calib_images/"
    echo "    $(ls "$SHARED_DIR/calib_images" | wc -l) images copied"
else
    echo "WARNING: No calibration images found. Run 1_download_datasets.py first."
    echo "         Compilation will use random data (lower accuracy)."
fi

# Write model name into a config file for the Docker script to read
echo "${ONNX_NAME}" > "$SHARED_DIR/model_name.txt"

# ----------------------------------------------------------------
# Launch (or resume) the Hailo Docker container and run compilation
# ----------------------------------------------------------------
CONTAINER_NAME="hailo8_ai_sw_suite_2025-10_container"
DOCKER_IMAGE="hailo8_ai_sw_suite_2025-10:1"

# Check if container already exists
if [ "$(docker ps -a -q -f name="$CONTAINER_NAME" 2>/dev/null | wc -l)" -ge 1 ]; then
    echo "==> Resuming existing Hailo container ..."
    docker start "$CONTAINER_NAME"
    # Wait until container is actually running
    for i in $(seq 1 20); do
        if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" = "true" ]; then
            break
        fi
        echo "    Waiting for container... ($i)"
        sleep 2
    done
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != "true" ]; then
        echo "ERROR: Container failed to start. Check: docker logs $CONTAINER_NAME"
        exit 1
    fi
    docker exec -i "$CONTAINER_NAME" bash -c \
        "cd /local/shared_with_docker && python3 compile_inside_docker.py"
else
    echo "==> No container found. Starting new container ..."
    echo "    (This will load the Docker image — may take a few minutes first time)"
    cd hailo8_ai_sw_suite_2025-10_docker
    bash hailo_ai_sw_suite_docker_run.sh &
    DOCKER_PID=$!
    # Wait for the container to be running
    echo "    Waiting for container to start..."
    sleep 5
    docker exec -i "$CONTAINER_NAME" bash -c \
        "cd /local/shared_with_docker && python3 compile_inside_docker.py"
    cd ..
fi

# ----------------------------------------------------------------
# Copy the resulting HEF back to the weights directory
# ----------------------------------------------------------------
HEF_SRC="$SHARED_DIR/${ONNX_NAME}.hef"
if [ -f "$HEF_SRC" ]; then
    cp "$HEF_SRC" "$ONNX_DIR/${ONNX_NAME}.hef"
    echo ""
    echo "SUCCESS: HEF written to $ONNX_DIR/${ONNX_NAME}.hef"
    echo "Next: run 5_deploy_to_rpi.sh or copy the HEF to your Raspberry Pi."
else
    echo ""
    echo "ERROR: HEF not found at $HEF_SRC — check compilation logs above."
    exit 1
fi
