#!/bin/bash
# ============================================================
# Pull everything post-flight from RPi in one go.
# Creates: flights/YYYY-MM-DD_flightNN/
# Pulls:   raw_frames/, mission.jsonl, droneDB.db
#
# This extends pull_logs_rpi.sh (which skydock2 uses to pull
# only mission.jsonl). ai_skydock owns the full post-flight
# pipeline — use this instead.
#
# Usage:
#   ./pull_flight.sh MISSION_ID [RPI_HOST] [RPI_USER]
#
# MISSION_ID is the skydock2 missions/NNNN folder on the RPi.
#
# Examples:
#   ./pull_flight.sh 0001
#   ./pull_flight.sh 0001 rpi.local fred
#
# After pulling, auto-label with:
#   python labeling/auto_label.py --flight FLIGHT_ID --stage
#   python add_data.py
# ============================================================

set -e

MISSION_ID="${1}"
RPI_HOST="${2:-rpi.local}"
RPI_USER="${3:-fred}"

# If no MISSION_ID given, auto-detect the latest mission on the RPi
if [ -z "$MISSION_ID" ]; then
    echo "No MISSION_ID given — detecting latest mission on ${RPI_USER}@${RPI_HOST} ..."
    MISSION_ID=$(ssh "${RPI_USER}@${RPI_HOST}" \
        "ls /home/${RPI_USER}/skydock2/missions/ 2>/dev/null | sort -V | tail -1" 2>/dev/null || true)
    if [ -z "$MISSION_ID" ]; then
        echo "ERROR: No missions found on RPi at /home/${RPI_USER}/skydock2/missions/"
        echo "Available missions:"
        ssh "${RPI_USER}@${RPI_HOST}" \
            "ls /home/${RPI_USER}/skydock2/missions/ 2>/dev/null || echo '  (none)'"
        exit 1
    fi
    echo "  Latest mission: ${MISSION_ID}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLIGHTS_DIR="${SCRIPT_DIR}/flights"
mkdir -p "$FLIGHTS_DIR"

# Auto-assign flight ID: YYYY-MM-DD_flightNN
DATE_STR=$(date +%Y-%m-%d)
INDEX=1
while [ -d "${FLIGHTS_DIR}/${DATE_STR}_flight$(printf '%02d' $INDEX)" ]; do
    INDEX=$((INDEX + 1))
done
FLIGHT_ID="${DATE_STR}_flight$(printf '%02d' $INDEX)"
LOCAL_DIR="${FLIGHTS_DIR}/${FLIGHT_ID}"

RPI_MISSION_DIR="/home/${RPI_USER}/skydock2/missions/${MISSION_ID}"

echo "=== pull_flight.sh ==="
echo "  Mission    : ${MISSION_ID} on ${RPI_USER}@${RPI_HOST}"
echo "  Local path : ${LOCAL_DIR}"
echo ""

# Verify mission exists
if ! ssh "${RPI_USER}@${RPI_HOST}" "test -d ${RPI_MISSION_DIR}" 2>/dev/null; then
    echo "ERROR: ${RPI_MISSION_DIR} not found on RPi."
    echo "Available missions:"
    ssh "${RPI_USER}@${RPI_HOST}" \
        "ls /home/${RPI_USER}/skydock2/missions/ 2>/dev/null || echo '  (none)'"
    exit 1
fi

mkdir -p "${LOCAL_DIR}/raw_frames"
mkdir -p "${LOCAL_DIR}/meta"

# ---- Pull frames (tar over SSH — one connection, no per-file overhead) ----
echo "==> Pulling raw_frames/ ..."
FRAME_COUNT_REMOTE=$(ssh "${RPI_USER}@${RPI_HOST}" \
    "ls ${RPI_MISSION_DIR}/frames/*.jpg 2>/dev/null | wc -l" 2>/dev/null || echo 0)
echo "    ${FRAME_COUNT_REMOTE} frames on RPi"

# JPEGs are already compressed — skip -z to avoid wasting CPU with no size benefit.
# Single tar stream over one SSH connection is faster than rsync's per-file overhead on WiFi.
ssh "${RPI_USER}@${RPI_HOST}" \
    "tar -C ${RPI_MISSION_DIR}/frames -cf - \$(ls *.jpg *.jpeg 2>/dev/null)" \
    | tar -xf - -C "${LOCAL_DIR}/raw_frames/"
echo "    OK"

# ---- Pull mission log (text — compress this one) ----
echo "==> Pulling mission.jsonl ..."
rsync -az "${RPI_USER}@${RPI_HOST}:${RPI_MISSION_DIR}/mission.jsonl" \
    "${LOCAL_DIR}/" 2>/dev/null && echo "    OK" || echo "    (not found)"

# ---- Pull drone DB ----
echo "==> Pulling droneDB.db ..."
rsync -a "${RPI_USER}@${RPI_HOST}:${RPI_MISSION_DIR}/droneDB.db" \
    "${LOCAL_DIR}/" 2>/dev/null && echo "    OK" || echo "    (not found)"

# ---- flight_meta.json ----
FRAME_COUNT=$(ls "${LOCAL_DIR}/raw_frames/"*.jpg 2>/dev/null | wc -l)
cat > "${LOCAL_DIR}/flight_meta.json" <<JSON
{
  "flight_id": "${FLIGHT_ID}",
  "mission_id": "${MISSION_ID}",
  "rpi_host": "${RPI_HOST}",
  "rpi_user": "${RPI_USER}",
  "pull_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "frame_count": ${FRAME_COUNT}
}
JSON

echo ""
echo "=== Done ==="
echo "  Flight ID  : ${FLIGHT_ID}"
echo "  Frames     : ${FRAME_COUNT}"
echo "  Local dir  : ${LOCAL_DIR}/"
echo ""
echo "Next:"
echo "  1. Auto-label (set SKYDOCK_YOLO_MODEL or use --model):"
echo "       python labeling/auto_label.py --flight ${FLIGHT_ID} --stage"
echo "  2. Human QA (edit meta/*.json if needed)"
echo "  3. Ingest: python add_data.py"
echo "  4. Train:  python 2_train.py --finetune-from PREV_VERSION"
