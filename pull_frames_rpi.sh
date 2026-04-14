#!/bin/bash
# ============================================================
# Pull a single flight's frames (and logs) from the RPi.
# Creates: flights/YYYY-MM-DD_flightNN/
#
# Usage:
#   ./pull_frames_rpi.sh MISSION_ID [RPI_HOST] [RPI_USER]
#
# MISSION_ID is the skydock2 mission folder name on the RPi,
# e.g. "0001" or "0042".  The local flight ID is auto-assigned.
#
# Examples:
#   ./pull_frames_rpi.sh 0001
#   ./pull_frames_rpi.sh 0042 rpi.local fred
# ============================================================

set -e

MISSION_ID="${1}"
RPI_HOST="${2:-rpi.local}"
RPI_USER="${3:-fred}"

if [ -z "$MISSION_ID" ]; then
    echo "Usage: $0 MISSION_ID [RPI_HOST] [RPI_USER]"
    echo "  e.g. $0 0001"
    exit 1
fi

# Auto-assign local flight ID: YYYY-MM-DD_flightNN
DATE_STR=$(date +%Y-%m-%d)
FLIGHTS_DIR="$(dirname "$0")/flights"
mkdir -p "$FLIGHTS_DIR"

# Find next available flightNN index
INDEX=1
while [ -d "${FLIGHTS_DIR}/${DATE_STR}_flight$(printf '%02d' $INDEX)" ]; do
    INDEX=$((INDEX + 1))
done
FLIGHT_ID="${DATE_STR}_flight$(printf '%02d' $INDEX)"
LOCAL_DIR="${FLIGHTS_DIR}/${FLIGHT_ID}"

echo "==> Mission   : ${MISSION_ID} on ${RPI_USER}@${RPI_HOST}"
echo "==> Local dir : ${LOCAL_DIR}"

mkdir -p "${LOCAL_DIR}/raw_frames"
mkdir -p "${LOCAL_DIR}/meta"

RPI_MISSION_DIR="/home/${RPI_USER}/skydock2/missions/${MISSION_ID}"

# Check mission exists on RPi
if ! ssh "${RPI_USER}@${RPI_HOST}" "test -d ${RPI_MISSION_DIR}"; then
    echo "ERROR: ${RPI_MISSION_DIR} not found on RPi."
    echo "  Available missions:"
    ssh "${RPI_USER}@${RPI_HOST}" "ls /home/${RPI_USER}/skydock2/missions/ 2>/dev/null || echo '  (none)'"
    exit 1
fi

# Pull frames (timestamp_ns.jpg)
echo "==> Pulling raw frames ..."
rsync -avz --progress \
    --include="*.jpg" --include="*.jpeg" \
    --exclude="*" \
    "${RPI_USER}@${RPI_HOST}:${RPI_MISSION_DIR}/frames/" \
    "${LOCAL_DIR}/raw_frames/"

# Pull mission log + DB
echo "==> Pulling mission logs ..."
rsync -avz --progress \
    "${RPI_USER}@${RPI_HOST}:${RPI_MISSION_DIR}/mission.jsonl" \
    "${LOCAL_DIR}/" 2>/dev/null || echo "  (no mission.jsonl)"

rsync -avz --progress \
    "${RPI_USER}@${RPI_HOST}:${RPI_MISSION_DIR}/droneDB.db" \
    "${LOCAL_DIR}/" 2>/dev/null || echo "  (no droneDB.db)"

# Write flight_meta.json
FRAME_COUNT=$(ls "${LOCAL_DIR}/raw_frames/"*.jpg 2>/dev/null | wc -l)
FLIGHT_META="${LOCAL_DIR}/flight_meta.json"
cat > "$FLIGHT_META" <<JSON
{
  "flight_id": "${FLIGHT_ID}",
  "mission_id": "${MISSION_ID}",
  "rpi_host": "${RPI_HOST}",
  "pull_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "frame_count": ${FRAME_COUNT}
}
JSON

echo ""
echo "==> Pulled ${FRAME_COUNT} frames to ${LOCAL_DIR}/raw_frames/"
echo "    Flight ID : ${FLIGHT_ID}"
echo ""
echo "Next steps:"
echo "  1. Auto-label: set SKYDOCK_YOLO_MODEL and run labeling/auto_label.py"
echo "     python labeling/auto_label.py --flight ${FLIGHT_ID}"
echo "  2. Human QA: review staging/meta/*.json"
echo "  3. Ingest: python add_data.py"
