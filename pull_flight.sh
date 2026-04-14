#!/bin/bash
# ============================================================
# Pull everything post-flight from RPi in one go.
# Creates: flights/YYYY-MM-DD_flightNN/
# Pulls:   raw_frames/, mission.jsonl, database_snapshot.json, droneDB.db
#
# Usage:
#   ./pull_flight.sh                          # interactive: picks mission + asks stride
#   ./pull_flight.sh 0240 rpi.local fred 5    # non-interactive: mission + stride required
#
# Stride N: pull every Nth frame locally. RPi always keeps all frames.
#   Stride 5 on a 16k-frame flight = ~3k frames, ~5x faster over WiFi.
#   Consecutive frames at 30fps are near-identical — stride 5 loses nothing useful.
#
# After pulling, auto-label with:
#   python labeling/auto_label.py --flight FLIGHT_ID --stage
#   python add_data.py
# ============================================================

set -e

MISSION_ID="${1}"
RPI_HOST="${2:-rpi.local}"
RPI_USER="${3:-fred}"
STRIDE="${4:-1}"

# If no MISSION_ID given, list missions from RPi and let user pick
if [ -z "$MISSION_ID" ]; then
    echo "Fetching missions from ${RPI_USER}@${RPI_HOST} ..."

    # Single SSH call — get mission name + frame count for each in one shot
    MISSION_INFO=$(ssh "${RPI_USER}@${RPI_HOST}" "
        BASE=/home/${RPI_USER}/skydock2/missions
        for d in \$(ls \$BASE 2>/dev/null | sort -V); do
            n=\$(ls \$BASE/\$d/frames/ 2>/dev/null | wc -l)
            echo \"\$d \$n\"
        done
    " 2>/dev/null || true)

    if [ -z "$MISSION_INFO" ]; then
        echo "ERROR: No missions found on RPi at /home/${RPI_USER}/skydock2/missions/"
        exit 1
    fi

    echo ""
    i=1
    while IFS=" " read -r m nframes; do
        printf "  %2d)  %s  (%s frames)\n" "$i" "$m" "$nframes"
        i=$((i + 1))
    done <<< "$MISSION_INFO"
    echo ""

    TOTAL=$((i - 1))
    printf "Select mission [1-%d] (default %d): " "$TOTAL" "$TOTAL"
    read -r CHOICE < /dev/tty
    CHOICE="${CHOICE:-$TOTAL}"

    if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || [ "$CHOICE" -lt 1 ] || [ "$CHOICE" -gt "$TOTAL" ]; then
        echo "ERROR: Invalid selection."
        exit 1
    fi

    MISSION_ID=$(awk "NR==${CHOICE} {print \$1}" <<< "$MISSION_INFO")
    echo "  Selected: ${MISSION_ID}"

    # Ask for stride — mandatory, no default
    echo ""
    echo "  RPi keeps all frames. Stride = how many to skip locally."
    echo "  Recommended: 5 (WiFi) or 3 (ethernet). 1 = pull everything."
    while true; do
        printf "Pull every Nth frame (stride): "
        read -r STRIDE_INPUT < /dev/tty
        if [[ "$STRIDE_INPUT" =~ ^[0-9]+$ ]] && [ "$STRIDE_INPUT" -ge 1 ]; then
            STRIDE="$STRIDE_INPUT"
            break
        fi
        echo "  Enter a number >= 1."
    done
    echo ""
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
echo "  Stride     : ${STRIDE} (every ${STRIDE}th frame)"
echo ""

# Verify mission exists
if ! ssh "${RPI_USER}@${RPI_HOST}" "test -d ${RPI_MISSION_DIR}" 2>/dev/null; then
    echo "ERROR: ${RPI_MISSION_DIR} not found on RPi."
    exit 1
fi

mkdir -p "${LOCAL_DIR}/raw_frames"
mkdir -p "${LOCAL_DIR}/meta"

# ---- Pull frames ----
echo "==> Pulling raw_frames/ ..."
FRAME_COUNT_REMOTE=$(ssh "${RPI_USER}@${RPI_HOST}" \
    "ls ${RPI_MISSION_DIR}/frames/ 2>/dev/null | wc -l" 2>/dev/null || echo 0)

if [ "${STRIDE}" -gt 1 ]; then
    FRAME_COUNT_EXPECTED=$(( (FRAME_COUNT_REMOTE + STRIDE - 1) / STRIDE ))
    echo "    ${FRAME_COUNT_REMOTE} frames on RPi → pulling every ${STRIDE}th (~${FRAME_COUNT_EXPECTED} frames)"
else
    echo "    ${FRAME_COUNT_REMOTE} frames on RPi"
fi

# Build file list on RPi (every Nth), pipe through tar, extract locally.
# JPEGs already compressed — no -z. pv shows progress if installed.
if command -v pv &>/dev/null; then
    PROGRESS="pv -pterb"
else
    PROGRESS="cat"
fi

ssh "${RPI_USER}@${RPI_HOST}" "
    cd ${RPI_MISSION_DIR}/frames
    ls | sort -V | awk 'NR % ${STRIDE} == 1 || ${STRIDE} == 1' | tar -cf - -T -
" | ${PROGRESS} | tar -xf - -C "${LOCAL_DIR}/raw_frames/"

echo "    OK"

# ---- Pull mission log (try both names) ----
echo "==> Pulling mission log ..."
rsync -az "${RPI_USER}@${RPI_HOST}:${RPI_MISSION_DIR}/mission.jsonl" \
    "${LOCAL_DIR}/" 2>/dev/null && echo "    mission.jsonl OK" || true
rsync -az "${RPI_USER}@${RPI_HOST}:${RPI_MISSION_DIR}/database_snapshot.json" \
    "${LOCAL_DIR}/" 2>/dev/null && echo "    database_snapshot.json OK" || true

# ---- Pull drone DB ----
echo "==> Pulling droneDB.db ..."
rsync -a "${RPI_USER}@${RPI_HOST}:${RPI_MISSION_DIR}/droneDB.db" \
    "${LOCAL_DIR}/" 2>/dev/null && echo "    OK" || echo "    (not found)"

# ---- flight_meta.json ----
FRAME_COUNT=$(ls "${LOCAL_DIR}/raw_frames/" 2>/dev/null | wc -l)
cat > "${LOCAL_DIR}/flight_meta.json" <<JSON
{
  "flight_id": "${FLIGHT_ID}",
  "mission_id": "${MISSION_ID}",
  "rpi_host": "${RPI_HOST}",
  "rpi_user": "${RPI_USER}",
  "pull_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "frame_count": ${FRAME_COUNT},
  "stride": ${STRIDE}
}
JSON

echo ""
echo "=== Done ==="
echo "  Flight ID  : ${FLIGHT_ID}"
echo "  Frames     : ${FRAME_COUNT} (stride ${STRIDE})"
echo "  Local dir  : ${LOCAL_DIR}/"
echo ""
echo "Next:"
echo "  1. Auto-label:"
echo "       python labeling/auto_label.py --flight ${FLIGHT_ID} --stage"
echo "  2. Ingest: python add_data.py"
echo "  3. Train:  python 2_train.py --finetune-from PREV_VERSION"
