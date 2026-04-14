#!/bin/bash
# ============================================================
# Deploy HEF + labels to RPi, restart skydock2, health check,
# mark model as deployed in registry.db.
#
# Usage:
#   ./5_deploy_to_rpi.sh MODEL_REGISTRY_VERSION [RPI_HOST] [RPI_USER]
#   ./5_deploy_to_rpi.sh model_registry/v003/best.hef  # also accepts direct path
#
# Examples:
#   ./5_deploy_to_rpi.sh v003
#   ./5_deploy_to_rpi.sh v003 rpi.local fred
#
# The version arg can be:
#   - A version string like "v003"   → resolves to model_registry/v003/best.hef
#   - A direct .hef path
# ============================================================

set -e

VERSION_OR_PATH="${1}"
RPI_HOST="${2:-rpi.local}"
RPI_USER="${3:-fred}"

if [ -z "$VERSION_OR_PATH" ]; then
    echo "Usage: $0 VERSION_OR_HEF_PATH [RPI_HOST] [RPI_USER]"
    echo "  e.g. $0 v003"
    echo "  e.g. $0 model_registry/v003/best.hef"
    exit 1
fi

# Resolve HEF path and version string
if [[ "$VERSION_OR_PATH" == v[0-9]* ]]; then
    VERSION="$VERSION_OR_PATH"
    HEF_PATH="model_registry/${VERSION}/best.hef"
else
    HEF_PATH="$VERSION_OR_PATH"
    # Try to extract version from path like model_registry/v003/best.hef
    VERSION=$(echo "$HEF_PATH" | grep -oP '(?<=model_registry/)v[0-9]+' || true)
fi

if [ ! -f "$HEF_PATH" ]; then
    echo "ERROR: HEF not found at $HEF_PATH"
    echo "  Run ./3_compile_hailo8.sh first."
    exit 1
fi

HEF_ABS=$(realpath "$HEF_PATH")
LABELS_ABS=$(realpath "$(dirname "$0")/ball_labels.json")
RPI_DEST="/home/${RPI_USER}/skydock2"

echo "==> Deploying ${VERSION:-unknown} to ${RPI_USER}@${RPI_HOST}"
echo "    HEF    : $HEF_ABS"
echo "    Labels : $LABELS_ABS"

# ---- Copy files ----
ssh "${RPI_USER}@${RPI_HOST}" "mkdir -p ${RPI_DEST}/models"
scp "$HEF_ABS"    "${RPI_USER}@${RPI_HOST}:${RPI_DEST}/models/ball_detection.hef"
scp "$LABELS_ABS" "${RPI_USER}@${RPI_HOST}:${RPI_DEST}/models/ball_labels.json"

echo "==> Files on RPi:"
ssh "${RPI_USER}@${RPI_HOST}" "ls -lh ${RPI_DEST}/models/"

# ---- Restart skydock2 if systemd service exists ----
RESTART_STATUS=0
if ssh "${RPI_USER}@${RPI_HOST}" "systemctl is-active --quiet skydock2 2>/dev/null"; then
    echo "==> Restarting skydock2 service ..."
    ssh "${RPI_USER}@${RPI_HOST}" "sudo systemctl restart skydock2"
    sleep 3
    if ssh "${RPI_USER}@${RPI_HOST}" "systemctl is-active --quiet skydock2 2>/dev/null"; then
        echo "    skydock2 restarted OK"
    else
        echo "    WARNING: skydock2 did not come back up — check: journalctl -u skydock2 -n 50"
        RESTART_STATUS=1
    fi
else
    echo "    (skydock2 systemd service not active — skipping restart)"
fi

# ---- Mark deployed in registry.db ----
if [ -n "$VERSION" ] && [ -f "registry.db" ]; then
    DEPLOYED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    python3 - <<PYEOF
import sqlite3, sys
from pathlib import Path

conn = sqlite3.connect("registry.db")
conn.row_factory = sqlite3.Row

version = "${VERSION}"
deployed_at = "${DEPLOYED_AT}"

row = conn.execute("SELECT id FROM models WHERE version=?", (version,)).fetchone()
if not row:
    print(f"WARNING: {version} not in registry.db — mark deployed skipped")
    sys.exit(0)

model_id = row["id"]

# Retire previous deployment
conn.execute("""
    UPDATE deployments SET retired_at=?
    WHERE model_id != ? AND retired_at IS NULL
""", (deployed_at, model_id))
conn.execute("""
    UPDATE models SET retired_at=?, deployed=0
    WHERE id != ? AND deployed=1
""", (deployed_at, model_id))

# Mark this version deployed
conn.execute("""
    UPDATE models SET deployed=1, deployed_at=?, retired_at=NULL WHERE id=?
""", (deployed_at, model_id))

conn.execute("""
    INSERT INTO deployments (model_id, deployed_at, notes)
    VALUES (?, ?, 'deployed via 5_deploy_to_rpi.sh')
""", (model_id, deployed_at))

conn.commit()
conn.close()
print(f"registry.db: {version} marked deployed at {deployed_at}")
PYEOF
fi

echo ""
if [ "$RESTART_STATUS" -eq 0 ]; then
    echo "SUCCESS: ${VERSION:-HEF} deployed to ${RPI_HOST}"
else
    echo "PARTIAL: Files copied but skydock2 restart failed — check RPi."
    exit 1
fi
