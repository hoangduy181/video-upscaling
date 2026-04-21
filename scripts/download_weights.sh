#!/usr/bin/env bash
# Download the RealBasicVSR x4 pretrained checkpoint and place it at
# checkpoints/RealBasicVSR_x4.pth — ready to use with the pipeline.
#
# Download sources tried in order:
#   1. Dropbox  — direct link, no auth required (fastest)
#   2. gdown    — Google Drive via pip tool (fallback if Dropbox fails)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINTS_DIR="$REPO_ROOT/checkpoints"
CKPT_FILE="$CHECKPOINTS_DIR/RealBasicVSR_x4.pth"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[weights]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}   $*"; }
error() { echo -e "${RED}[error]${NC}  $*" >&2; }

DROPBOX_URL="https://www.dropbox.com/s/eufigwhejgd8bnt/RealBasicVSR_x4.pth?dl=1"
GDRIVE_FILE_ID="1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID"

mkdir -p "$CHECKPOINTS_DIR"

# ── Already downloaded? ───────────────────────────────────────────────────────
if [ -f "$CKPT_FILE" ]; then
  SIZE=$(du -h "$CKPT_FILE" | cut -f1)
  info "Checkpoint already present ($SIZE): $CKPT_FILE"
  info "Nothing to do. Delete the file and re-run to force a fresh download."
  exit 0
fi

# ── Source 1: Dropbox (direct download, no auth) ──────────────────────────────
download_dropbox() {
  if command -v wget &>/dev/null; then
    info "Downloading via wget from Dropbox..."
    wget --show-progress -O "$CKPT_FILE" "$DROPBOX_URL"
  elif command -v curl &>/dev/null; then
    info "Downloading via curl from Dropbox..."
    curl -L --progress-bar -o "$CKPT_FILE" "$DROPBOX_URL"
  else
    error "Neither wget nor curl is available."
    return 1
  fi
}

# ── Source 2: Google Drive via gdown ─────────────────────────────────────────
download_gdrive() {
  if ! python -c "import gdown" &>/dev/null 2>&1; then
    info "Installing gdown..."
    pip install -q gdown
  fi
  info "Downloading via gdown from Google Drive (file ID: $GDRIVE_FILE_ID)..."
  python -m gdown "$GDRIVE_FILE_ID" -O "$CKPT_FILE"
}

# ── Try sources in order ──────────────────────────────────────────────────────
info "Target: $CKPT_FILE"
echo ""

if download_gdrive; then
  : # success
elif download_dropbox; then
  : # success
else
  error "All download sources failed."
  echo ""
  echo "  Please download the checkpoint manually from one of these links"
  echo "  and save it to:  $CKPT_FILE"
  echo ""
  echo "  Dropbox : https://www.dropbox.com/s/eufigwhejgd8bnt/RealBasicVSR_x4.pth"
  echo "  Google  : https://drive.google.com/file/d/1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID/view"
  echo "  OneDrive: https://entuedu-my.sharepoint.com/:u:/g/personal/chan0899_e_ntu_edu_sg/EVlhWlqBVuxOhjeSqLN9N4UBSbQ5Z-PvYFm2AO5kAGxJSg"
  echo ""
  exit 1
fi

# ── Verify the file looks like a real checkpoint ──────────────────────────────
SIZE_BYTES=$(stat -c%s "$CKPT_FILE" 2>/dev/null || stat -f%z "$CKPT_FILE")
SIZE_MB=$(( SIZE_BYTES / 1024 / 1024 ))

if [ "$SIZE_MB" -lt 50 ]; then
  error "Downloaded file is suspiciously small (${SIZE_MB} MB). It may be an error page."
  rm -f "$CKPT_FILE"
  exit 1
fi

echo ""
info "Done! Checkpoint saved: $CKPT_FILE (${SIZE_MB} MB)"
info "You can now run: python src/pipeline/upscale_video.py input.mp4 output.mp4"
