#!/usr/bin/env bash
# Setup script: clones RealBasicVSR into vendor/ and validates required tools.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor"
RB_DIR="$VENDOR_DIR/RealBasicVSR"
CHECKPOINTS_DIR="$REPO_ROOT/checkpoints"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
error()   { echo -e "${RED}[error]${NC} $*" >&2; }

# ── Prerequisite checks ───────────────────────────────────────────────────────
check_cmd() {
  if ! command -v "$1" &>/dev/null; then
    error "Required tool '$1' not found. Please install it first."
    exit 1
  fi
  info "Found: $(command -v "$1")"
}

info "Checking prerequisites..."
check_cmd python
check_cmd pip
check_cmd ffmpeg
check_cmd ffprobe
check_cmd git

PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python version: $PYTHON_VER"

# ── Clone RealBasicVSR ────────────────────────────────────────────────────────
mkdir -p "$VENDOR_DIR"

if [ -d "$RB_DIR/.git" ]; then
  info "RealBasicVSR already cloned at $RB_DIR — pulling latest..."
  git -C "$RB_DIR" pull --ff-only
else
  info "Cloning RealBasicVSR into $RB_DIR ..."
  git clone https://github.com/ckkelvinchan/RealBasicVSR.git "$RB_DIR"
fi

# ── PyTorch (must come before mmcv-full so mim can detect torch+cuda version) ─
if python -c "import torch" &>/dev/null 2>&1; then
  TORCH_VER=$(python -c "import torch; print(torch.__version__)")
  CUDA_VER=$(python -c "import torch; print(torch.version.cuda or 'cpu')")
  info "PyTorch already installed: torch=$TORCH_VER  cuda=$CUDA_VER"
else
  warn "PyTorch not found — detecting CUDA version to choose the right wheel..."

  # Detect CUDA from nvcc or nvidia-smi
  if command -v nvcc &>/dev/null; then
    CUDA_TAG=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | tr -d '.')
  elif command -v nvidia-smi &>/dev/null; then
    CUDA_TAG=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | tr -d '.')
  else
    CUDA_TAG="cpu"
  fi

  info "Detected CUDA tag: $CUDA_TAG"

  # Map system CUDA version to the closest PyTorch index that supports it.
  # CUDA is backward-compatible: a system with CUDA 12.8 can run cu121 wheels.
  case "$CUDA_TAG" in
    12*)       TORCH_IDX="https://download.pytorch.org/whl/cu121" ;;  # CUDA 12.x → cu121
    118|11.8)  TORCH_IDX="https://download.pytorch.org/whl/cu118" ;;
    117|11.7)  TORCH_IDX="https://download.pytorch.org/whl/cu117" ;;
    116|11.6)  TORCH_IDX="https://download.pytorch.org/whl/cu116" ;;
    113|11.3)  TORCH_IDX="https://download.pytorch.org/whl/cu113" ;;
    cpu)       TORCH_IDX="https://download.pytorch.org/whl/cpu"   ;;
    *)
      warn "Unknown CUDA tag '$CUDA_TAG' — defaulting to cu121."
      TORCH_IDX="https://download.pytorch.org/whl/cu121"
      ;;
  esac

  info "Installing PyTorch from $TORCH_IDX ..."
  pip install torch torchvision --index-url "$TORCH_IDX"
fi

# ── Python dependencies ───────────────────────────────────────────────────────
info "Installing openmim..."
pip install openmim

info "Installing mmcv-full (may take a few minutes on first run)..."
# mim reads the installed torch+cuda version to select the matching mmcv wheel
mim install mmcv-full

info "Installing mmedit..."
pip install mmedit

info "Installing project orchestration dependencies..."
pip install -r "$REPO_ROOT/requirements.txt"

# ── Checkpoints directory ─────────────────────────────────────────────────────
mkdir -p "$CHECKPOINTS_DIR"

CKPT_PATH="$CHECKPOINTS_DIR/RealBasicVSR_x4.pth"
if [ -f "$CKPT_PATH" ]; then
  info "Checkpoint already present: $CKPT_PATH"
else
  warn "Checkpoint not found at $CKPT_PATH"
  echo ""
  echo "  Download the pretrained weights from one of these links and save as:"
  echo "  $CKPT_PATH"
  echo ""
  echo "  Dropbox : https://www.dropbox.com/s/eufigwhejgd8bnt/RealBasicVSR_x4.pth"
  echo "  Google  : https://drive.google.com/file/d/1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID/view"
  echo "  OneDrive: https://entuedu-my.sharepoint.com/:u:/g/personal/chan0899_e_ntu_edu_sg/EVlhWlqBVuxOhjeSqLN9N4UBSbQ5Z-PvYFm2AO5kAGxJSg"
  echo ""
fi

# ── Symlink config and vendor inference script ────────────────────────────────
# Expose the RealBasicVSR configs directory at repo root level for convenience
if [ ! -L "$REPO_ROOT/realbasicvsr_configs" ]; then
  ln -s "$RB_DIR/configs" "$REPO_ROOT/realbasicvsr_configs"
  info "Created symlink: realbasicvsr_configs -> $RB_DIR/configs"
fi

info "Setup complete!"
echo ""
echo "  To upscale a video, run:"
echo "  python src/pipeline/upscale_video.py input.mp4 output.mp4"
echo ""
