"""Central defaults for the upscaling pipeline."""

import os
from pathlib import Path

# ── Repo layout ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_RB_DIR = REPO_ROOT / "vendor" / "RealBasicVSR"

# ── RealBasicVSR inference entry‑point ────────────────────────────────────────
INFERENCE_SCRIPT = VENDOR_RB_DIR / "inference_realbasicvsr.py"

# ── Default model artefacts ───────────────────────────────────────────────────
DEFAULT_CONFIG = VENDOR_RB_DIR / "configs" / "realbasicvsr_x4.py"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "RealBasicVSR_x4.pth"

# ── Upscale factor (matches the default x4 model) ────────────────────────────
DEFAULT_SCALE = 4

# ── FFmpeg output codec settings ──────────────────────────────────────────────
# libx264 + yuv420p ensures broad player compatibility.
FFMPEG_VIDEO_CODEC = "libx264"
FFMPEG_PIXEL_FORMAT = "yuv420p"
FFMPEG_CRF = 18           # constant-rate factor: lower = better quality
FFMPEG_PRESET = "slow"    # encoding speed/quality trade-off

# ── Frame image format used in the pipeline temp directory ───────────────────
FRAME_PATTERN = "%08d.png"     # ffmpeg naming pattern
FRAME_GLOB = "*.png"           # glob for reading back extracted frames

# ── Max frames passed to RealBasicVSR in a single forward pass ───────────────
# Reduce this on low-VRAM GPUs (e.g. 8 for 8 GB VRAM).
DEFAULT_MAX_SEQ_LEN = 30
