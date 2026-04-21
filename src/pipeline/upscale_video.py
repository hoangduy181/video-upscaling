"""
upscale_video.py — Single-video upscaling pipeline.

Pipeline:
  1. Extract source frames to a temp directory (ffmpeg)
  2. Run RealBasicVSR inference on those frames (subprocess)
  3. Assemble upscaled frames into a video (ffmpeg)
  4. Mux original audio back into the final video (ffmpeg)

---- As a callable function ----

  from src.pipeline.upscale_video import upscale_video

  upscale_video("input.mp4", "output.mp4")

  # All parameters are optional and have sensible defaults:
  upscale_video(
      input_path="input.mp4",
      output_path="output_4x.mp4",
      config="/path/to/realbasicvsr_x4.py",   # default: vendor config
      checkpoint="/path/to/weights.pth",       # default: checkpoints/RealBasicVSR_x4.pth
      max_seq_len=30,                          # reduce for low-VRAM GPUs
      scale=4,                                 # used for verification output only
      keep_frames=False,                       # set True to inspect intermediate frames
      work_dir="/tmp/my_workspace",            # default: system temp dir
  )

---- As a CLI ----

  python src/pipeline/upscale_video.py INPUT OUTPUT [OPTIONS]

  Run with --help for full option list.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_SCALE,
    FFMPEG_CRF,
    FFMPEG_PIXEL_FORMAT,
    FFMPEG_PRESET,
    FFMPEG_VIDEO_CODEC,
    FRAME_GLOB,
    FRAME_PATTERN,
    INFERENCE_SCRIPT,
    REPO_ROOT,
    VENDOR_RB_DIR,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list[str], *, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming its output, and raise on non-zero exit."""
    print(f"\n[cmd] {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        raise RuntimeError(f"[error] Command failed with exit code {result.returncode}:\n  {' '.join(str(c) for c in cmd)}")
    return result


def probe_video(path: Path) -> dict:
    """Return video metadata from ffprobe (dict with fps, duration, has_audio)."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(path),
    ]
    result = run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    streams = info.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video_stream is None:
        raise RuntimeError(f"[error] No video stream found in {path}")

    # Parse fractional fps string like "25/1" or "30000/1001"
    fps_str = video_stream.get("r_frame_rate", "25/1")
    num, den = map(int, fps_str.split("/"))
    fps = num / den

    return {
        "fps": fps,
        "fps_str": fps_str,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "duration": float(video_stream.get("duration", 0)),
        "has_audio": audio_stream is not None,
    }


# ── Preflight checks ──────────────────────────────────────────────────────────

def preflight(
    input_path: Path,
    output_path: Path,
    config: str,
    checkpoint: str,
) -> None:
    """Validate all inputs and tool availability before doing any work.

    Raises RuntimeError listing all problems found so the caller can
    handle it without calling sys.exit().
    """
    errors: list[str] = []

    for tool in ("ffmpeg", "ffprobe", "python"):
        if not shutil.which(tool):
            errors.append(f"Required tool not found on PATH: {tool}")

    if not input_path.exists():
        errors.append(f"Input file not found: {input_path}")

    config_path = Path(config)
    if not config_path.exists():
        errors.append(
            f"RealBasicVSR config not found: {config_path}\n"
            f"  Expected at: {DEFAULT_CONFIG}\n"
            f"  Run scripts/setup_realbasicvsr.sh to clone the vendor repo."
        )

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        errors.append(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"  Download it from the links shown by scripts/setup_realbasicvsr.sh\n"
            f"  and save it to checkpoints/RealBasicVSR_x4.pth"
        )

    if not INFERENCE_SCRIPT.exists():
        errors.append(
            f"RealBasicVSR inference script not found: {INFERENCE_SCRIPT}\n"
            f"  Run scripts/setup_realbasicvsr.sh first."
        )

    # Check that mmedit is importable in the subprocess Python environment
    python_bin = shutil.which("python") or "python"
    mmedit_check = subprocess.run(
        [python_bin, "-c", "import mmedit"],
        capture_output=True,
    )
    if mmedit_check.returncode != 0:
        errors.append(
            "Python package 'mmedit' is not installed.\n"
            "  Fix: pip install mmedit\n"
            "  Or re-run: bash scripts/setup_realbasicvsr.sh"
        )

    output_parent = output_path.parent
    if not output_parent.exists():
        try:
            output_parent.mkdir(parents=True)
        except OSError as exc:
            errors.append(f"Cannot create output directory {output_parent}: {exc}")

    if errors:
        msg = "\n[preflight] Found the following issues:\n" + "".join(
            f"\n  - {e}" for e in errors
        ) + "\n"
        raise RuntimeError(msg)

    print("[preflight] All checks passed.")


# ── Pipeline stages ───────────────────────────────────────────────────────────

def extract_frames(video: Path, frames_dir: Path, fps: float) -> int:
    """Extract all frames from *video* into *frames_dir* as PNG files."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y",
        "-i", str(video),
        "-fps_mode", "passthrough",   # no frame duplication or dropping
        str(frames_dir / FRAME_PATTERN),
    ])
    count = len(list(frames_dir.glob(FRAME_GLOB)))
    print(f"[extract] {count} frames extracted → {frames_dir}")
    return count


def run_inference(
    frames_dir: Path,
    upscaled_dir: Path,
    config: str,
    checkpoint: str,
    max_seq_len: int,
) -> None:
    """Call the RealBasicVSR inference script on the extracted frames."""
    upscaled_dir.mkdir(parents=True, exist_ok=True)

    # Build PYTHONPATH so realbasicvsr package inside the vendor dir is found
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{VENDOR_RB_DIR}{os.pathsep}{existing_pp}" if existing_pp
        else str(VENDOR_RB_DIR)
    )

    run([
        "python", str(INFERENCE_SCRIPT),
        config,
        checkpoint,
        str(frames_dir),
        str(upscaled_dir),
        f"--max_seq_len={max_seq_len}",
        "--is_save_as_png=True",
    ], env=env)

    count = len(list(upscaled_dir.glob(FRAME_GLOB)))
    print(f"[inference] {count} upscaled frames → {upscaled_dir}")


def assemble_video(
    upscaled_dir: Path,
    output_video: Path,
    fps: float,
    source_video: Path,
    has_audio: bool,
) -> None:
    """Encode upscaled frames → video and mux audio from the original source."""
    # Step 1: encode frames to a no-audio intermediate
    silent_video = output_video.with_suffix(".silent.mp4")
    run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(upscaled_dir / FRAME_PATTERN),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # ensure even dimensions
        "-c:v", FFMPEG_VIDEO_CODEC,
        "-crf", str(FFMPEG_CRF),
        "-preset", FFMPEG_PRESET,
        "-pix_fmt", FFMPEG_PIXEL_FORMAT,
        str(silent_video),
    ])

    if has_audio:
        # Step 2: mux original audio
        run([
            "ffmpeg", "-y",
            "-i", str(silent_video),
            "-i", str(source_video),
            "-map", "0:v:0",    # video from upscaled
            "-map", "1:a:0",    # audio from original
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",        # trim to shorter stream (video)
            str(output_video),
        ])
        silent_video.unlink()
    else:
        silent_video.rename(output_video)

    print(f"[assemble] Output video written → {output_video}")


def verify_output(input_path: Path, output_path: Path, scale: int) -> None:
    """Print a side-by-side comparison of input vs output metadata."""
    src = probe_video(input_path)
    dst = probe_video(output_path)
    print("\n── Verification ───────────────────────────────────────────")
    print(f"  Source  : {src['width']}×{src['height']}  "
          f"{src['fps']:.3f} fps  duration={src['duration']:.2f}s")
    print(f"  Output  : {dst['width']}×{dst['height']}  "
          f"{dst['fps']:.3f} fps  duration={dst['duration']:.2f}s")
    expected_w = src["width"] * scale
    expected_h = src["height"] * scale
    if dst["width"] == expected_w and dst["height"] == expected_h:
        print(f"  Scale   : ✓  {scale}x upscale confirmed ({expected_w}×{expected_h})")
    else:
        print(f"  Scale   : ⚠  Expected {expected_w}×{expected_h}, got "
              f"{dst['width']}×{dst['height']}")
    print("───────────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upscale a video using RealBasicVSR + ffmpeg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to the source video file.")
    parser.add_argument("output", type=Path, help="Path for the upscaled output video.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="RealBasicVSR config file (.py).",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="RealBasicVSR checkpoint file (.pth).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        dest="max_seq_len",
        help="Max frames per inference batch. Reduce on low-VRAM GPUs.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help="Expected upscale factor (used for verification only).",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        default=False,
        help="Keep extracted and upscaled frame directories after completion.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Custom directory for temp frames. Defaults to a system temp dir.",
    )
    return parser.parse_args()


def upscale_video(
    input_path: str | Path,
    output_path: str | Path,
    config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    scale: int = DEFAULT_SCALE,
    keep_frames: bool = False,
    work_dir: str | Path | None = None,
) -> Path:
    """Upscale a single video using RealBasicVSR + ffmpeg.

    Args:
        input_path:   Path to the source video file.
        output_path:  Path for the upscaled output video.
        config:       RealBasicVSR config (.py). Defaults to the vendor x4 config.
        checkpoint:   Model checkpoint (.pth). Defaults to checkpoints/RealBasicVSR_x4.pth.
        max_seq_len:  Max frames per inference batch. Reduce on low-VRAM GPUs (e.g. 8).
        scale:        Expected upscale factor — used only for the verification summary.
        keep_frames:  If True, extracted and upscaled frame dirs are not deleted.
        work_dir:     Custom directory for temp frames. Defaults to a system temp dir.

    Returns:
        Path to the finished output video.

    Raises:
        RuntimeError: If preflight validation fails.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    config = str(config) if config else str(DEFAULT_CONFIG)
    checkpoint = str(checkpoint) if checkpoint else str(DEFAULT_CHECKPOINT)

    print("=" * 60)
    print("  RealBasicVSR + FFmpeg Upscaling Pipeline")
    print("=" * 60)
    print(f"  Input      : {input_path}")
    print(f"  Output     : {output_path}")
    print(f"  Config     : {config}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  MaxSeqLen  : {max_seq_len}")
    print("=" * 60)

    preflight(input_path, output_path, config, checkpoint)

    meta = probe_video(input_path)
    print(f"\n[probe] Source: {meta['width']}×{meta['height']} @ "
          f"{meta['fps']:.3f} fps, audio={meta['has_audio']}")

    if work_dir:
        _work_dir = Path(work_dir)
        _work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        _work_dir = Path(tempfile.mkdtemp(prefix="rbvsr_"))
        cleanup = not keep_frames

    frames_dir = _work_dir / "source_frames"
    upscaled_dir = _work_dir / "upscaled_frames"

    try:
        print("\n── Stage 1/3: Extract frames ──────────────────────────────")
        extract_frames(input_path, frames_dir, meta["fps"])

        print("\n── Stage 2/3: RealBasicVSR inference ──────────────────────")
        run_inference(
            frames_dir=frames_dir,
            upscaled_dir=upscaled_dir,
            config=config,
            checkpoint=checkpoint,
            max_seq_len=max_seq_len,
        )

        print("\n── Stage 3/3: Assemble output video ────────────────────────")
        assemble_video(
            upscaled_dir=upscaled_dir,
            output_video=output_path,
            fps=meta["fps"],
            source_video=input_path,
            has_audio=meta["has_audio"],
        )

        verify_output(input_path, output_path, scale)

    finally:
        if cleanup and _work_dir.exists():
            shutil.rmtree(_work_dir, ignore_errors=True)
            print(f"[cleanup] Temp directory removed: {_work_dir}")
        elif keep_frames:
            print(f"[cleanup] Frames kept at: {_work_dir}")

    return output_path


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    try:
        upscale_video(
            input_path=args.input,
            output_path=args.output,
            config=args.config,
            checkpoint=args.checkpoint,
            max_seq_len=args.max_seq_len,
            scale=args.scale,
            keep_frames=args.keep_frames,
            work_dir=args.work_dir,
        )
    except RuntimeError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
