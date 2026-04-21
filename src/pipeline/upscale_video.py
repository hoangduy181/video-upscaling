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

# Sentinel value used to trigger automatic max_seq_len detection
_AUTO = "auto"


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


# ── Auto max_seq_len ──────────────────────────────────────────────────────────

def auto_max_seq_len(width: int, height: int, device: str = "cuda:0") -> int:
    """Estimate a safe max_seq_len from available VRAM and frame resolution.

    Uses a conservative empirical formula:
      VRAM per frame ≈ pixels × 3 channels × 4 bytes × 60× feature-map overhead

    Leaves 1.5 GB headroom and clamps the result to [1, 120].
    Falls back to 8 when torch is unavailable or the device is CPU.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        print("[auto] torch not importable here — defaulting to max_seq_len=8")
        return 8

    if device == "cpu" or not torch.cuda.is_available():
        print("[auto] CPU mode — defaulting to max_seq_len=8")
        return 8

    gpu_idx = int(device.split(":")[1]) if ":" in device else 0
    props = torch.cuda.get_device_properties(gpu_idx)
    total_vram = props.total_memory          # bytes
    used_vram  = torch.cuda.memory_allocated(gpu_idx)
    free_vram  = total_vram - used_vram

    headroom   = 1.5 * 1024 ** 3            # reserve 1.5 GB
    usable     = max(0.0, free_vram - headroom)

    # Empirical: each frame's VRAM cost during RealBasicVSR forward pass.
    # Feature maps are ~60× the raw pixel bytes (mid_channels=64, 20 prop blocks).
    bytes_per_frame = width * height * 3 * 4 * 60
    if bytes_per_frame == 0:
        return 30

    estimated = max(1, min(120, int(usable / bytes_per_frame)))
    total_gb   = total_vram / 1024 ** 3
    free_gb    = free_vram  / 1024 ** 3
    print(
        f"[auto] VRAM: {total_gb:.1f} GB total, {free_gb:.1f} GB free → "
        f"max_seq_len={estimated}"
    )
    return estimated


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


def _build_inference_env(device: str) -> dict:
    """Build the subprocess env with PYTHONPATH and CUDA_VISIBLE_DEVICES set."""
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{VENDOR_RB_DIR}{os.pathsep}{existing_pp}" if existing_pp
        else str(VENDOR_RB_DIR)
    )
    if device.startswith("cuda:"):
        gpu_idx = device.split(":")[1]
        env["CUDA_VISIBLE_DEVICES"] = gpu_idx
        print(f"[inference] Using GPU {gpu_idx} (CUDA_VISIBLE_DEVICES={gpu_idx})")
    elif device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        print("[inference] GPU disabled — running on CPU (slow)")
    return env


def _run_inference_chunk(
    chunk_frames_dir: Path,
    chunk_out_dir: Path,
    config: str,
    checkpoint: str,
    env: dict,
) -> None:
    """Run the RealBasicVSR inference script on one chunk directory."""
    chunk_out_dir.mkdir(parents=True, exist_ok=True)
    run([
        "python", str(INFERENCE_SCRIPT),
        config,
        checkpoint,
        str(chunk_frames_dir),
        str(chunk_out_dir),
        "--is_save_as_png=True",
    ], env=env)


def run_inference(
    frames_dir: Path,
    upscaled_dir: Path,
    config: str,
    checkpoint: str,
    max_seq_len: int,
    device: str = "cuda:0",
    chunk_overlap: int = 0,
) -> None:
    """Call the RealBasicVSR inference script on the extracted frames.

    When max_seq_len < total frames the video is processed in chunks.
    chunk_overlap adds extra frames at each chunk boundary so the model
    has temporal context across seams; the overlap frames are discarded
    from the output so the final frame count stays correct.

    Args:
        frames_dir:    Directory of extracted source PNG frames.
        upscaled_dir:  Directory to write upscaled PNG frames to.
        config:        Path to the RealBasicVSR config file.
        checkpoint:    Path to the model checkpoint.
        max_seq_len:   Max frames per forward pass. Lower = less VRAM used.
        device:        "cuda:0" / "cuda:1" / "cpu".
        chunk_overlap: Extra frames borrowed from adjacent chunks at each
                       boundary to preserve temporal context.
                       Recommended: 2–5. 0 disables overlap (original behaviour).
    """
    upscaled_dir.mkdir(parents=True, exist_ok=True)
    env = _build_inference_env(device)

    all_frames = sorted(frames_dir.glob(FRAME_GLOB))
    total = len(all_frames)

    if total == 0:
        raise RuntimeError(f"[inference] No frames found in {frames_dir}")

    # If everything fits in one pass, skip chunking entirely
    if total <= max_seq_len:
        print(f"[inference] Single pass ({total} frames ≤ max_seq_len={max_seq_len})")
        _run_inference_chunk(frames_dir, upscaled_dir, config, checkpoint, env)
        count = len(list(upscaled_dir.glob(FRAME_GLOB)))
        print(f"[inference] {count} upscaled frames → {upscaled_dir}")
        return

    # ── Chunked inference with optional overlap ────────────────────────────────
    chunk_starts = list(range(0, total, max_seq_len))
    print(
        f"[inference] {total} frames → {len(chunk_starts)} chunks "
        f"(max_seq_len={max_seq_len}, overlap={chunk_overlap})"
    )

    tmp_chunks_root = upscaled_dir.parent / "chunk_tmp"
    tmp_chunks_root.mkdir(parents=True, exist_ok=True)

    output_idx = 0  # global counter for output filenames

    try:
        for chunk_n, start in enumerate(chunk_starts):
            end = min(start + max_seq_len, total)

            # Expand window with overlap (clamped to valid frame range)
            lo = max(0, start - chunk_overlap)
            hi = min(total, end + chunk_overlap)
            window_frames = all_frames[lo:hi]

            chunk_in_dir  = tmp_chunks_root / f"chunk_{chunk_n:04d}_in"
            chunk_out_dir = tmp_chunks_root / f"chunk_{chunk_n:04d}_out"
            chunk_in_dir.mkdir(parents=True, exist_ok=True)

            # Symlink (or copy) the window frames into a temporary input dir
            for seq_i, src_frame in enumerate(window_frames):
                dst = chunk_in_dir / f"{seq_i:08d}.png"
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src_frame.resolve())

            print(
                f"[inference] Chunk {chunk_n + 1}/{len(chunk_starts)}: "
                f"frames {lo}–{hi - 1} (window={len(window_frames)})"
            )
            _run_inference_chunk(chunk_in_dir, chunk_out_dir, config, checkpoint, env)

            # Collect only the non-overlap output frames that belong to this chunk
            out_frames = sorted(chunk_out_dir.glob(FRAME_GLOB))
            # How many overlap frames were prepended / appended?
            pre_skip  = start - lo   # frames to skip at the start (left overlap)
            post_skip = hi - end     # frames to skip at the end  (right overlap)
            keep_frames_list = out_frames[
                pre_skip : len(out_frames) - post_skip if post_skip else None
            ]

            for frame in keep_frames_list:
                dst = upscaled_dir / f"{output_idx:08d}.png"
                shutil.copy2(frame, dst)
                output_idx += 1

    finally:
        shutil.rmtree(tmp_chunks_root, ignore_errors=True)

    print(f"[inference] {output_idx} upscaled frames → {upscaled_dir}")


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
        default=_AUTO,
        dest="max_seq_len",
        help=(
            "Max frames per inference batch. "
            "'auto' (default) detects a safe value from available VRAM. "
            "Pass an integer (e.g. 8) to override on low-VRAM GPUs."
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=3,
        dest="chunk_overlap",
        help=(
            "Extra frames borrowed from neighbouring chunks at each boundary "
            "to preserve temporal context across seams. "
            "Recommended: 2-5. Set 0 to disable (default: 3)."
        ),
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
    parser.add_argument(
        "--device",
        default="cuda:0",
        help=(
            "Device for inference: 'cuda:0' (default), 'cuda:1' (second GPU), "
            "or 'cpu' (no GPU)."
        ),
    )
    return parser.parse_args()


def upscale_video(
    input_path: str | Path,
    output_path: str | Path,
    config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    max_seq_len: int | str = _AUTO,
    scale: int = DEFAULT_SCALE,
    keep_frames: bool = False,
    work_dir: str | Path | None = None,
    device: str = "cuda:0",
    chunk_overlap: int = 3,
) -> Path:
    """Upscale a single video using RealBasicVSR + ffmpeg.

    Args:
        input_path:    Path to the source video file.
        output_path:   Path for the upscaled output video.
        config:        RealBasicVSR config (.py). Defaults to the vendor x4 config.
        checkpoint:    Model checkpoint (.pth). Defaults to checkpoints/RealBasicVSR_x4.pth.
        max_seq_len:   Max frames per forward pass — primary VRAM control lever.
                         "auto"  — detect safe value from available VRAM (default)
                         int     — explicit frame count (e.g. 8 for 8 GB GPU)
        scale:         Expected upscale factor — used only for the verification summary.
        keep_frames:   If True, extracted and upscaled frame dirs are not deleted.
        work_dir:      Custom directory for temp frames. Defaults to a system temp dir.
        device:        Which device to run inference on.
                         "cuda:0"  — first GPU (default)
                         "cuda:1"  — second GPU (multi-GPU machines)
                         "cpu"     — disable GPU entirely (slow, always works)
        chunk_overlap: Extra frames borrowed from neighbouring chunks at each boundary
                       so the model has temporal context across seams. The overlap
                       frames are discarded from the output (frame count unchanged).
                       Recommended: 2–5. Set to 0 to disable (original behaviour).

    Returns:
        Path to the finished output video.

    Raises:
        RuntimeError: If preflight validation fails.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    config = str(config) if config else str(DEFAULT_CONFIG)
    checkpoint = str(checkpoint) if checkpoint else str(DEFAULT_CHECKPOINT)

    preflight(input_path, output_path, config, checkpoint)

    meta = probe_video(input_path)

    # Resolve "auto" → concrete int now that we know the frame resolution
    if max_seq_len == _AUTO:
        resolved_seq_len = auto_max_seq_len(meta["width"], meta["height"], device)
    else:
        resolved_seq_len = int(max_seq_len)

    print("=" * 60)
    print("  RealBasicVSR + FFmpeg Upscaling Pipeline")
    print("=" * 60)
    print(f"  Input        : {input_path}")
    print(f"  Output       : {output_path}")
    print(f"  Config       : {config}")
    print(f"  Checkpoint   : {checkpoint}")
    print(f"  MaxSeqLen    : {resolved_seq_len}"
          + (" (auto)" if max_seq_len == _AUTO else ""))
    print(f"  ChunkOverlap : {chunk_overlap}")
    print(f"  Device       : {device}")
    print("=" * 60)

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
            max_seq_len=resolved_seq_len,
            device=device,
            chunk_overlap=chunk_overlap,
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
        # CLI passes max_seq_len as a string; convert to int unless it's "auto"
        msl = args.max_seq_len
        if msl != _AUTO:
            msl = int(msl)
        upscale_video(
            input_path=args.input,
            output_path=args.output,
            config=args.config,
            checkpoint=args.checkpoint,
            max_seq_len=msl,
            scale=args.scale,
            keep_frames=args.keep_frames,
            work_dir=args.work_dir,
            device=args.device,
            chunk_overlap=args.chunk_overlap,
        )
    except RuntimeError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
