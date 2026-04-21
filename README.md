# video-upscaling

Single-video upscaling pipeline powered by [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) (CVPR 2022) and ffmpeg.

## Pipeline overview

```
Input video
    │
    ▼
ffmpeg — extract PNG frames
    │
    ▼
RealBasicVSR — 4× super-resolution inference
    │
    ▼
ffmpeg — encode upscaled frames to H.264
    │
    ▼
ffmpeg — mux original audio
    │
    ▼
Output video (4× resolution, original fps & audio)
```

## Prerequisites

| Tool | Minimum version | Install hint |
|------|----------------|--------------|
| Python | 3.8 | `conda` or system package |
| CUDA | 10.1+ (recommended) | [NVIDIA docs](https://developer.nvidia.com/cuda-downloads) |
| ffmpeg | 4.x | `sudo apt install ffmpeg` |
| git | any | `sudo apt install git` |

## Setup


### 0. Environment
```bash
conda create -n enhancevideo python=3.8
conda activate enhancevideo 
conda install -c conda-forge ffmpeg
```
### 1. Clone this repository

```bash
git clone <this-repo-url>
cd video-upscaling
```

### 2. Run the bootstrap script



This script clones the official RealBasicVSR repo into `vendor/`, installs all Python dependencies, and tells you where to save the model checkpoint.

```bash
bash scripts/setup_realbasicvsr.sh
```

> If you use a conda environment, activate it **before** running the script so that `pip` and `mim` install into the right env.

### 3. Download the pretrained checkpoint

Run the download script — it tries Google Drive first via `gdown`, then falls back to Dropbox:

```bash
bash scripts/download_weights.sh
```

Or download manually from any of these mirrors and save to `checkpoints/RealBasicVSR_x4.pth`:

- [Dropbox](https://www.dropbox.com/s/eufigwhejgd8bnt/RealBasicVSR_x4.pth)
- [Google Drive](https://drive.google.com/file/d/1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID/view)
- [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/chan0899_e_ntu_edu_sg/EVlhWlqBVuxOhjeSqLN9N4UBSbQ5Z-PvYFm2AO5kAGxJSg)

```bash
# Verify the file is in place:
ls -lh checkpoints/RealBasicVSR_x4.pth
```

## Usage

Run the pipeline from the repository root:

```bash
python src/pipeline/upscale_video.py INPUT_VIDEO OUTPUT_VIDEO [OPTIONS]
```

### Minimal example

```bash
python src/pipeline/upscale_video.py input.mp4 output_4x.mp4
```

### All options

```
positional arguments:
  input                 Path to the source video file.
  output                Path for the upscaled output video.

optional arguments:
  --config CONFIG       RealBasicVSR config file (.py).
                        Default: vendor/RealBasicVSR/configs/realbasicvsr_x4.py
  --checkpoint CKPT     RealBasicVSR checkpoint file (.pth).
                        Default: checkpoints/RealBasicVSR_x4.pth
  --max-seq-len N       Max frames per inference batch.
                        Reduce to 8–15 on GPUs with less than 16 GB VRAM.
                        Default: 30
  --scale N             Expected upscale factor (used for output verification).
                        Default: 4
  --keep-frames         Keep extracted and upscaled frame directories.
  --work-dir DIR        Custom directory for temp frames (default: system temp).
```

### Low-VRAM GPU (e.g. 8 GB)

```bash
python src/pipeline/upscale_video.py input.mp4 output_4x.mp4 --max-seq-len 8
```

### Keep intermediate frames for inspection

```bash
python src/pipeline/upscale_video.py input.mp4 output_4x.mp4 \
  --keep-frames --work-dir ./workspace/my_video
```

## Output verification

The pipeline prints a verification table at the end of each run:

```
── Verification ───────────────────────────────────────────
  Source  : 640×360   25.000 fps  duration=12.04s
  Output  : 2560×1440 25.000 fps  duration=12.04s
  Scale   : ✓  4x upscale confirmed (2560×1440)
───────────────────────────────────────────────────────────
```

You can also verify manually with ffprobe:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate \
  -of default=noprint_wrappers=1 output_4x.mp4
```

## Project structure

```
video-upscaling/
├── checkpoints/                # Place RealBasicVSR_x4.pth here
├── scripts/
│   └── setup_realbasicvsr.sh  # One-time environment bootstrap
├── src/
│   └── pipeline/
│       ├── config.py           # Centralized defaults (paths, codec, etc.)
│       └── upscale_video.py    # Main CLI orchestration script
├── vendor/
│   └── RealBasicVSR/           # Cloned by setup script
├── requirements.txt
└── README.md
```

## Citation

If you use this pipeline in your work, please also cite the original RealBasicVSR paper:

```bibtex
@inproceedings{chan2022investigating,
  author    = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  title     = {Investigating Tradeoffs in Real-World Video Super-Resolution},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2022}
}
```
