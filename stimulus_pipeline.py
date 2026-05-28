"""Prepare multi-camera skating videos for MEG stimulus presentation.

The pipeline uses a fixed input-folder structure and can run each editing stage
independently so intermediate videos can be inspected before continuing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
from pathlib import Path


STAGES = ["cut_big", "cut_small", "clean_bg", "masks", "overlays", "camera_cut"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
DEFAULT_COLOR_MAP = {
    (255, 0, 0): ((255, 0, 0), 0.2),
    (0, 0, 255): ((0, 0, 255), 0.2),
    (100, 0, 0): ((255, 0, 0), 0.4),
    (0, 100, 0): ((0, 255, 0), 0.4),
    (0, 0, 100): ((0, 0, 255), 0.4),
    (0, 0, 0): ((0, 255, 255), 0.4),
}
DEFAULT_INPUT_DIR = Path("Vids/input")
DEFAULT_OUTPUT_DIR = Path("Vids/output")
DEFAULT_CONFIG_PATH = Path("pipeline_config.json")
CONFIG_TEMPLATE = {
    "data_root": "/path/to/mounted/secure/server/skater-data",
    "input_dir": "input",
    "output_dir": "output",
}


def resolve_data_path(path: Path, data_root: Path | None) -> Path:
    """Resolve a configured data path, optionally relative to a shared data root."""
    return path if path.is_absolute() or data_root is None else data_root / path


def load_path_config(path: Path | None) -> dict[str, str]:
    """Read optional local path configuration from JSON."""
    if path is None:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    allowed_keys = {"data_root", "input_dir", "output_dir"}
    unknown_keys = sorted(set(config) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown config key(s) in {path}: {', '.join(unknown_keys)}. "
            f"Allowed keys: {', '.join(sorted(allowed_keys))}."
        )
    return {key: str(value) for key, value in config.items() if value not in (None, "")}


def write_config_template(path: Path) -> None:
    """Create a local path configuration template."""
    if path.exists():
        raise FileExistsError(f"Config file already exists: {path}")
    path.write_text(json.dumps(CONFIG_TEMPLATE, indent=2) + "\n", encoding="utf-8")


def configure_data_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Apply defaults, config file values, environment overrides, and CLI paths."""
    config_path = args.config
    if config_path is None:
        env_config = os.environ.get("SKATER_PIPELINE_CONFIG")
        config_path = Path(env_config) if env_config else DEFAULT_CONFIG_PATH

    if args.write_config:
        write_config_template(args.write_config)
        print(f"Wrote config template to: {args.write_config}")
        raise SystemExit(0)

    config = load_path_config(config_path)
    data_root_value = (
        args.data_root
        or os.environ.get("SKATER_DATA_ROOT")
        or config.get("data_root")
    )
    data_root = Path(data_root_value).expanduser() if data_root_value else None

    input_value = (
        args.input_dir
        or os.environ.get("SKATER_INPUT_DIR")
        or config.get("input_dir")
        or DEFAULT_INPUT_DIR
    )
    output_value = (
        args.output_dir
        or os.environ.get("SKATER_OUTPUT_DIR")
        or config.get("output_dir")
        or DEFAULT_OUTPUT_DIR
    )

    args.config = config_path
    args.data_root = data_root
    args.input_dir = resolve_data_path(Path(input_value).expanduser(), data_root)
    args.output_dir = resolve_data_path(Path(output_value).expanduser(), data_root)

    if args.input_dir == args.output_dir:
        parser.error("--input-dir and --output-dir must be different paths")


def show_configured_paths(args: argparse.Namespace) -> None:
    """Print the effective data paths after config/env/CLI resolution."""
    print(f"Config file: {args.config}")
    print(f"Data root: {args.data_root or '(none)'}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")


def require_tool(name: str) -> None:
    """Fail early when a required command-line video tool is unavailable."""
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command-line tool not found: {name}")


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    """Run a command or print it when previewing a video-processing stage."""
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def timecode_to_seconds(timecode: str, fps: int) -> float:
    """Convert HH:MM:SS or HH:MM:SS:FF timecodes to seconds."""
    parts = [int(part) for part in timecode.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        frame = 0
    elif len(parts) == 4:
        h, m, s, frame = parts
    else:
        raise ValueError(f"Timecode must be HH:MM:SS or HH:MM:SS:FF: {timecode}")
    return h * 3600 + m * 60 + s + frame / fps


def best_small_chunk_duration(durations: list[float], lower: float, upper: float) -> float:
    """Choose a duration that leaves the least unused footage across chunks."""
    best_duration = lower
    best_remainder = math.inf
    step = 0.25
    n_steps = int((upper - lower) / step) + 1
    for i in range(n_steps):
        candidate = lower + i * step
        remainder = sum(duration % candidate for duration in durations)
        if remainder < best_remainder:
            best_remainder = remainder
            best_duration = candidate
    return best_duration


def camera_dirs(input_dir: Path) -> list[Path]:
    """Return camera folders in stable numeric/name order."""
    cameras_root = input_dir / "cameras"
    if not cameras_root.exists():
        raise FileNotFoundError(f"Missing camera folder: {cameras_root}")
    dirs = [p for p in cameras_root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No camera folders found in: {cameras_root}")
    return sorted(dirs, key=lambda p: p.name)


def raw_video_path(camera_dir: Path) -> Path:
    """Find the raw recording for one camera folder."""
    preferred = camera_dir / "raw.mp4"
    if preferred.exists():
        return preferred
    videos = sorted(p for p in camera_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if len(videos) != 1:
        raise FileNotFoundError(
            f"Expected raw.mp4 or exactly one video file in {camera_dir}; found {len(videos)}."
        )
    return videos[0]


def read_first_frame(camera_dir: Path) -> int:
    """Read the synchronization first-frame value for one camera."""
    path = camera_dir / "first_frame.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing synchronization file: {path}")
    return int(path.read_text(encoding="utf-8").strip())


def read_timing(input_dir: Path) -> list[dict[str, str]]:
    """Read chunk start/end timecodes from input_dir/timing.csv."""
    path = input_dir / "timing.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing timing file: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"chunk", "start", "end"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError("timing.csv must contain columns: chunk,start,end")
    return rows


def synced_chunk_plan(
    input_dir: Path,
    fps: int,
    lead_in_seconds: float,
) -> list[list[dict[str, object]]]:
    """Calculate synchronized cut times for every camera and big chunk."""
    cameras = camera_dirs(input_dir)
    first_frames = [read_first_frame(camera_dir) for camera_dir in cameras]
    timing_rows = read_timing(input_dir)
    ref_first_frame = first_frames[0]

    plan = []
    for camera_index, (camera_dir, first_frame) in enumerate(
        zip(cameras, first_frames), start=1
    ):
        offset = (first_frame - ref_first_frame) / fps
        raw_video = raw_video_path(camera_dir)
        camera_plan = []
        for row in timing_rows:
            start = timecode_to_seconds(row["start"], fps) + lead_in_seconds + offset
            end = timecode_to_seconds(row["end"], fps) + offset
            camera_plan.append(
                {
                    "camera": camera_index,
                    "camera_name": camera_dir.name,
                    "chunk": int(row["chunk"]),
                    "video_file": raw_video,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                }
            )
        plan.append(camera_plan)
    return plan


def get_video_info(video_path: Path) -> tuple[int, int, float]:
    """Return video width, height, and frame rate using FFprobe."""
    require_tool("ffprobe")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)["streams"][0]
    width = int(info["width"])
    height = int(info["height"])
    num, den = map(int, info["r_frame_rate"].split("/"))
    return width, height, num / den


def get_video_duration(path: Path) -> float:
    """Return video duration in seconds using FFprobe."""
    require_tool("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd)
    return float(json.loads(out)["streams"][0]["duration"])


def camera_output_root(args: argparse.Namespace, camera_name: str) -> Path:
    """Return the root output folder for one camera."""
    return args.output_dir / camera_name


def big_chunk_dir(args: argparse.Namespace, camera_name: str) -> Path:
    """Return the folder for synchronized big chunks from one camera."""
    return camera_output_root(args, camera_name) / "big"


def small_chunk_dir(args: argparse.Namespace, camera_name: str) -> Path:
    """Return the folder for small chunks from one camera."""
    return camera_output_root(args, camera_name) / "small"


def cut_big_chunks(args: argparse.Namespace) -> None:
    """Cut long raw camera recordings into synchronized big chunks."""
    require_tool("ffmpeg")
    for camera_plan in synced_chunk_plan(args.input_dir, args.fps, args.lead_in_seconds):
        for item in camera_plan:
            camera_dir = big_chunk_dir(args, str(item["camera_name"]))
            output_file = camera_dir / f"big{item['chunk']}.mp4"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(item["start"]),
                "-to",
                str(item["end"]),
                "-i",
                str(item["video_file"]),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output_file),
            ]
            run_command(cmd, dry_run=args.dry_run)


def cut_small_chunks(args: argparse.Namespace) -> None:
    """Split big chunks into smaller clips for inspection and later editing."""
    require_tool("ffmpeg")
    big_chunks = sorted(args.output_dir.glob("*/big/big*.mp4"))
    if not big_chunks:
        raise FileNotFoundError(f"No big chunks found under: {args.output_dir}")

    if args.small_duration is None:
        durations = [get_video_duration(path) for path in big_chunks]
        small_duration = best_small_chunk_duration(
            durations, args.small_duration_min, args.small_duration_max
        )
    else:
        small_duration = args.small_duration
    print(f"Small chunk duration: {small_duration:.2f}s")

    for big_chunk_file in big_chunks:
        camera_name = big_chunk_file.parent.parent.name
        output_dir = small_chunk_dir(args, camera_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        duration = get_video_duration(big_chunk_file)
        num_chunks = int(math.floor(duration / small_duration))
        for index in range(num_chunks):
            start = index * small_duration
            end = start + small_duration
            output_file = output_dir / f"{big_chunk_file.stem}_small{index + 1}.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-to",
                str(end),
                "-i",
                str(big_chunk_file),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output_file),
            ]
            run_command(cmd, dry_run=args.dry_run)


def clean_background(args: argparse.Namespace) -> None:
    """Replace masked background pixels with a clean reference frame."""
    import numpy as np
    from PIL import Image

    require_tool("ffmpeg")
    require_tool("ffprobe")

    for camera_input_dir in camera_dirs(args.input_dir):
        clean_frame_path = camera_input_dir / "empty_frame.png"
        background_mask_path = camera_input_dir / "background_mask.png"
        if not clean_frame_path.exists() or not background_mask_path.exists():
            print(f"Skipping {camera_input_dir.name}: no clean_frame.png/background_mask.png")
            continue

        camera_output_dir = camera_output_root(args, camera_input_dir.name)
        small_dir = small_chunk_dir(args, camera_input_dir.name)
        cleaned_dir = camera_output_dir / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        clean = np.asarray(Image.open(clean_frame_path).convert("RGB"), dtype=np.float32)
        mask = np.asarray(Image.open(background_mask_path).convert("L"), dtype=np.float32) / 255.0
        mask = mask[..., None]

        for video_path in sorted(small_dir.glob(args.chunk_pattern)):
            output_path = cleaned_dir / f"{video_path.stem}_cleaned.mp4"
            width, height, fps = get_video_info(video_path)
            if clean.shape[:2] != (height, width):
                raise ValueError(f"Clean image size does not match video: {video_path}")
            if mask.shape[:2] != (height, width):
                raise ValueError(f"Background mask size does not match video: {video_path}")

            print(f"Cleaning background: {video_path} -> {output_path}")
            in_pipe = subprocess.Popen(
                [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ],
                stdout=subprocess.PIPE,
            )
            out_pipe = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(args.clean_crf),
                    "-preset",
                    args.clean_preset,
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                stdin=subprocess.PIPE,
            )

            frame_size = width * height * 3
            frame_count = 0
            if in_pipe.stdout is None or out_pipe.stdin is None:
                raise RuntimeError("Could not open ffmpeg pipes.")

            while True:
                raw = in_pipe.stdout.read(frame_size)
                if len(raw) != frame_size:
                    break
                frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
                frame = frame.astype(np.float32)
                result = frame * mask + clean * (1 - mask)
                out_pipe.stdin.write(np.clip(result, 0, 255).astype(np.uint8).tobytes())
                frame_count += 1

            in_pipe.stdout.close()
            out_pipe.stdin.close()
            in_pipe.wait()
            out_pipe.wait()
            print(f"Frames processed: {frame_count}")


def mask_source_videos(args: argparse.Namespace, camera_name: str) -> list[Path]:
    """Return videos to use as inputs for mask generation."""
    camera_output_dir = camera_output_root(args, camera_name)
    cleaned_dir = camera_output_dir / "cleaned"
    videos = []
    for small_video in sorted(small_chunk_dir(args, camera_name).glob(args.chunk_pattern)):
        cleaned_video = cleaned_dir / f"{small_video.stem}_cleaned.mp4"
        videos.append(cleaned_video if cleaned_video.exists() else small_video)
    return videos


def generate_frame_masks(args: argparse.Namespace) -> None:
    """Create one mask image per video frame using reference-frame differencing."""
    import cv2
    import numpy as np

    def align_image_to_frame(
        image: np.ndarray,
        frame: np.ndarray,
        interpolation: int,
    ) -> np.ndarray:
        """Resize an image to match the current video frame when needed."""
        frame_h, frame_w = frame.shape[:2]
        if image.shape[:2] == (frame_h, frame_w):
            return image
        return cv2.resize(image, (frame_w, frame_h), interpolation=interpolation)

    processed_videos = 0
    for camera_input_dir in camera_dirs(args.input_dir):
        empty_frame_path = camera_input_dir / "empty_frame.png"
        overlay_mask_path = camera_input_dir / "overlay_mask.png"
        if not empty_frame_path.exists() or not overlay_mask_path.exists():
            print(f"Skipping {camera_input_dir.name}: no empty_frame.png/overlay_mask.png")
            continue

        ref = cv2.imread(str(empty_frame_path), cv2.IMREAD_COLOR)
        background_mask = cv2.imread(str(overlay_mask_path), cv2.IMREAD_UNCHANGED)
        if ref is None:
            raise RuntimeError(f"Cannot read empty frame: {empty_frame_path}")
        if background_mask is None:
            raise RuntimeError(f"Cannot read overlay mask: {overlay_mask_path}")
        if background_mask.ndim == 2:
            background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGRA)
        elif background_mask.shape[2] == 3:
            background_mask = cv2.cvtColor(background_mask, cv2.COLOR_BGR2BGRA)

        source_videos = mask_source_videos(args, camera_input_dir.name)
        if not source_videos:
            print(
                f"Skipping {camera_input_dir.name}: no videos matching "
                f"{args.chunk_pattern!r} in {small_chunk_dir(args, camera_input_dir.name)}"
            )
            continue

        for video_path in source_videos:
            output_dir = args.output_dir / "masks" / camera_input_dir.name / video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

            video_ref = None
            video_background_mask = None
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if video_ref is None:
                    video_ref = align_image_to_frame(ref, frame, cv2.INTER_LINEAR)
                    video_background_mask = align_image_to_frame(
                        background_mask, frame, cv2.INTER_NEAREST
                    )

                diff = cv2.absdiff(frame.astype("float32"), video_ref.astype("float32"))
                diff = cv2.GaussianBlur(diff, tuple(args.blur_kernel), 0)
                diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                _, skater_mask = cv2.threshold(
                    diff_gray, args.mask_threshold, 255, cv2.THRESH_BINARY
                )
                kernel = np.ones(tuple(args.close_kernel), np.uint8)
                skater_mask = cv2.morphologyEx(skater_mask, cv2.MORPH_CLOSE, kernel)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    skater_mask, connectivity=8
                )

                if num_labels > 1:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    largest_blob = np.zeros_like(skater_mask)
                    largest_blob[labels == largest_label] = 255
                    skater_mask = largest_blob

                final_mask = video_background_mask.copy()
                final_mask[skater_mask == 255] = [255, 255, 255, 255]
                cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}.png"), final_mask)
                frame_idx += 1

            cap.release()
            print(f"Saved {frame_idx} masks to: {output_dir}")
            processed_videos += 1

    if processed_videos == 0:
        raise FileNotFoundError(
            "No mask inputs were processed. Check that --input-dir contains camera "
            "empty_frame.png/overlay_mask.png files and that --output-dir contains "
            f"small chunks under camera*/small/ matching {args.chunk_pattern!r}."
        )


def read_color_map(input_dir: Path) -> dict[tuple[int, int, int], tuple[tuple[int, int, int], float]]:
    """Load input_dir/color_map.csv when present, otherwise use defaults."""
    path = input_dir / "color_map.csv"
    if not path.exists():
        return DEFAULT_COLOR_MAP

    color_map = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mask = (int(row["mask_b"]), int(row["mask_g"]), int(row["mask_r"]))
            overlay = (
                int(row["overlay_b"]),
                int(row["overlay_g"]),
                int(row["overlay_r"]),
            )
            color_map[mask] = (overlay, float(row["alpha"]))
    return color_map


def source_video_for_masks(args: argparse.Namespace, camera_name: str, stem: str) -> Path:
    """Find the video that corresponds to one mask-frame folder."""
    candidates = [
        args.output_dir / camera_name / "cleaned" / f"{stem}.mp4",
        small_chunk_dir(args, camera_name) / f"{stem}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No source video found for mask folder: {camera_name}/{stem}")


def overlay_colours(args: argparse.Namespace) -> None:
    """Render colored landmark overlays from per-frame mask images."""
    import cv2
    import numpy as np

    require_tool("ffmpeg")
    color_map = read_color_map(args.input_dir)
    colors = np.array(list(color_map.keys()), dtype=np.int16)
    overlays = np.array([value[0] for value in color_map.values()], dtype=np.float32)
    alphas = np.array([value[1] for value in color_map.values()], dtype=np.float32)

    masks_root = args.output_dir / "masks"
    if not masks_root.exists():
        raise FileNotFoundError(f"No masks folder found: {masks_root}")

    for camera_mask_dir in sorted(p for p in masks_root.iterdir() if p.is_dir()):
        camera_name = camera_mask_dir.name
        output_dir = args.output_dir / camera_name / "overlaid"
        output_dir.mkdir(parents=True, exist_ok=True)

        for mask_dir in sorted(p for p in camera_mask_dir.iterdir() if p.is_dir()):
            video_path = source_video_for_masks(args, camera_name, mask_dir.name)
            output_path = output_dir / f"{mask_dir.name}_overlay.mp4"
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Cannot read video: {video_path}")

            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
            mask_files = sorted(
                f
                for f in mask_dir.iterdir()
                if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            if not mask_files:
                raise RuntimeError(f"No mask images found in: {mask_dir}")

            masks = []
            for file_path in mask_files:
                mask = cv2.imread(str(file_path))
                if mask is None:
                    raise RuntimeError(f"Cannot load mask: {file_path}")
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h))
                masks.append(mask)

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(int(fps)),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]

            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= len(masks):
                    break

                frame_f32 = frame.astype(np.float32)
                mask_i16 = masks[frame_idx].astype(np.int16)
                diff = np.abs(mask_i16[:, :, None, :] - colors[None, None, :, :])
                matches = np.all(diff < args.color_tolerance, axis=-1)
                idx = np.argmax(matches, axis=-1)
                overlay_map = overlays[idx]
                alpha_map = alphas[idx][..., None]
                valid = np.any(matches, axis=-1)[..., None]
                result = np.where(
                    valid,
                    frame_f32 * (1 - alpha_map) + overlay_map * alpha_map,
                    frame_f32,
                )
                proc.stdin.write(np.clip(result, 0, 255).astype(np.uint8).tobytes())
                frame_idx += 1

            if proc.stdin:
                proc.stdin.close()
            proc.wait()
            cap.release()
            print(f"Saved overlay video to: {output_path}")


def camera_switch_source(args: argparse.Namespace, camera_name: str) -> Path:
    """Select the best available source clip for final viewpoint switching."""
    base = f"big{args.big_index}_small{args.small_index}"
    candidates = [
        args.output_dir / camera_name / "overlaid" / f"{base}_cleaned_overlay.mp4",
        args.output_dir / camera_name / "overlaid" / f"{base}_overlay.mp4",
        args.output_dir / camera_name / "cleaned" / f"{base}_cleaned.mp4",
        small_chunk_dir(args, camera_name) / f"{base}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No switch source found for {camera_name}, {base}")


def create_camera_switch_video(args: argparse.Namespace) -> None:
    """Create a final stimulus video and frame-level camera log."""
    require_tool("ffmpeg")
    files = [
        camera_switch_source(args, camera_dir.name)
        for camera_dir in camera_dirs(args.input_dir)
    ]
    output_dir = args.output_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / (
        f"big{args.big_index}_small{args.small_index}_switch_{args.cut_frames}f.mp4"
    )
    output_log = output_video.with_name(f"{output_video.stem}_camlog.csv")

    random.seed(args.seed)
    duration = get_video_duration(files[0])
    total_frames = int(duration * args.fps)
    cameras = list(range(len(files)))
    current_camera = random.choice(cameras)
    segments = []

    frame = 0
    while frame < total_frames:
        end = min(frame + args.cut_frames, total_frames)
        segments.append((frame, end, current_camera))
        current_camera = random.choice([c for c in cameras if c != current_camera])
        frame = end

    with output_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "camera"])
        for start, end, camera in segments:
            for frame_idx in range(start, end):
                writer.writerow([frame_idx + 1, camera + 1])

    filters = []
    concat_inputs = []
    for i, (start, end, camera) in enumerate(segments):
        t_start = start / args.fps
        t_end = end / args.fps
        label = f"v{i}"
        filters.append(
            f"[{camera}:v]trim=start={t_start}:end={t_end},"
            f"setpts=PTS-STARTPTS[{label}]"
        )
        concat_inputs.append(f"[{label}]")

    filter_complex = (
        ";\n".join(filters)
        + ";\n"
        + "".join(concat_inputs)
        + f"concat=n={len(segments)}:v=1:a=0[outv]"
    )

    cmd = ["ffmpeg", "-y"]
    for file_path in files:
        cmd.extend(["-i", str(file_path)])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-r",
            str(args.fps),
            str(output_video),
        ]
    )
    run_command(cmd, dry_run=args.dry_run)
    print(f"Saved camera log to: {output_log}")


def run_stage(stage: str, args: argparse.Namespace) -> None:
    """Dispatch one named pipeline stage."""
    if stage == "cut_big":
        cut_big_chunks(args)
    elif stage == "cut_small":
        cut_small_chunks(args)
    elif stage == "clean_bg":
        clean_background(args)
    elif stage == "masks":
        generate_frame_masks(args)
    elif stage == "overlays":
        overlay_colours(args)
    elif stage == "camera_cut":
        create_camera_switch_video(args)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def parse_kernel(value: str) -> list[int]:
    """Parse kernel sizes written as WIDTH,HEIGHT."""
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Kernel values must be WIDTH,HEIGHT")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MEG video stimuli.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional JSON config with data_root, input_dir, and output_dir. "
            "Defaults to SKATER_PIPELINE_CONFIG or pipeline_config.json when present."
        ),
    )
    parser.add_argument(
        "--write-config",
        type=Path,
        default=None,
        help="Write a starter path config JSON file and exit.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Base folder for relative input/output paths, such as a mounted server folder.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input folder. Relative paths are resolved under --data-root when set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Relative paths are resolved under --data-root when set.",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print the effective config/input/output paths and exit.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STAGES + ["all", "preprocess"],
        default=None,
        help="Stages to run. Use preprocess for cut_big followed by cut_small.",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--lead-in-seconds", type=float, default=4)
    parser.add_argument("--small-duration", type=float, default=None)
    parser.add_argument("--small-duration-min", type=float, default=120)
    parser.add_argument("--small-duration-max", type=float, default=180)
    parser.add_argument("--chunk-pattern", default="big*_small*.mp4")
    parser.add_argument("--clean-crf", type=int, default=0)
    parser.add_argument("--clean-preset", default="medium")
    parser.add_argument("--mask-threshold", type=int, default=50)
    parser.add_argument("--blur-kernel", type=parse_kernel, default=[21, 21])
    parser.add_argument("--close-kernel", type=parse_kernel, default=[3, 3])
    parser.add_argument("--color-tolerance", type=int, default=10)
    parser.add_argument("--big-index", type=int, default=1)
    parser.add_argument("--small-index", type=int, default=2)
    parser.add_argument("--cut-frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_data_paths(args, parser)
    if args.show_paths:
        show_configured_paths(args)
        raise SystemExit(0)

    steps = args.steps or STAGES
    if "all" in steps:
        steps = STAGES
    if "preprocess" in steps:
        steps = [step for step in steps if step != "preprocess"]
        steps = ["cut_big", "cut_small", *steps]

    for step in steps:
        run_stage(step, args)


if __name__ == "__main__":
    main()
