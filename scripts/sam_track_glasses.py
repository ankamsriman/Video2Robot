"""Follow each glass through a video with SAM 2, by its shape rather than its shine.

Runs in the separate SAM environment (.venv-sam), so the main pipeline never depends on
PyTorch. Claude's saved starting points say which things are glasses; SAM 2.1 tiny then
outlines each one in every sampled frame. Writes each glass's track to
data/processed/<video>/sam_tracks.json, and an overlay video with each glass coloured.

.venv-sam/bin/python scripts/sam_track_glasses.py --video data/raw/task2_triangle_sunlight_take1.mov
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "data/processed/task2_setups/claude_points.json"
CHECKPOINT = ROOT / "checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
COLOURS = ((60, 200, 60), (230, 140, 30), (60, 60, 230), (40, 200, 230), (200, 60, 200))


def _sample_frames(video: Path, folder: Path, stride: int, width: int) -> tuple[float, float, int]:
    """Save every `stride`-th frame, shrunk to `width`, as numbered JPEGs for SAM 2."""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    scale = None
    index = saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            scale = width / frame.shape[1]
            small = cv2.resize(frame, (width, round(frame.shape[0] * scale)))
            cv2.imwrite(str(folder / f"{saved:05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        index += 1
    capture.release()
    return fps, scale or 1.0, saved


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3, help="use every Nth frame")
    parser.add_argument("--width", type=int, default=810, help="frame width given to SAM 2")
    args = parser.parse_args()
    points = json.loads(POINTS.read_text())[args.video.stem]
    output = ROOT / "data/processed" / args.video.stem
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="sam_frames_"))
    started = time.monotonic()
    try:
        fps, scale, count = _sample_frames(args.video, folder, args.stride, args.width)
        device = _device()
        predictor = build_sam2_video_predictor(MODEL_CONFIG, str(CHECKPOINT), device=device)
        tracks: dict[int, list] = {i: [] for i in range(len(points))}
        masks_by_frame: dict[int, dict[int, np.ndarray]] = {}
        with torch.inference_mode():
            state = predictor.init_state(video_path=str(folder))
            for glass, (x, y) in enumerate(points):
                predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=glass,
                    points=np.array([[x * scale, y * scale]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )  # fmt: skip
            for frame, ids, logits in predictor.propagate_in_video(state):
                masks_by_frame[frame] = {}
                for glass, logit in zip(ids, logits, strict=True):
                    mask = (logit[0] > 0).cpu().numpy()
                    masks_by_frame[frame][int(glass)] = mask
                    ys, xs = np.nonzero(mask)
                    if len(xs) == 0:
                        tracks[int(glass)].append([frame * args.stride, None, None, 0])
                        continue
                    # The mask's top middle is the glass's closed top, as the reader sees it.
                    top = ys.min() + max(int((ys.max() - ys.min()) * 0.1), 1)
                    row = xs[ys <= top]
                    tracks[int(glass)].append(
                        [frame * args.stride, float(row.mean() / scale),
                         float(ys.min() / scale), int(len(xs))]
                    )  # fmt: skip
        frames = sorted(folder.glob("*.jpg"))
        overlay = output / "sam_tracks.mp4"
        first = cv2.imread(str(frames[0]))
        size = (first.shape[1] // 2 * 2, first.shape[0] // 2 * 2)
        writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter.fourcc(*"mp4v"), fps / args.stride, size)
        for index, path in enumerate(frames):
            image = cv2.imread(str(path))[: size[1], : size[0]]
            for glass, mask in masks_by_frame.get(index, {}).items():
                colour = np.array(COLOURS[glass % len(COLOURS)], dtype=np.uint8)
                region = mask[: size[1], : size[0]]
                image[region] = (0.5 * image[region] + 0.5 * colour).astype(np.uint8)
            cv2.putText(image, f"SAM 2: {index * args.stride / fps:4.1f}s", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(image)
        writer.release()
        result = {
            "video": args.video.stem,
            "fps": fps,
            "stride": args.stride,
            "device": str(device),
            "seconds": round(time.monotonic() - started, 1),
            "frames_used": count,
            "prompts": points,
            "tracks": {str(k): v for k, v in tracks.items()},
        }
        (output / "sam_tracks.json").write_text(json.dumps(result))
        print(json.dumps({k: result[k] for k in ("video", "device", "seconds", "frames_used")}))
    finally:
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    main()
