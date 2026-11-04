"""Join showcase clips into one demo reel, each after a short title card.

Every clip is scaled and padded to 1920 x 1080, so clips of any size join cleanly.

PYTHONPATH=. .venv/bin/python scripts/make_reel.py --output docs/videos/reel_real_world.mp4 \\
    "Title one|Line under it|path/to/clip.mp4" "Title two|Line|path/to/other.mp4"
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

WIDTH, HEIGHT, FPS = 1920, 1080, 30
CARD_SECONDS = 3.0
BACKGROUND = (28, 28, 28)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _card(path: Path, title: str, line: str) -> None:
    """A still title card, written as a short video."""
    image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
    for text, scale, y, colour in ((title, 2.0, 500, (255, 255, 255)), (line, 1.0, 590, (200, 200, 200))):
        width = cv2.getTextSize(text, FONT, scale, 2)[0][0]
        cv2.putText(image, text, ((WIDTH - width) // 2, y), FONT, scale, colour, 2, cv2.LINE_AA)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for _ in range(int(CARD_SECONDS * FPS)):
        writer.write(image)
    writer.release()


def _normalise(ffmpeg: str, source: Path, target: Path) -> None:
    """Scale a clip to fit 1920 x 1080, pad the rest, and set one frame rate."""
    fit = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x1c1c1c,fps={FPS}"
    )
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", str(source), "-vf", fit, "-an",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(target)],
        check=True,
    )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("clips", nargs="+", help='"Title|Line under it|clip.mp4"')
    args = parser.parse_args()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as folder:
        parts = []
        for index, entry in enumerate(args.clips):
            title, line, clip = entry.split("|")
            card = Path(folder) / f"{index:02d}_card.mp4"
            _card(card, title, line)
            for source, name in ((card, f"{index:02d}_a.mp4"), (Path(clip), f"{index:02d}_b.mp4")):
                target = Path(folder) / name
                _normalise(ffmpeg, source, target)
                parts.append(target)
        listing = Path(folder) / "parts.txt"
        listing.write_text("".join(f"file '{part}'\n" for part in parts))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", str(args.output)],
            check=True,
        )  # fmt: skip
    print(args.output)


if __name__ == "__main__":
    main()
