"""Ask a vision language model to turn an instruction and pictures into a plan.

The key is read from the environment or the project's `.env` file, which git ignores. It is
never logged or written anywhere else.
"""

import base64
import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from src.reasoning.instruction import (
    InstructionError,
    InstructionPlan,
    Order,
    Task,
    parse_plan,
)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
KEY_NAME = "ANTHROPIC_API_KEY"
logger = logging.getLogger(__name__)


class VisionModel(Protocol):
    def ask(self, prompt: str, images: Sequence[bytes]) -> str: ...


@dataclass(frozen=True)
class VlmConfig:
    model: str
    max_tokens: int
    timeout_seconds: float
    frames: int
    image_width: int


def load_api_key(env_file: Path) -> str:
    """The key from the environment, or else from a KEY=value line in `env_file`."""
    key = os.environ.get(KEY_NAME, "").strip()
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == KEY_NAME:
                key = value.strip()
    if not key:
        raise InstructionError(f"No {KEY_NAME} found in the environment or {env_file.name}")
    return key


class AnthropicVision:
    """Claude, through its Messages API, with no extra package needed."""

    def __init__(self, config: VlmConfig, key: str) -> None:
        self._config = config
        self._key = key

    def ask(self, prompt: str, images: Sequence[bytes]) -> str:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(image).decode(),
                },
            }
            for image in images
        ]
        body = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "messages": [{"role": "user", "content": [*content, {"type": "text", "text": prompt}]}],
        }
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(body).encode(),
            headers={
                "x-api-key": self._key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout_seconds) as reply:
                answer = json.load(reply)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:200].replace(self._key, "***")
            raise InstructionError(f"The model service refused: {error.code} {detail}") from error
        except urllib.error.URLError as error:
            raise InstructionError(f"Couldn't reach the model service: {error.reason}") from error
        logger.info("Model %s used %s", answer.get("model"), answer.get("usage"))
        return "".join(part.get("text", "") for part in answer.get("content", []))


def video_frames(video: Path, count: int, width: int) -> list[bytes]:
    """Evenly spaced frames from the first half of a video, as small JPEG pictures."""
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    pictures = []
    for index in np.linspace(0, max(total // 2, 1), count).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        scale = width / frame.shape[1]
        small = cv2.resize(frame, (width, int(frame.shape[0] * scale)))
        pictures.append(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    capture.release()
    if not pictures:
        raise InstructionError(f"Couldn't read frames from {video.name}")
    return pictures


def _prompt(instruction: str, glass_count: int, video_has_flip: bool) -> str:
    tasks = ", ".join(task.value for task in Task)
    orders = ", ".join(order.value for order in Order)
    flip_note = (
        "The video also shows a fourth glass being flipped."
        if video_has_flip
        else "No glass is flipped in the video."
    )
    return f"""You plan a task for a robot at a table. The pictures are frames from a video of
a person who stands at one edge of the table, facing a camera at the far edge. The camera
faces the person, so the person's left appears on the right of each picture. The robot will
stand where the person stands, facing the same way.

Measurements found {glass_count} glasses standing upside down on the table. {flip_note}

Tasks: "triangle" means two glasses side by side and a third across their tops.
"flipped_tower" means that triangle with a fourth glass flipped and stood on top.
Orders say which glasses go first: the first two become the lower glasses and the third goes
on top. For "flipped_tower", the order covers all four glasses and the last one is the glass
that gets flipped, so "nearest_first" flips the farthest glass. "nearest" means nearest to the person's edge of the table, and "left" means the
person's left. "as_in_video" keeps the person's own order.

Instruction from the user: "{instruction}"

Reply with only one JSON object and nothing else:
{{"task": one of [{tasks}], "order": one of [{orders}],
"understood": true or false, "reason": "one short sentence"}}
Set "understood" to false if the instruction asks for something outside these options."""


def find_glass_tops(
    model: VisionModel, picture: bytes, width: int, height: int
) -> list[tuple[float, float]]:
    """Ask where every upside-down glass's top is, in the picture's own pixels."""
    prompt = f"""This is a photo of a table with upside-down steel drinking glasses standing
on it. The image is {width} pixels wide and {height} pixels tall. For every glass standing on
the table, give the pixel position of the centre of its round, closed top. Ignore anything
that isn't a drinking glass. Reply with only JSON: {{"glasses": [{{"x": number, "y": number}}]}}"""
    reply = model.ask(prompt, [picture])
    found = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        glasses = json.loads(found.group(0))["glasses"] if found else None
        points = [(float(g["x"]), float(g["y"])) for g in glasses]
    except (TypeError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise InstructionError(f"The model didn't give glass positions: {reply[:120]}") from error
    return [(x, y) for x, y in points if 0 <= x <= width and 0 <= y <= height]


def plan_from_instruction(
    model: VisionModel,
    instruction: str,
    images: Sequence[bytes],
    glass_count: int,
    video_has_flip: bool,
) -> InstructionPlan:
    reply = model.ask(_prompt(instruction, glass_count, video_has_flip), images)
    return parse_plan(reply)
