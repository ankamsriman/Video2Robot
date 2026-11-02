#!/usr/bin/env bash
# Creates the Python environment and downloads the G1 robot model.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
# The exact versions tested. A newer solver or robot model changes how the robot moves.
.venv/bin/pip install -q -r requirements.lock.txt

MENAGERIE_VERSION=a662de603224c47123b0783821c58ad4402ad3e3
if [ ! -d third_party/mujoco_menagerie ]; then
  mkdir -p third_party
  git clone -q --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
  git -C third_party/mujoco_menagerie sparse-checkout set unitree_g1
  git -C third_party/mujoco_menagerie checkout -q "$MENAGERIE_VERSION"
fi
if [ "$(git -C third_party/mujoco_menagerie rev-parse HEAD)" != "$MENAGERIE_VERSION" ]; then
  echo "Warning: third_party/mujoco_menagerie isn't the tested version, $MENAGERIE_VERSION" >&2
fi

if [ ! -f checkpoints/hand_landmarker.task ]; then
  mkdir -p checkpoints
  curl -fsSL -o checkpoints/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
fi

echo "Setup complete. Run the tests with: .venv/bin/pytest"
