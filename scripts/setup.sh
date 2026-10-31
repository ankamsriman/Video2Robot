#!/usr/bin/env bash
# Creates the Python environment and downloads the G1 robot model.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

if [ ! -d third_party/mujoco_menagerie ]; then
  mkdir -p third_party
  git clone -q --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
  git -C third_party/mujoco_menagerie sparse-checkout set unitree_g1
fi

if [ ! -f checkpoints/hand_landmarker.task ]; then
  mkdir -p checkpoints
  curl -fsSL -o checkpoints/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
fi

echo "Setup complete. Run the tests with: .venv/bin/pytest"
