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

echo "Setup complete. Run the tests with: .venv/bin/pytest"
