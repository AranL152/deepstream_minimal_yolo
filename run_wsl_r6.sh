#!/usr/bin/env bash
set -euo pipefail

VIDEO=${1:-"Rainbow Six 2024-07-31 13-30-24.mp4"}
CFG=${2:-"DeepStream-Yolo/config_infer_primary_yoloV8.txt"}

export DISPLAY=${DISPLAY:-:0}
export QT_X11_NO_MITSHM=1
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp}

# WSL: avoid NVIDIA hardware decoders that may try to open /dev/nvidia0
export GST_PLUGIN_FEATURE_RANK=${GST_PLUGIN_FEATURE_RANK:-"nvh264dec:0,nvh265dec:0,nvv4l2decoder:0"}

make -C "$(dirname "$0")" -s

./minimal_yolo "$VIDEO" "$CFG"
