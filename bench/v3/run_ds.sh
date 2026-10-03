#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VIDEO="${1:-/home/aranl/deepstream_minimal_yolo/Rainbow Six 2024-07-30 15-57-26.mp4}"
CFG="${2:-nvinfer_yolov8n_fp16.txt}"

export GST_PLUGIN_FEATURE_RANK="nvh264dec:0,nvh265dec:0,nvv4l2decoder:0"

read -r VW VH < <(ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0 "$VIDEO" | tr , " ")

URI="file://$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "$VIDEO")"

echo "Video: ${VW}x${VH}"
echo "Config: $CFG"
echo "Running DeepStream pipeline..."

gst-launch-1.0 -v \
    uridecodebin uri="$URI" ! queue ! videoconvert \
    ! nvvideoconvert ! "video/x-raw(memory:NVMM),format=NV12" \
    ! m.sink_0 nvstreammux name=m batch-size=1 width="$VW" height="$VH" \
      batched-push-timeout=4000000 live-source=0 \
    ! nvinfer config-file-path="$CFG" \
    ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false \
    2>&1 | grep -o 'average: [0-9.]*' | tail -1 | sed 's/average: /DeepStream steady-state FPS: /'
