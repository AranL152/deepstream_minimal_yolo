#!/usr/bin/env bash
# DeepStream end-to-end throughput: CPU decode (as in WSL) -> NVMM -> nvstreammux -> nvinfer -> fakesink (no sync).
# usage: bench_ds.sh <video> <nvinfer_config> [decode-only]
set -euo pipefail
VIDEO=$(realpath "$1"); CFG=$2
export GST_PLUGIN_FEATURE_RANK="nvh264dec:0,nvh265dec:0,nvv4l2decoder:0"
read W H < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$VIDEO" | tr , " ")
URI="file://$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))' "$VIDEO")"
if [[ "${3:-}" == "decode-only" ]]; then
  TAIL="fpsdisplaysink video-sink=fakesink text-overlay=false sync=false"
else
  TAIL="nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! m.sink_0 nvstreammux name=m batch-size=1 width=$W height=$H batched-push-timeout=4000000 live-source=0 ! nvinfer config-file-path=$CFG ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false"
fi
# fpsdisplaysink's running average excludes pipeline startup (engine deserialize etc.)
gst-launch-1.0 -v uridecodebin uri="$URI" ! queue ! videoconvert ! $TAIL 2>&1 \
  | grep -o 'average: [0-9.]*' | tail -1 | sed 's/average: /steady-state FPS: /'
