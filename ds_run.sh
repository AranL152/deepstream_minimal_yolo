#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  ./ds_run.sh [--rt|--fast] [--w N --h N] [--fps N] [--out out.mp4] <video.mp4> <nvinfer_config.txt>

Defaults:
  --rt        : realtime display (sync=true)
  --fast      : fastest display (sync=false)
  --w/--h     : 1920x1080
  --fps       : not forced (uses timestamps); set if you need to clamp e.g. --fps 25
  --out       : if provided, writes annotated mp4 instead of showing a window
Notes:
  - In WSL we default-disable NVIDIA decoders (avoids /dev/nvidia0 errors).
EOF
}

MODE="rt"
W=1920
H=1080
FPS=""
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rt) MODE="rt"; shift ;;
    --fast) MODE="fast"; shift ;;
    --w) W="$2"; shift 2 ;;
    --h) H="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) break ;;
  esac
done

if [[ $# -lt 2 ]]; then
  usage; exit 1
fi

VIDEO="$1"
CFG="$2"

# WSL/X11 env (safe even if already set)
export DISPLAY="${DISPLAY:-:0}"
export QT_X11_NO_MITSHM=1
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"

# WSL safety: prevent decodebin from auto-picking NVIDIA decoders that require /dev/nvidia0
export GST_PLUGIN_FEATURE_RANK="${GST_PLUGIN_FEATURE_RANK:-nvh264dec:0,nvh265dec:0,nvv4l2decoder:0}"

SYNC="true"
[[ "$MODE" == "fast" ]] && SYNC="false"

# Make a proper file:// URI (handles spaces etc.)
URI="$(python3 -c 'import pathlib,urllib.parse,sys; p=pathlib.Path(sys.argv[1]).resolve(); print("file://" + urllib.parse.quote(str(p)))' "$VIDEO")"

# Optional: force fps after videorate (only if user asked)
FPS_STAGE=""
if [[ -n "$FPS" ]]; then
  FPS_STAGE=' ! videorate ! "video/x-raw,framerate='"$FPS"'/1"'
fi

# Choose sink: display OR file output
if [[ -z "$OUT" ]]; then
  SINK='ximagesink sync='"$SYNC"
else
  # Prefer x264enc if present, else fallback to avenc_h264
  if gst-inspect-1.0 x264enc >/dev/null 2>&1; then
    ENC='x264enc tune=zerolatency speed-preset=ultrafast'
  else
    ENC='avenc_h264'
  fi
  # Write mp4
  SINK="$ENC ! mp4mux ! filesink location=\"$OUT\""
fi

CMD=$(cat <<EOF
gst-launch-1.0 -e \\
  uridecodebin uri="$URI" name=dec \\
  dec. ! queue ! "video/x-raw" ! videoconvert ! videoscale ! "video/x-raw,width=$W,height=$H" $FPS_STAGE ! \\
  nvvideoconvert ! "video/x-raw(memory:NVMM),format=NV12,width=$W,height=$H" ! m.sink_0 \\
  nvstreammux name=m batch-size=1 width=$W height=$H batched-push-timeout=4000000 live-source=0 ! \\
  nvinfer config-file-path="$CFG" ! \\
  nvdsosd ! nvvideoconvert ! "video/x-raw,format=RGBA" ! videoconvert ! \\
  $SINK
EOF
)

echo
echo "Running:"
echo "$CMD"
echo

eval "$CMD"
