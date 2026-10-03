#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  DeepInference (car) vs DeepStream – apples-to-apples benchmark
#
#  Both pipelines run the same YOLOv8n model on the same video.
#  Reports: throughput (FPS), per-frame latency, GPU memory.
#
#  usage: ./compare.sh [video] [precision]
#    video     defaults to ../../Rainbow Six 2024-07-30 15-57-26.mp4
#    precision fp16 (default) or fp32
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

VIDEO="${1:-../../Rainbow Six 2024-07-30 15-57-26.mp4}"
VIDEO="$(realpath "$VIDEO")"
PREC="${2:-fp16}"

TRTEXEC="${TRTEXEC:-$HOME/TensorRT-8.6.1.6/bin/trtexec}"
TRT_DIR="${TRT_DIR:-$HOME/TensorRT-8.6.1.6}"
CUDA_DIR="${CUDA_DIR:-/usr/local/cuda-12.6}"
MAX_FRAMES="${MAX_FRAMES:-500}"

CAR_ONNX="yolov8n_car.onnx"
DS_ONNX="yolov8n_ds.onnx"
CAR_ENGINE="yolov8n_car_${PREC}.engine"
DS_ENGINE="yolov8n_ds_${PREC}.engine"
DS_CONFIG="nvinfer_yolov8n_${PREC}.txt"

header() { printf "\n\033[1;36m══ %s ══\033[0m\n" "$1"; }
ok()     { printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail()   { printf "  \033[31m✗ %s\033[0m\n" "$1"; exit 1; }

# ── 1. Export model ─────────────────────────────────────────────────
header "Step 1/5: Export YOLOv8n ONNX"
if [[ -f "$CAR_ONNX" && -f "$DS_ONNX" ]]; then
    ok "ONNX files already exist"
else
    pip3 install -q onnx 2>/dev/null || true
    python3 export_yolov8n.py
    [[ -f "$CAR_ONNX" && -f "$DS_ONNX" ]] || fail "ONNX export failed"
    ok "Exported"
fi

# ── 2. Build TensorRT engines ──────────────────────────────────────
header "Step 2/5: Build TRT engines (${PREC})"
TRT_FLAGS="--warmUp=500 --duration=5"
[[ "$PREC" == "fp16" ]] && TRT_FLAGS="$TRT_FLAGS --fp16"

if [[ ! -f "$CAR_ENGINE" ]]; then
    echo "  Building car engine (this may take a few minutes)..."
    "$TRTEXEC" --onnx="$CAR_ONNX" --saveEngine="$CAR_ENGINE" $TRT_FLAGS \
        > "trtexec_car_${PREC}.log" 2>&1
    ok "Car engine → $CAR_ENGINE"
else
    ok "Car engine exists"
fi

if [[ ! -f "$DS_ENGINE" ]]; then
    echo "  Building DS engine (this may take a few minutes)..."
    "$TRTEXEC" --onnx="$DS_ONNX" --saveEngine="$DS_ENGINE" $TRT_FLAGS \
        > "trtexec_ds_${PREC}.log" 2>&1
    ok "DS engine → $DS_ENGINE"
else
    ok "DS engine exists"
fi

# ── 3. Build car benchmark binary ─────────────────────────────────
header "Step 3/5: Build bench_car"
BENCH_BIN="./bench_car"
if [[ ! -f "$BENCH_BIN" ]] || [[ ../car/deep_inference.cpp -nt "$BENCH_BIN" ]]; then
    g++ -O2 -std=c++17 -Wall -Wno-deprecated-declarations \
        -I../car -I"$TRT_DIR/include" -I"$CUDA_DIR/include" \
        $(pkg-config --cflags opencv4) \
        -o "$BENCH_BIN" ../bench_car.cpp ../car/deep_inference.cpp \
        -L"$TRT_DIR/lib" -lnvinfer \
        -L"$CUDA_DIR/lib64" -lcudart \
        $(pkg-config --libs opencv4) \
        -Wl,-rpath,"$TRT_DIR/lib"
    ok "Compiled bench_car"
else
    ok "bench_car already built"
fi

# ── 4. Run car benchmark ──────────────────────────────────────────
header "Step 4/5: Car pipeline (DeepInference C++ / raw TRT)"
echo "  Running $MAX_FRAMES frames..."
CAR_OUT="$("$BENCH_BIN" "$CAR_ENGINE" "$VIDEO" "$MAX_FRAMES" 2>&1)"
echo "$CAR_OUT"
echo "$CAR_OUT" > "result_car_${PREC}.txt"

# Extract car FPS from output
CAR_E2E_FPS=$(echo "$CAR_OUT" | grep 'end-to-end throughput' | grep -oP '[0-9]+\.[0-9]+')
CAR_INF_FPS=$(echo "$CAR_OUT" | grep 'inference-only' | grep -oP '[0-9]+\.[0-9]+')

# ── 5. Run DeepStream benchmark ───────────────────────────────────
header "Step 5/5: DeepStream pipeline (nvinfer + GStreamer)"

# Suppress NVIDIA HW decoders in WSL (they often fail)
export GST_PLUGIN_FEATURE_RANK="nvh264dec:0,nvh265dec:0,nvv4l2decoder:0"

read -r VW VH < <(ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0 "$VIDEO" | tr , " ")

URI="file://$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))' "$VIDEO")"

echo "  Video: ${VW}x${VH}"
echo "  Running DeepStream pipeline (full video, max throughput)..."

DS_OUT=$(gst-launch-1.0 -v \
    uridecodebin uri="$URI" ! queue ! videoconvert \
    ! nvvideoconvert ! "video/x-raw(memory:NVMM),format=NV12" \
    ! m.sink_0 nvstreammux name=m batch-size=1 width="$VW" height="$VH" \
      batched-push-timeout=4000000 live-source=0 \
    ! nvinfer config-file-path="$DS_CONFIG" \
    ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false \
    2>&1)

DS_FPS=$(echo "$DS_OUT" | grep -oP 'average: [0-9.]+' | tail -1 | grep -oP '[0-9.]+')
echo "  DeepStream steady-state FPS: ${DS_FPS:-N/A}"
echo "$DS_OUT" > "result_ds_${PREC}.txt"

# ── Also run trtexec for pure engine throughput baseline ──────────
header "Bonus: Pure TRT engine throughput (trtexec, no I/O overhead)"
TRTEXEC_OUT=$("$TRTEXEC" --loadEngine="$CAR_ENGINE" --warmUp=500 --duration=5 --useCudaGraph 2>&1)
TRT_QPS=$(echo "$TRTEXEC_OUT" | grep 'Throughput:' | grep -oP '[0-9]+\.[0-9]+' | head -1)
TRT_LAT=$(echo "$TRTEXEC_OUT" | grep 'Latency:.*mean' | grep -oP 'mean = [0-9.]+' | grep -oP '[0-9.]+')
echo "  trtexec throughput: ${TRT_QPS:-N/A} qps"
echo "  trtexec mean latency: ${TRT_LAT:-N/A} ms"

# ═══════════════════════════════════════════════════════════════════
header "COMPARISON RESULTS — YOLOv8n ${PREC^^}"
echo ""
printf "  %-40s %12s\n" "" "FPS"
printf "  %-40s %12s\n" "────────────────────────────────────────" "────────────"
printf "  %-40s %12s\n" "TRT engine only (trtexec, no decode)" "${TRT_QPS:-N/A}"
printf "  %-40s %12s\n" "Car pipeline (decode+CPU preproc+TRT)" "${CAR_E2E_FPS:-N/A}"
printf "  %-40s %12s\n" "  └ inference-only (letterbox+infer)" "${CAR_INF_FPS:-N/A}"
printf "  %-40s %12s\n" "DeepStream (decode+GPU preproc+nvinfer)" "${DS_FPS:-N/A}"
echo ""
echo "  Car pipeline:  CPU letterbox → CPU normalize → cudaMemcpy H2D → TRT → cudaMemcpy D2H → CPU postprocess"
echo "  DeepStream:    CPU decode → GPU nvvideoconvert → GPU nvinfer (preprocess+TRT+parse) → fakesink"
echo ""

if [[ -n "${CAR_INF_FPS:-}" && -n "${DS_FPS:-}" ]]; then
    RATIO=$(python3 -c "print(f'{float(\"$DS_FPS\")/float(\"$CAR_INF_FPS\"):.2f}')")
    echo "  DeepStream / Car inference-only ratio: ${RATIO}x"
    echo ""
fi

echo "  Full per-frame latency breakdown is in result_car_${PREC}.txt"
echo "  DeepStream raw output is in result_ds_${PREC}.txt"
echo ""
echo "  NOTE: These results are from your desktop GPU (RTX 4080 SUPER)."
echo "  On a Jetson, the ratio may differ due to:"
echo "    - Shared CPU/GPU memory (no H2D/D2H copy penalty)"
echo "    - Weaker CPU (CPU preprocessing costs more relatively)"
echo "    - NVDEC hardware decoder (DeepStream can use it, car pipeline can't)"
echo ""
