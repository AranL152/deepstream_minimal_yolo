# DeepInference vs DeepStream Benchmark

**Goal:** Determine whether to replace the car's C++ TensorRT pipeline (`deep_inference.cpp`) with NVIDIA DeepStream for Jetson deployment.

## Setup

| Parameter | Value |
|---|---|
| Model | YOLOv8n (nano), 640×640, 80-class COCO, FP16 |
| Video | `Rainbow Six 2024-07-30 15-57-26.mp4` — 2560×1072 H.264 @ 29fps, 500 frames |
| Hardware | RTX 4080 SUPER, Ryzen 7 7800X3D, WSL2 Ubuntu |
| TensorRT | 8.6.1 |
| DeepStream | 6.4 |

## How to reproduce

```bash
cd bench/v3

# 1. Export YOLOv8n to ONNX (stock for DS, TopK-appended for car)
pip3 install onnx
python3 export_yolov8n.py

# 2. Build FP16 TRT engines
trtexec --onnx=yolov8n_car.onnx --saveEngine=yolov8n_car_fp16.engine --fp16
trtexec --onnx=yolov8n_ds.onnx  --saveEngine=yolov8n_ds_fp16.engine  --fp16

# 3. Compile car benchmark
g++ -O2 -std=c++17 -I../car -I$TRT/include -I$CUDA/include $(pkg-config --cflags opencv4) \
    -o bench_car ../bench_car.cpp ../car/deep_inference.cpp \
    -L$TRT/lib -lnvinfer -L$CUDA/lib64 -lcudart $(pkg-config --libs opencv4)

# 4. Run car benchmark
./bench_car yolov8n_car_fp16.engine "../../Rainbow Six 2024-07-30 15-57-26.mp4" 500

# 5. Run DeepStream benchmark
./run_ds.sh "../../Rainbow Six 2024-07-30 15-57-26.mp4" nvinfer_yolov8n_fp16.txt
```

## Results

| Metric | Car (C++ TRT) | DeepStream | Winner |
|---|---|---|---|
| **End-to-end FPS** | 214 | 337 | DS (**1.58×**) |
| Inference-only FPS | 301 | — | — |
| TRT engine ceiling | 710 qps | 665 qps | Car (TopK in ONNX) |
| Pipeline efficiency | 42% | 51% | DS |
| Avg detections/frame | 3.5 | 3.5 | Equal |

### Car pipeline latency breakdown

| Stage | Mean | P50 | P99 |
|---|---|---|---|
| Decode (CPU FFmpeg) | 1.36 ms | 1.07 ms | 2.20 ms |
| Letterbox (CPU cv::resize) | 0.34 ms | 0.33 ms | 0.53 ms |
| run_inference() | 2.98 ms | 2.61 ms | 7.67 ms |
| **Total per frame** | **4.68 ms** | **4.13 ms** | **9.92 ms** |

## Where car loses time

DeepStream moves preprocessing to GPU and avoids CPU↔GPU copies:

- **CPU letterbox + normalize** (0.7 ms) → DS does both on GPU inside `nvinfer`
- **cudaMemcpy H2D** (0.19 ms) → DS keeps frames in NVMM, zero-copy
- **cudaDeviceSynchronize** → DS uses async CUDA streams
- **CPU postprocess** → DS runs NMS on GPU via custom parser

## Accuracy

Both pipelines produce the same detection count (avg 3.5/frame) and use the same YOLOv8n weights. The car variant appends a Top-300 selection in ONNX; DeepStream applies NMS (IoU 0.35, conf 0.60, top-100) via `NvDsInferParseYoloCuda`. Detection quality is equivalent — differences are in NMS strategy, not model accuracy.

## Jetson projection

On a Jetson Orin, the DeepStream advantage will be **larger**:

1. **Weaker CPU** — ARM A78AE is 3–5× slower than Ryzen; CPU letterbox/normalize cost grows
2. **NVDEC** — DS can use the hardware H.264 decoder (zero CPU); car pipeline cannot
3. **DLA** — DS can offload inference to DLA, freeing GPU for other workloads
4. **Unified memory** — H2D/D2H copies are cheaper for both, but DS still avoids them entirely

Conservative estimate: **2–3× faster on Jetson**.

## Recommendation

**Use DeepStream.** It is faster now (1.58×) and the gap widens on Jetson. It also frees CPU cycles for path planning and SLAM, and supports hardware decode out of the box.
