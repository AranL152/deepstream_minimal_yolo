#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cmath>
#include "nvdsinfer_custom_impl.h"

#define NUM_CLASSES 80

__global__ void decodeTensorYoloV8(NvDsInferParseObjectInfo *binfo, const float* output, const uint numDetections, const uint netW, const uint netH, const float confThreshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numDetections) return;

    float cx = output[0 * numDetections + idx];
    float cy = output[1 * numDetections + idx];
    float w  = output[2 * numDetections + idx];
    float h  = output[3 * numDetections + idx];

    int bestClassId = 0;
    float bestScore = -1.0f;
    for (int c = 0; c < NUM_CLASSES; c++) {
        float score = output[(4 + c) * numDetections + idx];
        if (score > bestScore) { bestScore = score; bestClassId = c; }
    }

    float confidence = 1.0f / (1.0f + expf(-bestScore));
    if (confidence < confThreshold) { binfo[idx].detectionConfidence = 0.0f; return; }

    float x1 = cx - w / 2.0f;
    float y1 = cy - h / 2.0f;
    x1 = fminf((float)netW, fmaxf(0.0f, x1));
    y1 = fminf((float)netH, fmaxf(0.0f, y1));
    w = fminf((float)netW - x1, fmaxf(0.0f, w));
    h = fminf((float)netH - y1, fmaxf(0.0f, h));
    if (w < 1.0f || h < 1.0f) { binfo[idx].detectionConfidence = 0.0f; return; }

    binfo[idx].left = x1;
    binfo[idx].top = y1;
    binfo[idx].width = w;
    binfo[idx].height = h;
    binfo[idx].detectionConfidence = confidence;
    binfo[idx].classId = bestClassId;
}

static bool NvDsInferParseCustomYoloCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList) {
    if (outputLayersInfo.empty()) { std::cerr << "ERROR: No output layer" << std::endl; return false; }
    const NvDsInferLayerInfo& output = outputLayersInfo[0];
    const uint numDetections = 8400;
    float confThreshold = detectionParams.perClassPreclusterThreshold.size() > 0 ? detectionParams.perClassPreclusterThreshold[0] : 0.25f;

    thrust::device_vector<NvDsInferParseObjectInfo> d_objects(numDetections);
    decodeTensorYoloV8<<<(numDetections + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_objects.data()), (const float*)output.buffer, numDetections, networkInfo.width, networkInfo.height, confThreshold);
    cudaDeviceSynchronize();

    thrust::host_vector<NvDsInferParseObjectInfo> h_objects = d_objects;
    objectList.clear();
    for (uint i = 0; i < numDetections; i++) { if (h_objects[i].detectionConfidence > 0.0f) objectList.push_back(h_objects[i]); }
    return true;
}

extern "C" bool NvDsInferParseYoloCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList) {
    return NvDsInferParseCustomYoloCuda(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloCuda);
