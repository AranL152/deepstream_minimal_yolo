// nvdsparsebbox_Yolo.cpp - Fixed for YOLOv8 [1, 84, 8400] output format

#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

static const int NUM_CLASSES = 80;

// Sigmoid function
inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// NMS comparison
static bool cmpFunc(const NvDsInferObjectDetectionInfo& a, 
                    const NvDsInferObjectDetectionInfo& b) {
    return a.detectionConfidence > b.detectionConfidence;
}

// IoU calculation
static float iou(const NvDsInferObjectDetectionInfo& a, 
                 const NvDsInferObjectDetectionInfo& b) {
    float left   = std::max(a.left, b.left);
    float top    = std::max(a.top, b.top);
    float right  = std::min(a.left + a.width, b.left + b.width);
    float bottom = std::min(a.top + a.height, b.top + b.height);
    
    if (right <= left || bottom <= top) return 0.0f;
    
    float interArea = (right - left) * (bottom - top);
    float unionArea = a.width * a.height + b.width * b.height - interArea;
    
    return interArea / unionArea;
}

// NMS
static void nms(std::vector<NvDsInferObjectDetectionInfo>& detections, 
                float nmsThreshold) {
    std::sort(detections.begin(), detections.end(), cmpFunc);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].detectionConfidence == 0) continue;
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (detections[j].detectionConfidence == 0) continue;
            if (detections[i].classId != detections[j].classId) continue;
            
            if (iou(detections[i], detections[j]) > nmsThreshold) {
                detections[j].detectionConfidence = 0;
            }
        }
    }
    
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
            [](const NvDsInferObjectDetectionInfo& d) { 
                return d.detectionConfidence == 0; 
            }),
        detections.end()
    );
}

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    std::cerr << "PARSER CALLED!" << std::endl; if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: No output layers found!" << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const float* output = static_cast<const float*>(layer.buffer);
    
    // YOLOv8 output shape: [1, 84, 8400]
    // 84 = 4 (cx, cy, w, h) + 80 (class scores)
    // 8400 = number of detections
    
    const int numDetections = 8400;
    const int numAttributes = 84;  // 4 bbox + 80 classes
    
    // Debug: Print first few raw values to verify format
    static bool firstCall = true;
    if (firstCall) {
        std::cout << "\n=== YOLOv8 Parser Debug ===" << std::endl;
        std::cout << "Layer name: " << layer.layerName << std::endl;
        std::cout << "Dims: ";
        for (unsigned int i = 0; i < layer.inferDims.numDims; i++) {
            std::cout << layer.inferDims.d[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Network: " << networkInfo.width << "x" << networkInfo.height << std::endl;
        
        // Print first detection's raw values
        std::cout << "First detection raw values:" << std::endl;
        std::cout << "  cx (row 0): " << output[0 * numDetections + 0] << std::endl;
        std::cout << "  cy (row 1): " << output[1 * numDetections + 0] << std::endl;
        std::cout << "  w  (row 2): " << output[2 * numDetections + 0] << std::endl;
        std::cout << "  h  (row 3): " << output[3 * numDetections + 0] << std::endl;
        std::cout << "  class0 (row 4): " << output[4 * numDetections + 0] << std::endl;
        firstCall = false;
    }
    
    float confThreshold = detectionParams.perClassThreshold.size() > 0 
                          ? detectionParams.perClassThreshold[0] 
                          : 0.25f;
    float nmsThreshold = 0.45f;
    
    std::vector<NvDsInferObjectDetectionInfo> detections;
    
    // Iterate over all 8400 detections
    for (int i = 0; i < numDetections; ++i) {
        // YOLOv8 format: [1, 84, 8400] - row-major
        // Row 0: all cx values
        // Row 1: all cy values  
        // Row 2: all w values
        // Row 3: all h values
        // Row 4-83: class scores
        
        float cx = output[0 * numDetections + i];
        float cy = output[1 * numDetections + i];
        float w  = output[2 * numDetections + i];
        float h  = output[3 * numDetections + i];
        
        // Find best class
        int bestClassId = 0;
        float bestScore = -1.0f;
        
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float score = output[(4 + c) * numDetections + i];
            if (score > bestScore) {
                bestScore = score;
                bestClassId = c;
            }
        }
        
        // YOLOv8 does NOT use sigmoid on bbox, only on class scores
        float confidence = bestScore;
        
        if (confidence < confThreshold) continue;
        
        // Convert from center format to corner format
        // Coordinates are already in pixels (model input size)
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        
        // Clamp to valid range
        x1 = std::max(0.0f, std::min(x1, (float)networkInfo.width));
        y1 = std::max(0.0f, std::min(y1, (float)networkInfo.height));
        w = std::max(0.0f, std::min(w, (float)networkInfo.width - x1));
        h = std::max(0.0f, std::min(h, (float)networkInfo.height - y1));
        
        if (w < 1 || h < 1) continue;
        
        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClassId;
        obj.detectionConfidence = confidence;
        obj.left = x1;
        obj.top = y1;
        obj.width = w;
        obj.height = h;
        
        detections.push_back(obj);
    }
    
    // Apply NMS
    nms(detections, nmsThreshold);
    
    objectList = std::move(detections);
    
    return true;
}

// Register the parser
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
