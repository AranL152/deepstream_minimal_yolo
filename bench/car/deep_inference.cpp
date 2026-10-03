/*
██╗   ██╗████████╗███████╗██████╗         ██████╗ ██╗   ██╗
██║   ██║╚══██╔══╝██╔════╝██╔══██╗        ██╔══██╗██║   ██║
██║   ██║   ██║   █████╗  ██████╔╝        ██║  ██║██║   ██║
██║   ██║   ██║   ██╔══╝  ██╔══██╗        ██║  ██║╚██╗ ██╔╝
╚██████╔╝   ██║   ██║     ██║  ██║███████╗██████╔╝ ╚████╔╝
 ╚═════╝    ╚═╝   ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝   ╚═══╝
*
* file: deep_inference.cpp
* auth: Alex Zhang
* desc: deep learning inference file in C++
*/

#include "deep_inference.hpp"

#include "debug_tensor.hpp"
#include "debug_tensor_functions.hpp"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// SET TO TRUE TO ENABLE DEBUGGING OF INPUT/OUTPUT TENSORS
const bool DEBUG_TENSORS = false;

// SET TO TRUE TO SAVE PREPROCESSED INPUT IMAGES (post-resize/letterbox, pre-normalization)
const bool DEBUG_SAVE_PREPROCESSED_IMAGES = false;

namespace {
void save_preprocessed_debug_image(const cv::Mat &image, PreprocessMethod preprocess_method) {
    if (image.empty()) {
        return;
    }

    static std::atomic<uint64_t> image_counter{0};
    constexpr const char *kDebugDir = "/home/utfr-dv/utfr_dv/debug/preprocessed_inputs";

    try {
        // std::filesystem::create_directories(kDebugDir);

        const auto stamp_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count();
        const uint64_t index = image_counter.fetch_add(1, std::memory_order_relaxed);
        const char *method_name =
            (preprocess_method == PreprocessMethod::LETTERBOX) ? "letterbox" : "transform";

        std::ostringstream filename;
        filename << kDebugDir << "/pre_" << method_name << "_" << std::setfill('0')
                 << std::setw(6) << index << "_" << stamp_ns << "_" << image.cols << "x"
                 << image.rows << ".jpg";

        if (!cv::imwrite(filename.str(), image)) {
            std::cerr << "Warning: Failed to save preprocessed debug image: " << filename.str()
                      << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "Warning: Failed to save preprocessed debug image: " << e.what()
                  << std::endl;
    }
}
} // namespace

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

static Logger gLogger;

DeepInference::DeepInference(const std::string &model_path, const std::string &model_config_path,
                             ModelType model_type, PreprocessMethod preprocess_method)
    : model_type_(model_type), input_height_(832), input_width_(832), input_channels_(3),
      max_batch_size_(1), max_buffer_height_(1920),
      max_buffer_width_(1920), // Max size for dynamic models
      conf_threshold_(0.4f), use_half_precision_(false), preprocess_method_(preprocess_method),
    last_inference_time_ms_(0.0), enable_debug_logs_(true), input_buffer_(nullptr),
    output_buffer_(nullptr),
      input_buffer_size_(0), output_buffer_size_(0) {

    // Load TensorRT engine
    if (!loadEngine(model_path)) {
        throw std::runtime_error("Failed to load TensorRT engine: " + model_path);
    }

    // Load config if provided
    if (!model_config_path.empty()) {
        loadConfig(model_config_path);
    }

    // Initialize class names
    initializeClassNames();

    // Allocate CUDA buffers
    allocateBuffers();
}

DeepInference::~DeepInference() {
    // Free CUDA buffers
    if (input_buffer_) {
        cudaFree(input_buffer_);
    }
    if (output_buffer_) {
        cudaFree(output_buffer_);
    }

    // TensorRT will clean up automatically via unique_ptr
}

bool DeepInference::loadEngine(const std::string &engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Cannot open engine file: " << engine_path << std::endl;
        return false;
    }

    // Read engine file
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // Create TensorRT runtime
    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
    if (!runtime_) {
        std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    // Deserialize engine
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size, nullptr));
    if (!engine_) {
        std::cerr << "Error: Failed to deserialize engine" << std::endl;
        return false;
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Error: Failed to create execution context" << std::endl;
        return false;
    }

    // Get input dimensions from engine
    int num_bindings = engine_->getNbBindings();
    for (int i = 0; i < num_bindings; ++i) {
        if (engine_->bindingIsInput(i)) {
            auto dims = engine_->getBindingDimensions(i);
            if (dims.nbDims == 4) {
                if (dims.d[0] > 0) {
                    max_batch_size_ = std::max(1, dims.d[0]);
                } else {
                    auto max_profile_dims =
                        engine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
                    if (max_profile_dims.nbDims == 4 && max_profile_dims.d[0] > 0) {
                        max_batch_size_ = std::max(1, max_profile_dims.d[0]);
                    }
                }

                input_height_ = dims.d[2];
                input_width_ = dims.d[3];
                input_channels_ = dims.d[1];

                // Handle dynamic dimensions (-1): use sensible defaults
                if (input_height_ <= 0) {
                    input_height_ = (model_type_ == ModelType::CUSTOM) ? 60 : 720;
                }
                if (input_width_ <= 0) {
                    input_width_ = (model_type_ == ModelType::CUSTOM) ? 50 : 1920;
                }
                if (input_channels_ <= 0) {
                    input_channels_ = 3;
                }
            }
        }
    }

    std::cout << "Loaded TensorRT engine: " << engine_path << std::endl;
    std::cout << "Input size: " << input_width_ << "x" << input_height_ << "x" << input_channels_
              << std::endl;
    std::cout << "Max batch size: " << max_batch_size_ << std::endl;

    return true;
}

bool DeepInference::loadConfig(const std::string &config_path) {
    // For now, we'll parse basic YAML in C++ or rely on Python to pass values
    // This is a simplified version - you may want to use yaml-cpp for full parsing
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open config file: " << config_path << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Simple key-value parsing (basic YAML support)
        if (line.find("imgsz:") != std::string::npos) {
            std::istringstream iss(line);
            std::string key, value;
            iss >> key >> value;
            int imgsz = std::stoi(value);
            input_height_ = imgsz;
            input_width_ = imgsz;
        } else if (line.find("conf:") != std::string::npos) {
            std::istringstream iss(line);
            std::string key, value;
            iss >> key >> value;
            conf_threshold_ = std::stof(value);
        } else if (line.find("half:") != std::string::npos) {
            std::istringstream iss(line);
            std::string key, value;
            iss >> key >> value;
            use_half_precision_ = (value == "true" || value == "True" || value == "1");
        }
    }

    file.close();
    return true;
}

void DeepInference::allocateBuffers() {
    // Calculate buffer sizes using max dimensions for dynamic models
    input_buffer_size_ = static_cast<size_t>(max_batch_size_) * max_buffer_height_ *
                         max_buffer_width_ * input_channels_;
    if (use_half_precision_) {
        input_buffer_size_ *= sizeof(__half);
    } else {
        input_buffer_size_ *= sizeof(float);
    }

    // Output buffer size depends on model type
    switch (model_type_) {
    case ModelType::YOLO:
        // YOLO output: [batch, num_detections, 6] where 6 = [x, y, w, h, conf, class]
        output_buffer_size_ = 1000 * 6 * sizeof(float); // Max 1000 detections
        break;

    case ModelType::CUSTOM:
        // Classification CNN output: [batch_size, num_classes]
        // For cone classification: 5 classes (unknown, blue, yellow, orange, large_orange)
        output_buffer_size_ = static_cast<size_t>(max_batch_size_) * 5 * sizeof(float);
        break;

    case ModelType::AUTO_DETECT:
        // Fall back to YOLO for now - future: query engine metadata
        output_buffer_size_ = 1000 * 6 * sizeof(float);
        std::cout << "Warning: AUTO_DETECT not fully implemented, using YOLO output format"
                  << std::endl;
        break;
    }

    // Allocate CUDA memory
    cudaError_t err;
    err = cudaMalloc(&input_buffer_, input_buffer_size_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate input buffer: " +
                                 std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&output_buffer_, output_buffer_size_);
    if (err != cudaSuccess) {
        cudaFree(input_buffer_);
        throw std::runtime_error("Failed to allocate output buffer: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

cv::Mat DeepInference::buildPreprocessedBGR(const cv::Mat &frame, int target_height,
                                            int target_width) const {
    cv::Mat resized;
    if (preprocess_method_ == PreprocessMethod::LETTERBOX) {
        const float scale =
            std::min(static_cast<float>(target_width) / static_cast<float>(frame.cols),
                     static_cast<float>(target_height) / static_cast<float>(frame.rows));

        const int resized_width = std::max(1, static_cast<int>(std::round(frame.cols * scale)));
        const int resized_height = std::max(1, static_cast<int>(std::round(frame.rows * scale)));

        cv::Mat scaled;
        cv::resize(frame, scaled, cv::Size(resized_width, resized_height));

        resized = cv::Mat(target_height, target_width, frame.type(), cv::Scalar(0, 0, 0));
        const int x_offset = (target_width - resized_width) / 2;
        const int y_offset = (target_height - resized_height) / 2;
        cv::Rect roi(x_offset, y_offset, resized_width, resized_height);
        scaled.copyTo(resized(roi));
    } else {
        cv::resize(frame, resized, cv::Size(target_width, target_height));
    }
    return resized;
}

cv::Mat DeepInference::get_preprocessed_debug_image(const cv::Mat &frame) const {
    if (frame.empty()) {
        return cv::Mat();
    }

    int target_height = 60;
    int target_width = 50;
    if (model_type_ != ModelType::CUSTOM) {
        target_height = frame.rows;
        target_width = frame.cols;
    }

    return buildPreprocessedBGR(frame, target_height, target_width);
}

void DeepInference::preprocessImage(const cv::Mat &frame, float *buffer, int target_height,
                                    int target_width) {
    // For classification, match training transform input size exactly.
    if (model_type_ == ModelType::CUSTOM) {
        target_height = 60;
        target_width = 50;
    } else {
        // Use actual frame size if target not specified (for dynamic models)
        if (target_height <= 0)
            target_height = frame.rows;
        if (target_width <= 0)
            target_width = frame.cols;
    }

    cv::Mat resized = buildPreprocessedBGR(frame, target_height, target_width);
    if (DEBUG_SAVE_PREPROCESSED_IMAGES) {
        save_preprocessed_debug_image(resized, preprocess_method_);
    }

    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Normalize to [0, 1]
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // Apply training-time normalization for custom classifier.
    if (model_type_ == ModelType::CUSTOM) {
        const cv::Scalar mean(0.485, 0.456, 0.406);
        const cv::Scalar std(0.229, 0.224, 0.225);
        cv::subtract(rgb, mean, rgb);
        cv::divide(rgb, std, rgb);
    }

    // Convert HWC to CHW format
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);

    // Copy to buffer (CHW format)
    int channel_size = target_height * target_width;

    for (int c = 0; c < input_channels_; ++c) {
        std::memcpy(buffer + c * channel_size, channels[c].data, channel_size * sizeof(float));
    }
}

std::vector<std::vector<float>> DeepInference::postprocessOutput(float *output_buffer,
                                                                 int num_detections) {
    float *output = output_buffer;
    std::vector<std::vector<float>> boxes;

    switch (model_type_) {
    case ModelType::YOLO: {
        // YOLO output: [batch, num_detections, 6] where 6 = [x, y, w, h, conf, class_id]
        for (int i = 0; i < num_detections; ++i) {
            float conf = output[i * 6 + 4];
            if (conf < conf_threshold_) {
                continue;
            }

            std::vector<float> box(6);
            box[0] = output[i * 6 + 0]; // x
            box[1] = output[i * 6 + 1]; // y
            box[2] = output[i * 6 + 2]; // w
            box[3] = output[i * 6 + 3]; // h
            box[4] = conf;              // confidence
            box[5] = output[i * 6 + 5]; // class_id

            boxes.push_back(box);
        }
        break;
    }

    case ModelType::CUSTOM: {
        // Classification CNN output: [batch_size, num_classes] = [1, 5]
        // Output contains class logits, need to apply softmax and find argmax

        const int num_classes = 5;

        // Apply softmax to convert logits to probabilities
        std::vector<float> probs(num_classes);
        float max_logit = *std::max_element(output, output + num_classes);
        float sum_exp = 0.0f;

        for (int i = 0; i < num_classes; ++i) {
            probs[i] = std::exp(output[i] - max_logit); // Subtract max for numerical stability
            sum_exp += probs[i];
        }

        for (int i = 0; i < num_classes; ++i) {
            probs[i] /= sum_exp;
        }

        // Find class with highest probability
        int class_id = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        float confidence = probs[class_id];

        // Only add detection if confidence exceeds threshold
        if (confidence >= conf_threshold_) {
            std::vector<float> result(6);
            // First 4 elements unused for classification (will be ignored by caller)
            result[4] = confidence;
            result[5] = static_cast<float>(class_id);
            boxes.push_back(result);
        }
        break;
    }

    case ModelType::AUTO_DETECT: {
        // Fall back to YOLO processing for now
        for (int i = 0; i < num_detections; ++i) {
            float conf = output[i * 6 + 4];
            if (conf < conf_threshold_) {
                continue;
            }

            std::vector<float> box(6);
            box[0] = output[i * 6 + 0];
            box[1] = output[i * 6 + 1];
            box[2] = output[i * 6 + 2];
            box[3] = output[i * 6 + 3];
            box[4] = conf;
            box[5] = output[i * 6 + 5];

            boxes.push_back(box);
        }
        break;
    }
    }

    return boxes;
}

std::vector<std::vector<int>>
DeepInference::convertToXYWH(const std::vector<std::vector<float>> &boxes_xyxy) {
    std::vector<std::vector<int>> boxes_xywh;

    for (const auto &box : boxes_xyxy) {
        // Convert from [x1, y1, x2, y2] to [x, y, w, h]
        int x = static_cast<int>(box[0]);
        int y = static_cast<int>(box[1]);
        int w = static_cast<int>(box[2] - box[0]);
        int h = static_cast<int>(box[3] - box[1]);

        boxes_xywh.push_back({x, y, w, h});
    }

    return boxes_xywh;
}

void DeepInference::initializeClassNames() {
    // Initialize class names - adjust based on your YOLO model
    // These are typical cone detection classes
    class_names_[0] = "unknown_cone";
    class_names_[1] = "blue_cone";
    class_names_[2] = "yellow_cone";
    class_names_[3] = "orange_cone";
    class_names_[4] = "large_orange_cone";
}

std::tuple<std::vector<std::vector<int>>, std::vector<std::string>, std::vector<float>>
DeepInference::run_inference(const cv::Mat &frame) {
    if (frame.empty()) {
        return std::make_tuple(std::vector<std::vector<int>>(), std::vector<std::string>(),
                               std::vector<float>());
    }

    // Synchronize CUDA
    cudaDeviceSynchronize();

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Use fixed input size for custom classifier, dynamic for other model types.
    int model_input_height = (model_type_ == ModelType::CUSTOM) ? 60 : frame.rows;
    int model_input_width = (model_type_ == ModelType::CUSTOM) ? 50 : frame.cols;

    // Validate dimensions don't exceed buffer capacity
    if (model_input_height > max_buffer_height_ || model_input_width > max_buffer_width_) {
        std::cerr << "Error: Frame size (" << model_input_width << "x" << model_input_height
                  << ") exceeds max buffer (" << max_buffer_width_ << "x" << max_buffer_height_
                  << ")" << std::endl;
        return std::make_tuple(std::vector<std::vector<int>>(), std::vector<std::string>(),
                               std::vector<float>());
    }

    // Preprocess image with model input dimensions
    std::vector<float> host_input_float(model_input_height * model_input_width * input_channels_);
    preprocessImage(frame, host_input_float.data(), model_input_height, model_input_width);

    // DEBUG: Dump preprocessed input tensor
    if (DEBUG_TENSORS) {
        dump_input_after_preprocess(host_input_float.data(), model_input_height, model_input_width);
    }

    // Calculate actual data size for this inference
    size_t actual_data_size = model_input_height * model_input_width * input_channels_;
    size_t actual_byte_size =
        actual_data_size * (use_half_precision_ ? sizeof(__half) : sizeof(float));

    // Convert to half precision on host if needed, then copy to GPU
    if (use_half_precision_) {
        std::vector<__half> host_input_half(actual_data_size);
        for (size_t i = 0; i < host_input_float.size(); ++i) {
            host_input_half[i] = __float2half(host_input_float[i]);
        }
        cudaMemcpy(input_buffer_, host_input_half.data(), actual_byte_size, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(input_buffer_, host_input_float.data(), actual_byte_size,
                   cudaMemcpyHostToDevice);
    }

    // Set up bindings
    void *bindings[2] = {input_buffer_, output_buffer_};

    // For models with dynamic dimensions, set the actual input shape (batch=1)
    int input_binding_idx = -1;
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (engine_->bindingIsInput(i)) {
            input_binding_idx = i;
            break;
        }
    }

    if (input_binding_idx >= 0) {
        // Set dimensions to model input size.
        nvinfer1::Dims4 input_dims(1, input_channels_, model_input_height, model_input_width);
        if (!context_->setBindingDimensions(input_binding_idx, input_dims)) {
            std::cerr << "Error: Failed to set binding dimensions to " << model_input_width << "x"
                      << model_input_height << std::endl;
        }
    }

    // Run inference
    bool success = context_->enqueueV2(bindings, nullptr, nullptr);
    if (!success) {
        std::cerr << "Error: Inference execution failed" << std::endl;
        return std::make_tuple(std::vector<std::vector<int>>(), std::vector<std::string>(),
                               std::vector<float>());
    }

    // Synchronize CUDA
    cudaDeviceSynchronize();

    // Copy output from GPU
    std::vector<float> host_output(output_buffer_size_ / sizeof(float));
    cudaMemcpy(host_output.data(), output_buffer_, output_buffer_size_, cudaMemcpyDeviceToHost);

    // DEBUG: Dump raw output from TensorRT
    if (model_type_ == ModelType::CUSTOM && DEBUG_TENSORS) {
        dump_output_after_inference(host_output.data(), 5); // 5 classes
    }

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    last_inference_time_ms_ = duration.count() / 1000.0;

    // Postprocess output based on model type
    std::vector<std::vector<int>> boxes_xywh;
    std::vector<std::string> class_names;
    std::vector<float> scores;

    if (model_type_ == ModelType::CUSTOM) {
        // Classification model - no bboxes, just class predictions
        std::vector<std::vector<float>> results = postprocessOutput(host_output.data(), 1);

        for (const auto &result : results) {
            int class_id = static_cast<int>(result[5]);
            // Return class_id as string to avoid double conversion (lidar_proc_node will parse it)
            class_names.push_back(std::to_string(class_id));
            scores.push_back(result[4]);
        }
        // boxes_xywh remains empty for classification models

        if (enable_debug_logs_) {
            std::cout << "Deep-process: " << last_inference_time_ms_
                      << " ms (class_id: "
                      << (class_names.empty() ? "none" : class_names[0]) << ")" << std::endl;
        }
    } else {
        // Detection model (YOLO) - has bboxes
        std::vector<std::vector<float>> boxes_xyxy =
            postprocessOutput(host_output.data(), 100); // Adjust num_detections

        // Convert to XYWH format
        boxes_xywh = convertToXYWH(boxes_xyxy);

        // Extract class names and scores
        for (const auto &box : boxes_xyxy) {
            int class_id = static_cast<int>(box[5]);
            class_names.push_back(class_names_.count(class_id) ? class_names_[class_id]
                                                               : "unknown_cone");
            scores.push_back(box[4]);
        }

        if (enable_debug_logs_) {
            std::cout << "Deep-process: " << last_inference_time_ms_ << " ms ("
                      << boxes_xywh.size() << " objs)" << std::endl;
        }
    }

    return std::make_tuple(boxes_xywh, class_names, scores);
}

void DeepInference::warmup(int num_runs) {
    if (num_runs <= 0) {
        return;
    }

    const int warmup_height = (model_type_ == ModelType::CUSTOM) ? 60 : input_height_;
    const int warmup_width = (model_type_ == ModelType::CUSTOM) ? 50 : input_width_;

    if (warmup_height <= 0 || warmup_width <= 0) {
        std::cerr << "Warning: Skipping model warmup due to invalid input size " << warmup_width
                  << "x" << warmup_height << std::endl;
        return;
    }

    cv::Mat warmup_frame(warmup_height, warmup_width, CV_8UC3, cv::Scalar(0, 0, 0));
    const bool previous_debug_log_state = enable_debug_logs_;
    const double previous_inference_time_ms = last_inference_time_ms_;
    enable_debug_logs_ = false;

    for (int i = 0; i < num_runs; ++i) {
        run_inference(warmup_frame);
    }

    enable_debug_logs_ = previous_debug_log_state;
    last_inference_time_ms_ = previous_inference_time_ms;
}

void DeepInference::warmupBatch(int num_batches, int batch_size) {
    if (num_batches <= 0 || batch_size <= 0) {
        return;
    }

    const int warmup_height = (model_type_ == ModelType::CUSTOM) ? 60 : input_height_;
    const int warmup_width = (model_type_ == ModelType::CUSTOM) ? 50 : input_width_;

    if (warmup_height <= 0 || warmup_width <= 0) {
        std::cerr << "Warning: Skipping batched model warmup due to invalid input size "
                  << warmup_width << "x" << warmup_height << std::endl;
        return;
    }

    cv::Mat warmup_frame(warmup_height, warmup_width, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<cv::Mat> warmup_frames(batch_size, warmup_frame);

    const bool previous_debug_log_state = enable_debug_logs_;
    const double previous_inference_time_ms = last_inference_time_ms_;
    enable_debug_logs_ = false;

    for (int i = 0; i < num_batches; ++i) {
        run_inference_batch(warmup_frames);
    }

    enable_debug_logs_ = previous_debug_log_state;
    last_inference_time_ms_ = previous_inference_time_ms;
}

std::vector<std::tuple<std::vector<std::vector<int>>, std::vector<std::string>, std::vector<float>>>
DeepInference::run_inference_batch(const std::vector<cv::Mat> &frames) {
    using InferenceResult =
        std::tuple<std::vector<std::vector<int>>, std::vector<std::string>, std::vector<float>>;

    std::vector<InferenceResult> results;
    if (frames.empty()) {
        return results;
    }

    results.resize(frames.size(), InferenceResult{{}, {}, {}});

    // Current batched path is for classifier use-case.
    // For non-classification models, fallback to per-frame execution.
    if (model_type_ != ModelType::CUSTOM) {
        for (size_t i = 0; i < frames.size(); ++i) {
            results[i] = run_inference(frames[i]);
        }
        return results;
    }

    const int model_input_height = 60;
    const int model_input_width = 50;
    const int num_classes = 5;
    const size_t single_sample_elements =
        static_cast<size_t>(model_input_height) * model_input_width * input_channels_;

    auto start = std::chrono::high_resolution_clock::now();

    size_t offset = 0;
    while (offset < frames.size()) {
        size_t chunk_size = std::min(static_cast<size_t>(max_batch_size_), frames.size() - offset);

        std::vector<float> host_input_float(chunk_size * single_sample_elements);
        for (size_t i = 0; i < chunk_size; ++i) {
            if (frames[offset + i].empty()) {
                continue;
            }
            preprocessImage(frames[offset + i],
                            host_input_float.data() + (i * single_sample_elements),
                            model_input_height, model_input_width);
        }

        size_t actual_data_size = chunk_size * single_sample_elements;
        size_t actual_byte_size =
            actual_data_size * (use_half_precision_ ? sizeof(__half) : sizeof(float));

        if (use_half_precision_) {
            std::vector<__half> host_input_half(actual_data_size);
            for (size_t i = 0; i < host_input_float.size(); ++i) {
                host_input_half[i] = __float2half(host_input_float[i]);
            }
            cudaMemcpy(input_buffer_, host_input_half.data(), actual_byte_size,
                       cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(input_buffer_, host_input_float.data(), actual_byte_size,
                       cudaMemcpyHostToDevice);
        }

        void *bindings[2] = {input_buffer_, output_buffer_};

        int input_binding_idx = -1;
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            if (engine_->bindingIsInput(i)) {
                input_binding_idx = i;
                break;
            }
        }

        if (input_binding_idx >= 0) {
            nvinfer1::Dims4 input_dims(static_cast<int>(chunk_size), input_channels_,
                                       model_input_height, model_input_width);
            if (!context_->setBindingDimensions(input_binding_idx, input_dims)) {
                std::cerr << "Error: Failed to set batch binding dimensions for batch size "
                          << chunk_size << std::endl;
                offset += chunk_size;
                continue;
            }
        }

        bool success = context_->enqueueV2(bindings, nullptr, nullptr);
        if (!success) {
            std::cerr << "Error: Batched inference execution failed" << std::endl;
            offset += chunk_size;
            continue;
        }

        cudaDeviceSynchronize();

        std::vector<float> host_output(chunk_size * num_classes);
        cudaMemcpy(host_output.data(), output_buffer_, chunk_size * num_classes * sizeof(float),
                   cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < chunk_size; ++i) {
            const float *sample_logits = host_output.data() + (i * num_classes);

            std::vector<float> probs(num_classes);
            float max_logit = *std::max_element(sample_logits, sample_logits + num_classes);
            float sum_exp = 0.0f;
            for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
                probs[class_idx] = std::exp(sample_logits[class_idx] - max_logit);
                sum_exp += probs[class_idx];
            }
            for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
                probs[class_idx] /= sum_exp;
            }

            int class_id =
                std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
            float confidence = probs[class_id];

            if (confidence >= conf_threshold_) {
                auto &class_names = std::get<1>(results[offset + i]);
                auto &scores = std::get<2>(results[offset + i]);
                class_names.push_back(std::to_string(class_id));
                scores.push_back(confidence);
            }
        }

        offset += chunk_size;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    last_inference_time_ms_ = duration.count() / 1000.0;

    if (enable_debug_logs_) {
        std::cout << "Deep-process batch: " << last_inference_time_ms_ << " ms (batch="
                  << frames.size() << ")" << std::endl;
    }

    return results;
}
