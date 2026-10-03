// Reconstructed header for the car's deep_inference.cpp (original .hpp was not provided).
// Members/types inferred from their use in deep_inference.cpp.
#pragma once

#include <NvInfer.h>
#include <cstring>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>

enum class ModelType { YOLO, CUSTOM, AUTO_DETECT };
enum class PreprocessMethod { LETTERBOX, TRANSFORM };

class DeepInference {
  public:
    DeepInference(const std::string &model_path, const std::string &model_config_path,
                  ModelType model_type, PreprocessMethod preprocess_method);
    ~DeepInference();

    std::tuple<std::vector<std::vector<int>>, std::vector<std::string>, std::vector<float>>
    run_inference(const cv::Mat &frame);
    std::vector<std::tuple<std::vector<std::vector<int>>, std::vector<std::string>, std::vector<float>>>
    run_inference_batch(const std::vector<cv::Mat> &frames);
    void warmup(int num_runs);
    void warmupBatch(int num_batches, int batch_size);
    cv::Mat get_preprocessed_debug_image(const cv::Mat &frame) const;

    double last_inference_time_ms() const { return last_inference_time_ms_; }
    void set_debug_logs(bool on) { enable_debug_logs_ = on; }

  private:
    bool loadEngine(const std::string &engine_path);
    bool loadConfig(const std::string &config_path);
    void allocateBuffers();
    cv::Mat buildPreprocessedBGR(const cv::Mat &frame, int target_height, int target_width) const;
    void preprocessImage(const cv::Mat &frame, float *buffer, int target_height, int target_width);
    std::vector<std::vector<float>> postprocessOutput(float *output_buffer, int num_detections);
    std::vector<std::vector<int>> convertToXYWH(const std::vector<std::vector<float>> &boxes_xyxy);
    void initializeClassNames();

    ModelType model_type_;
    int input_height_;
    int input_width_;
    int input_channels_;
    int max_batch_size_;
    int max_buffer_height_;
    int max_buffer_width_;
    float conf_threshold_;
    bool use_half_precision_;
    PreprocessMethod preprocess_method_;
    double last_inference_time_ms_;
    bool enable_debug_logs_;
    void *input_buffer_;
    void *output_buffer_;
    size_t input_buffer_size_;
    size_t output_buffer_size_;

    // Declaration order matters: destroyed in reverse (context, engine, runtime).
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::map<int, std::string> class_names_;
};
