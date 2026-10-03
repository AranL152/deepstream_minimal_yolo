// Benchmark the car's DeepInference (unmodified) on a video file.
//
// Per frame: decode (OpenCV/FFmpeg, CPU) -> letterbox to engine size (CPU, the caller's job:
// the YOLO path uses frame.rows/cols as the network input size) -> run_inference().
// Reports per-stage latency and end-to-end throughput.
//
// usage: bench_car <engine> <video> [max_frames]

#include "deep_inference.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static void report(const char *name, std::vector<double> v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    auto pct = [&](double p) { return v[std::min(v.size() - 1, (size_t)(p * v.size()))]; };
    std::printf("  %-22s mean %7.3f  p50 %7.3f  p99 %7.3f  max %7.3f ms\n", name, mean, pct(0.5),
                pct(0.99), v.back());
}

static cv::Mat letterbox(const cv::Mat &src, int w, int h) {
    float s = std::min((float)w / src.cols, (float)h / src.rows);
    int rw = (int)std::round(src.cols * s), rh = (int)std::round(src.rows * s);
    cv::Mat scaled, out(h, w, src.type(), cv::Scalar(114, 114, 114));
    cv::resize(src, scaled, cv::Size(rw, rh));
    scaled.copyTo(out(cv::Rect((w - rw) / 2, (h - rh) / 2, rw, rh)));
    return out;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <engine> <video> [max_frames]\n", argv[0]);
        return 1;
    }
    const int max_frames = argc > 3 ? std::atoi(argv[3]) : 1 << 30;
    const int net = 640;

    DeepInference inf(argv[1], "", ModelType::YOLO, PreprocessMethod::LETTERBOX);
    inf.set_debug_logs(false);
    inf.warmup(50);

    cv::VideoCapture cap(argv[2]);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "cannot open %s\n", argv[2]);
        return 1;
    }

    std::vector<double> t_decode, t_letterbox, t_infer_call, t_infer_internal, t_total;
    size_t n_dets = 0;
    cv::Mat frame;
    cv::Size src;
    auto t_start = Clock::now();
    for (int i = 0; i < max_frames; ++i) {
        auto t0 = Clock::now();
        if (!cap.read(frame)) break;
        t_decode.push_back(ms_since(t0));
        src = frame.size();

        auto t1 = Clock::now();
        cv::Mat in = letterbox(frame, net, net);
        t_letterbox.push_back(ms_since(t1));

        auto t2 = Clock::now();
        auto res = inf.run_inference(in);
        t_infer_call.push_back(ms_since(t2));
        t_infer_internal.push_back(inf.last_inference_time_ms());
        n_dets += std::get<0>(res).size();

        t_total.push_back(ms_since(t0));
    }
    double wall = ms_since(t_start);
    size_t n = t_total.size();

    std::printf("frames %zu  (%dx%d source)  avg dets/frame %.1f\n", n, src.width, src.height,
                n ? (double)n_dets / n : 0.0);
    report("decode (CPU)", t_decode);
    report("letterbox (CPU)", t_letterbox);
    report("run_inference()", t_infer_call);
    report("  internal timer", t_infer_internal);
    report("total per frame", t_total);
    std::printf("  end-to-end throughput  %.1f FPS\n", n * 1000.0 / wall);
    std::printf("  inference-only (letterbox+run_inference) %.1f FPS\n",
                1000.0 / ((std::accumulate(t_letterbox.begin(), t_letterbox.end(), 0.0) +
                           std::accumulate(t_infer_call.begin(), t_infer_call.end(), 0.0)) / n));
    return 0;
}
