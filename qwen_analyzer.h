#pragma once

// ============================================================================
//  qwen_analyzer.h
//  Qwen3-VL-2B 旁路风险分析器（RK3588 NPU）
//
//  定位：在现有"双摄yolov5s+UNet"实时管线之外，作为独立的低频语义理解层。
//  - 不改变现有 YOLO/UNet/编码/推流管线（实时部分保持 30fps 不变）
//  - 只对"事件驱动/定时"抽出的关键帧做 Qwen3-VL 多模态推理
//  - 输入：去畸变后的单帧 BGR 图 + 底层检测结果的文本化上下文
//  - 输出：结构化 JSON（风险等级/原因/建议），由调用方决定如何消费
//
//  【重要】需要板端提供如下库与头文件：
//    - librkllmrt.so + rkllm.h        （Rockchip RKLLM 运行时，语言模型 W8A8）
//    - librknnrt.so + rknn_api.h      （视觉编码器 RKNN）
//    模型文件（需提前转换好，见 README_Qwen.md）：
//    - Qwen3-VL-2B_llm_w8a8_rk3588.rkllm   （语言模型）
//    - Qwen3-VL-2B_vision_rk3588.rknn      （视觉编码器）
//
//  【编译开关】只有定义 QWEN_ENABLED 时才会启用真实推理；未定义时本类是
//  空壳（initialize 返回 false，analyze 返回 false），保证现有管线不受影响。
//  CMake 中通过 find_path/find_library 检测到 RKLLM 后自动定义该宏。
// ============================================================================

#include <opencv2/core.hpp>
#include <string>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <vector>
#include <chrono>

#ifdef QWEN_ENABLED
#include "rkllm.h"
#include "rknn_api.h"
#endif

// 视觉编码器 RKNN 上下文（参考官方 image_enc 结构）
#ifdef QWEN_ENABLED
typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_width;
    int model_height;
    int model_channel;
} QwenVisionContext;
#endif

class QwenAnalyzer {
public:
    QwenAnalyzer();
    ~QwenAnalyzer();

    // 初始化：加载 RKLLM（语言模型）+ RKNN（视觉编码器）。
    // 返回 true 表示就绪。失败时不抛异常，返回 false。
    bool initialize(const char* llm_model_path, const char* vision_model_path);

    // 释放所有资源（幂等）
    void release();

#ifdef QWEN_ENABLED
    // 释放资源（调用方须已持有 mutex_）
    void release_locked();
#endif

    // 是否已就绪
    bool is_ready() const { return ready_; }

    // 同步分析一帧：输入 BGR 帧 + 检测上下文文本，输出 LLM 生成的 JSON 文本。
    // timeout_ms：等待结果的最长时间（LLM 解码较慢，默认 8s）。
    // 返回 true 表示成功产出结果（out_json 可能仍是模型的原始文本，需下游按需解析）。
    bool analyze(const cv::Mat& frame_bgr,
                 const std::string& detection_context,
                 std::string& out_json,
                 int timeout_ms = 8000);

private:
    // ---- RKLLM / RKNN 内部实现（仅在 QWEN_ENABLED 下存在） ----
#ifdef QWEN_ENABLED
    // RKLLM 回调（static，通过 userdata 拿到 this 再转发到实例方法）
    static void llm_callback(RKLLMResult* result, void* userdata, LLMCallState state);
    // 实例方法：在回调线程中收集生成文本并唤醒等待者
    void on_llm_result(RKLLMResult* result, LLMCallState state);

    bool init_vision(const char* model_path);
    void release_vision();
    bool run_vision(const cv::Mat& rgb, std::vector<float>& image_embed);
    void expand_to_square(const cv::Mat& src, cv::Mat& dst, const cv::Scalar& bg);
#endif

    bool ready_ = false;
    std::mutex mutex_;   // 串行化整个推理流程（RKLLM 单实例不可并发）

#ifdef QWEN_ENABLED
    LLMHandle llm_handle_ = nullptr;
    QwenVisionContext vision_ctx_{};
    bool vision_loaded_ = false;

    // LLM 结果收集（在回调中写入，analyze 阻塞等待直到 RUN_FINISH/超时）
    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    std::string collected_text_;
    bool finished_ = false;
    bool has_error_ = false;
#endif
};
