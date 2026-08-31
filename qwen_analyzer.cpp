// ============================================================================
//  qwen_analyzer.cpp
//  Qwen3-VL-2B 旁路风险分析器（RK3588 NPU）实现
//
//  参考 Rockchip 官方 rknn-llm 的 Multimodal_Interactive_Dialogue_Demo：
//    - 语言模型：rkllm_init + rkllm_run（多模态输入）
//    - 视觉编码：init_imgenc/run_imgenc/release_imgenc（RKNN）
//
//  【需要按实际 Qwen3-VL-2B 转换模型调整的参数】均集中在顶部常量区：
//    kVisionSize / kNImageTokens / kImageEmbedLen / 特殊 token / chat template。
//  模型转换后的视觉编码器输出 shape 决定 n_image_tokens 与 image_embed_len。
// ============================================================================

#include "qwen_analyzer.h"

#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

namespace {

// ---------- 模型相关参数 ----------
// 视觉编码输入尺寸：Qwen3-VL-2B / Qwen3.5-0.8B 均用 448x448。
// n_image_tokens / embed_size 从 RKNN 模型输出 shape【动态读取】（见 init_vision），
// 下面的常量仅作为"模型输出信息缺失"时的回退值。
constexpr int    kDefaultVisionSize    = 448;
constexpr int    kFallbackImageTokens  = 256;     // 回退：n_image_tokens
constexpr int    kFallbackEmbedLen     = 1536;    // 回退：每 token embedding 维度
constexpr int    kMaxNewTokens         = 512;     // 最大生成 token 数
constexpr int    kMaxContextLen        = 4096;    // 最大上下文长度
const char* const kImgStart           = "<|vision_start|>";
const char* const kImgEnd             = "<|vision_end|>";
const char* const kImgContent         = "<|image_pad|>";
// Chat template（Qwen 风格）。作者、系统提示、user/assistant 分隔。
const char* const kSystemPrompt       = "<|im_start|>system\n"
                                        "你是车载道路场景理解与风险分析助手。\n"
                                        "根据输入的道路图像和已检测到的目标信息，判断是否存在风险，"
                                        "并以严格的 JSON 格式输出。JSON 字段："
                                        "{\"risk_level\":\"low|medium|high\",\"risk_reasons\":[...],"
                                        "\"objects\":[...],\"actions\":[...]}.\n"
                                        "只输出 JSON，不要输出多余文字。<|im_end|>\n";
const char* const kUserPrefix          = "<|im_start|>user\n";
const char* const kAssistantPrefixJunk = "<|im_end|>\n<|im_start|>assistant\n";

// 构造给 LLM 的 user prompt（含图像占位符 <image> + 检测结果上下文）
std::string build_prompt(const std::string& detection_context)
{
    std::string prompt = "<image>请分析这幅道路场景图像，结合以下底层感知结果，判断是否存在风险：\n"
                         "<detections>\n" + detection_context + "</detections>\n"
                         "请按系统提示的 JSON 格式输出风险分析结果。";
    return prompt;
}

} // namespace

QwenAnalyzer::QwenAnalyzer() {}

QwenAnalyzer::~QwenAnalyzer()
{
    release();
}

bool QwenAnalyzer::initialize(const char* llm_model_path, const char* vision_model_path)
{
#ifdef QWEN_ENABLED
    std::lock_guard<std::mutex> lock(mutex_);
    if (ready_) {
        return true;
    }

    // ---- 1) 初始化 RKLLM（语言模型） ----
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = const_cast<char*>(llm_model_path);
    param.top_k = 1;
    param.max_new_tokens = kMaxNewTokens;
    param.max_context_len = kMaxContextLen;
    param.skip_special_token = true;
    param.img_start = const_cast<char*>(kImgStart);
    param.img_end   = const_cast<char*>(kImgEnd);
    param.img_content = const_cast<char*>(kImgContent);
    param.extend_param.base_domain_id = 1;

    if (rkllm_init(&llm_handle_, &param, &QwenAnalyzer::llm_callback) != 0) {
        std::fprintf(stderr, "[Qwen] rkllm_init failed: %s\n", llm_model_path);
        release_locked();
        return false;
    }
    std::fprintf(stderr, "[Qwen] RKLLM (语言模型) init ok\n");

    // 自定义 chat template（系统提示 + 分隔符）
    rkllm_set_chat_template(llm_handle_,
                            kSystemPrompt,
                            kUserPrefix,
                            kAssistantPrefixJunk);

    // ---- 2) 初始化视觉编码器（RKNN） ----
    if (!init_vision(vision_model_path)) {
        std::fprintf(stderr, "[Qwen] vision encoder init failed\n");
        release_locked();
        return false;
    }

    ready_ = true;
    std::fprintf(stderr, "[Qwen] QwenAnalyzer ready (vision=%d, llm=%d)\n",
                 vision_ctx_.model_width, vision_ctx_.model_height);
    return true;
#else
    (void)llm_model_path;
    (void)vision_model_path;
    return false;   // 未启用 QWEN_ENABLED：空壳
#endif
}

void QwenAnalyzer::release()
{
#ifdef QWEN_ENABLED
    std::lock_guard<std::mutex> lock(mutex_);
    release_locked();
#else
    ready_ = false;
#endif
}

#ifdef QWEN_ENABLED
// 释放资源（调用方须已持有 mutex_，用于 initialize 失败路径避免二次加锁死锁）
void QwenAnalyzer::release_locked()
{
    if (llm_handle_) {
        rkllm_destroy(llm_handle_);
        llm_handle_ = nullptr;
    }
    release_vision();
    ready_ = false;
}
#endif

bool QwenAnalyzer::analyze(const cv::Mat& frame_bgr,
                           const std::string& detection_context,
                           std::string& out_json,
                           int timeout_ms)
{
#ifdef QWEN_ENABLED
    if (!ready_ || frame_bgr.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);   // 串行化，RKLLM 单实例不可并发

    // ---- 预处理：BGR -> RGB -> 正方形 -> resize ----
    cv::Mat rgb;
    cv::cvtColor(frame_bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat square;
    expand_to_square(rgb, square, cv::Scalar(127.5, 127.5, 127.5));
    int vs = vision_ctx_.model_width > 0 ? vision_ctx_.model_width : kDefaultVisionSize;
    cv::Mat resized;
    cv::resize(square, resized, cv::Size(vs, vs), 0, 0, cv::INTER_LINEAR);

    // ---- 运行视觉编码器，得到图像 embedding ----
    std::vector<float> image_embed;
    if (!run_vision(resized, image_embed)) {
        out_json.clear();
        return false;
    }
    const int n_image_tokens = vision_ctx_.model_image_token;
    const int embed_size = vision_ctx_.model_embed_size;
    if ((int)image_embed.size() < n_image_tokens * embed_size) {
        std::fprintf(stderr, "[Qwen] image_embed too small: %zu (expect %d)\n",
                     image_embed.size(), n_image_tokens * embed_size);
        out_json.clear();
        return false;
    }
    float* img_vec = image_embed.data();

    // ---- 构造多模态输入 ----
    std::string user_prompt = build_prompt(detection_context);
    RKLLMInput rkllm_input;
    std::memset(&rkllm_input, 0, sizeof(rkllm_input));
    rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
#ifdef QWEN_RKLLM_130
    // RKLLM 1.3.0 新版多模态结构（Qwen3.5-0.8B 等新模型）
    rkllm_input.multimodal_input.prompt = const_cast<char*>(user_prompt.c_str());
    rkllm_input.multimodal_input.image.image_embed = img_vec;
    rkllm_input.multimodal_input.image.n_image_tokens = n_image_tokens;
    rkllm_input.multimodal_input.image.n_image = 1;
    rkllm_input.multimodal_input.image.image_height = vision_ctx_.model_height;
    rkllm_input.multimodal_input.image.image_width = vision_ctx_.model_width;
    rkllm_input.multimodal_input.image.image_start = const_cast<char*>(kImgStart);
    rkllm_input.multimodal_input.image.image_end = const_cast<char*>(kImgEnd);
    rkllm_input.multimodal_input.image.image_content = const_cast<char*>(kImgContent);
#else
    // RKLLM 1.2.3 旧版多模态结构（Qwen2-VL / Qwen3-VL-2B 官方 demo）
    rkllm_input.multimodal_input.prompt = const_cast<char*>(user_prompt.c_str());
    rkllm_input.multimodal_input.image_embed = img_vec;
    rkllm_input.multimodal_input.n_image_tokens = n_image_tokens;
#endif

    RKLLMInferParam infer_params;
    std::memset(&infer_params, 0, sizeof(infer_params));
    infer_params.mode = RKLLM_INFER_GENERATE;
    infer_params.keep_history = 0;   // 风险分析无多轮需求，每帧独立

    // ---- 触发推理，等待回调收集结果 ----
    {
        std::lock_guard<std::mutex> rl(result_mutex_);
        collected_text_.clear();
        finished_ = false;
        has_error_ = false;
    }
    if (rkllm_run(llm_handle_, &rkllm_input, &infer_params, this) != 0) {
        std::fprintf(stderr, "[Qwen] rkllm_run failed\n");
        out_json.clear();
        return false;
    }

    // ---- 等待 RUN_FINISH / RUN_ERROR / 超时 ----
    std::unique_lock<std::mutex> rl(result_mutex_);
    if (!result_cv_.wait_for(rl, std::chrono::milliseconds(timeout_ms),
                             [this] { return finished_ || has_error_; })) {
        std::fprintf(stderr, "[Qwen] inference timeout (%d ms)\n", timeout_ms);
        out_json.clear();
        return false;
    }
    out_json = collected_text_;
    rl.unlock();
    return !out_json.empty() && !has_error_;
#else
    (void)frame_bgr;
    (void)detection_context;
    (void)out_json;
    (void)timeout_ms;
    return false;
#endif
}

#ifdef QWEN_ENABLED
// 静态回调：RKLLM 框架通过 userdata 传入 this，转发到实例方法。
void QwenAnalyzer::llm_callback(RKLLMResult* result, void* userdata, LLMCallState state)
{
    QwenAnalyzer* self = static_cast<QwenAnalyzer*>(userdata);
    if (self) {
        self->on_llm_result(result, state);
    }
}

// 实例方法：RKLLM 流向式产出 token，这里累积文本，RUN_FINISH/ERROR 时唤醒等待者。
void QwenAnalyzer::on_llm_result(RKLLMResult* result, LLMCallState state)
{
    if (state == RKLLM_RUN_NORMAL && result && result->text) {
        std::lock_guard<std::mutex> lock(result_mutex_);
        collected_text_ += result->text;
    } else if (state == RKLLM_RUN_FINISH) {
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            finished_ = true;
        }
        result_cv_.notify_all();
    } else if (state == RKLLM_RUN_ERROR) {
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            has_error_ = true;
        }
        result_cv_.notify_all();
    }
}

// ---- 视觉编码器（RKNN）实现，参考官方 image_enc ----
static int read_data_from_file(const char* path, char** out_data)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);
    char* data = (char*)malloc(size + 1);
    if (!data) { fclose(fp); return -1; }
    data[size] = 0;
    if ((long)fread(data, 1, size, fp) != size) { free(data); fclose(fp); return -1; }
    fclose(fp);
    *out_data = data;
    return (int)size;
}

bool QwenAnalyzer::init_vision(const char* model_path)
{
    char* model = nullptr;
    int model_len = read_data_from_file(model_path, &model);
    if (model_len <= 0 || !model) {
        std::fprintf(stderr, "[Qwen] load vision model fail: %s\n", model_path);
        return false;
    }
    int ret = rknn_init(&vision_ctx_.rknn_ctx, model, model_len, 0, nullptr);
    free(model);
    if (ret < 0) {
        std::fprintf(stderr, "[Qwen] rknn_init fail: %d\n", ret);
        return false;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(vision_ctx_.rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) { release_vision(); return false; }
    vision_ctx_.io_num = io_num;

    if (io_num.n_input < 1) { release_vision(); return false; }
    vision_ctx_.input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    vision_ctx_.output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    std::memset(vision_ctx_.input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    std::memset(vision_ctx_.output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));

    for (int i = 0; i < io_num.n_input; ++i) {
        vision_ctx_.input_attrs[i].index = i;
        rknn_query(vision_ctx_.rknn_ctx, RKNN_QUERY_INPUT_ATTR, &vision_ctx_.input_attrs[i], sizeof(rknn_tensor_attr));
    }
    for (int i = 0; i < io_num.n_output; ++i) {
        vision_ctx_.output_attrs[i].index = i;
        rknn_query(vision_ctx_.rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &vision_ctx_.output_attrs[i], sizeof(rknn_tensor_attr));
    }

    if (vision_ctx_.input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        vision_ctx_.model_channel = vision_ctx_.input_attrs[0].dims[1];
        vision_ctx_.model_height  = vision_ctx_.input_attrs[0].dims[2];
        vision_ctx_.model_width   = vision_ctx_.input_attrs[0].dims[3];
    } else {
        vision_ctx_.model_height  = vision_ctx_.input_attrs[0].dims[1];
        vision_ctx_.model_width   = vision_ctx_.input_attrs[0].dims[2];
        vision_ctx_.model_channel = vision_ctx_.input_attrs[0].dims[3];
    }

    // 动态读取 image token 数与 embedding 维度：
    // 视觉编码器输出 shape 通常为 [1, n_image_tokens, embed_size]，
    // 取第一个 >1 的维度为 token 数，其后一维为 embedding 维度。
    vision_ctx_.model_image_token = 0;
    vision_ctx_.model_embed_size  = 0;
    if (io_num.n_output >= 1) {
        for (int i = 0; i < vision_ctx_.output_attrs[0].n_dims - 1; ++i) {
            if (vision_ctx_.output_attrs[0].dims[i] > 1) {
                vision_ctx_.model_image_token = vision_ctx_.output_attrs[0].dims[i];
                vision_ctx_.model_embed_size  = vision_ctx_.output_attrs[0].dims[i + 1];
                break;
            }
        }
    }
    if (vision_ctx_.model_image_token <= 0) vision_ctx_.model_image_token = kFallbackImageTokens;
    if (vision_ctx_.model_embed_size  <= 0) vision_ctx_.model_embed_size  = kFallbackEmbedLen;

    vision_loaded_ = true;
    std::fprintf(stderr, "[Qwen] vision input %dx%dx%d, image_token=%d embed=%d\n",
                 vision_ctx_.model_height, vision_ctx_.model_width, vision_ctx_.model_channel,
                 vision_ctx_.model_image_token, vision_ctx_.model_embed_size);
    return true;
}

void QwenAnalyzer::release_vision()
{
    if (!vision_loaded_) return;
    if (vision_ctx_.input_attrs) { free(vision_ctx_.input_attrs); vision_ctx_.input_attrs = nullptr; }
    if (vision_ctx_.output_attrs) { free(vision_ctx_.output_attrs); vision_ctx_.output_attrs = nullptr; }
    if (vision_ctx_.rknn_ctx != 0) { rknn_destroy(vision_ctx_.rknn_ctx); vision_ctx_.rknn_ctx = 0; }
    vision_loaded_ = false;
}

bool QwenAnalyzer::run_vision(const cv::Mat& rgb, std::vector<float>& image_embed)
{
    if (!vision_loaded_ || rgb.empty()) return false;
    if (!rgb.isContinuous()) return false;   // resize 后应连续；此处防御

    rknn_input inputs[1];
    rknn_output outputs[1];
    std::memset(inputs, 0, sizeof(inputs));
    std::memset(outputs, 0, sizeof(outputs));

    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = (size_t)vision_ctx_.model_width * vision_ctx_.model_height * vision_ctx_.model_channel;
    inputs[0].buf   = const_cast<uchar*>(rgb.data);

    if (rknn_inputs_set(vision_ctx_.rknn_ctx, 1, inputs) < 0) return false;
    if (rknn_run(vision_ctx_.rknn_ctx, nullptr) < 0) return false;

    outputs[0].want_float = 1;
    if (rknn_outputs_get(vision_ctx_.rknn_ctx, 1, outputs, nullptr) < 0) return false;

    size_t n_float = outputs[0].size / sizeof(float);
    image_embed.resize(n_float);
    std::memcpy(image_embed.data(), outputs[0].buf, outputs[0].size);
    rknn_outputs_release(vision_ctx_.rknn_ctx, 1, outputs);
    return true;
}

// 将图像扩为正方形并填充背景色（与官方 modeling_minicpmv.py 一致）
void QwenAnalyzer::expand_to_square(const cv::Mat& src, cv::Mat& dst, const cv::Scalar& bg)
{
    if (src.cols == src.rows) { dst = src.clone(); return; }
    int size = std::max(src.cols, src.rows);
    dst = cv::Mat(size, size, src.type(), bg);
    int x = (size - src.cols) / 2;
    int y = (size - src.rows) / 2;
    src.copyTo(dst(cv::Rect(x, y, src.cols, src.rows)));
}

#endif // QWEN_ENABLED
