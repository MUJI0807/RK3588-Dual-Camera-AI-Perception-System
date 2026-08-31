#include "unet_lane_detector.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace cv;

static void print_unet_tensor_attr(const char* prefix, const rknn_tensor_attr& attr)
{
    printf("%s index=%d name=%s n_dims=%d dims=[%d,%d,%d,%d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d zp=%d scale=%f\n",
           prefix, attr.index, attr.name, attr.n_dims,
           attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3],
           attr.n_elems, attr.size, attr.fmt, attr.type, attr.qnt_type,
           attr.zp, attr.scale);
}

UNetLaneDetector::UNetLaneDetector() {}
UNetLaneDetector::UNetLaneDetector(const char* model_path) { init(model_path); }
UNetLaneDetector::~UNetLaneDetector() { release(); }

unsigned char* UNetLaneDetector::load_model(const char* model_path, int& model_size)
{
    FILE* fp = fopen(model_path, "rb");
    if (!fp)
    {
        printf("[UNetLane] open model failed: %s\n", model_path);
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char* data = (unsigned char*)malloc(model_size);
    if (!data)
    {
        printf("[UNetLane] malloc model buffer failed\n");
        fclose(fp);
        return nullptr;
    }

    size_t read_size = fread(data, 1, model_size, fp);
    fclose(fp);

    if ((int)read_size != model_size)
    {
        printf("[UNetLane] read model failed, read=%zu size=%d\n", read_size, model_size);
        free(data);
        return nullptr;
    }

    return data;
}

bool UNetLaneDetector::query_model_info()
{
    int ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret != RKNN_SUCC)
    {
        printf("[UNetLane] query io num failed: %d\n", ret);
        return false;
    }

    printf("[UNetLane] input num=%d output num=%d\n", io_num_.n_input, io_num_.n_output);

    input_attrs_.resize(io_num_.n_input);
    output_attrs_.resize(io_num_.n_output);

    for (int i = 0; i < io_num_.n_input; ++i)
    {
        memset(&input_attrs_[i], 0, sizeof(rknn_tensor_attr));
        input_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) return false;
        print_unet_tensor_attr("[UNetLane][input]", input_attrs_[i]);
    }

    for (int i = 0; i < io_num_.n_output; ++i)
    {
        memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) return false;
        print_unet_tensor_attr("[UNetLane][output]", output_attrs_[i]);
    }

    if (input_attrs_.empty()) return false;

    input_fmt_ = input_attrs_[0].fmt;
    if (input_attrs_[0].fmt == RKNN_TENSOR_NCHW)
    {
        input_c_ = input_attrs_[0].dims[1];
        input_h_ = input_attrs_[0].dims[2];
        input_w_ = input_attrs_[0].dims[3];
    }
    else
    {
        input_h_ = input_attrs_[0].dims[1];
        input_w_ = input_attrs_[0].dims[2];
        input_c_ = input_attrs_[0].dims[3];
    }

    printf("[UNetLane] model input: w=%d h=%d c=%d fmt=%s\n",
           input_w_, input_h_, input_c_,
           input_fmt_ == RKNN_TENSOR_NCHW ? "NCHW" : "NHWC");
    return true;
}

bool UNetLaneDetector::init(const char* model_path)
{
    release();

    model_data_ = load_model(model_path, model_size_);
    if (!model_data_) return false;

    int ret = rknn_init(&ctx_, model_data_, model_size_, 0, nullptr);
    if (ret != RKNN_SUCC)
    {
        printf("[UNetLane] rknn_init failed: %d\n", ret);
        release();
        return false;
    }

    if (!query_model_info())
    {
        release();
        return false;
    }

    loaded_ = true;
    printf("[UNetLane] init success: %s\n", model_path);
    return true;
}

void UNetLaneDetector::release()
{
    if (ctx_)
    {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
    if (model_data_)
    {
        free(model_data_);
        model_data_ = nullptr;
    }
    model_size_ = 0;
    loaded_ = false;
    input_attrs_.clear();
    output_attrs_.clear();
    memset(&io_num_, 0, sizeof(io_num_));
}

bool UNetLaneDetector::postprocess_segmentation(int8_t* seg_ptr,
                                                int seg_c,
                                                int seg_h,
                                                int seg_w,
                                                int orig_w,
                                                int orig_h,
                                                cv::Mat& lane_mask)
{
    if (!seg_ptr || seg_c <= 1 || seg_h <= 0 || seg_w <= 0) return false;

    Mat small_mask(seg_h, seg_w, CV_8UC1, Scalar(0));

    // UNetMultiLane demo 的输出布局是 C,H,W：PtrSeg[H*W*c + W*h + w]
    for (int y = 0; y < seg_h; ++y)
    {
        uchar* out = small_mask.ptr<uchar>(y);
        for (int x = 0; x < seg_w; ++x)
        {
            int best_cls = 0;
            int best_val = (int)seg_ptr[seg_h * seg_w * 0 + seg_w * y + x];
            for (int c = 1; c < seg_c; ++c)
            {
                int v = (int)seg_ptr[seg_h * seg_w * c + seg_w * y + x];
                if (v > best_val)
                {
                    best_val = v;
                    best_cls = c;
                }
            }
            out[x] = (best_cls == 0) ? 0 : 255;
        }
    }

    resize(small_mask, lane_mask, Size(orig_w, orig_h), 0, 0, INTER_NEAREST);

    morphologyEx(lane_mask, lane_mask, MORPH_OPEN,
                 getStructuringElement(MORPH_RECT, Size(2, 2)));
    morphologyEx(lane_mask, lane_mask, MORPH_CLOSE,
                 getStructuringElement(MORPH_RECT, Size(3, 5)));
    return true;
}

bool UNetLaneDetector::inference(const cv::Mat& bgr_img, cv::Mat& lane_mask)
{
    lane_mask.release();
    if (!loaded_ || !ctx_ || bgr_img.empty()) return false;

    Mat rgb;
    cvtColor(bgr_img, rgb, COLOR_BGR2RGB);

    Mat resized;
    if (rgb.cols != input_w_ || rgb.rows != input_h_)
        resize(rgb, resized, Size(input_w_, input_h_), 0, 0, INTER_LINEAR);
    else
        resized = rgb;

    if (!resized.isContinuous()) resized = resized.clone();

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = input_w_ * input_h_ * input_c_;
    inputs[0].buf = resized.data;
    inputs[0].pass_through = 0;

    int ret = rknn_inputs_set(ctx_, 1, inputs);
    if (ret != RKNN_SUCC)
    {
        printf("[UNetLane] rknn_inputs_set failed: %d\n", ret);
        return false;
    }

    std::vector<rknn_output> outputs(io_num_.n_output);
    memset(outputs.data(), 0, sizeof(rknn_output) * io_num_.n_output);
    for (int i = 0; i < (int)io_num_.n_output; ++i)
        outputs[i].want_float = 0;

    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC)
    {
        printf("[UNetLane] rknn_run failed: %d\n", ret);
        return false;
    }

    ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
    if (ret != RKNN_SUCC)
    {
        printf("[UNetLane] rknn_outputs_get failed: %d\n", ret);
        return false;
    }

    if (io_num_.n_output < 1 || !outputs[0].buf)
    {
        rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
        return false;
    }

    int seg_c = 9;
    int seg_h = 480;
    int seg_w = 640;
    if (!output_attrs_.empty() && output_attrs_[0].n_dims >= 4)
    {
        // 该仓库 demo 为 [1, 9, 480, 640]
        if (output_attrs_[0].dims[1] > 0) seg_c = output_attrs_[0].dims[1];
        if (output_attrs_[0].dims[2] > 0) seg_h = output_attrs_[0].dims[2];
        if (output_attrs_[0].dims[3] > 0) seg_w = output_attrs_[0].dims[3];
    }

    bool ok = postprocess_segmentation((int8_t*)outputs[0].buf,
                                       seg_c, seg_h, seg_w,
                                       bgr_img.cols, bgr_img.rows,
                                       lane_mask);

    rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
    return ok && !lane_mask.empty();
}
