#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "rknn_api.h"

class UNetLaneDetector
{
public:
    UNetLaneDetector();
    explicit UNetLaneDetector(const char* model_path);
    ~UNetLaneDetector();

    bool init(const char* model_path);
    bool is_loaded() const { return loaded_; }
    bool inference(const cv::Mat& bgr_img, cv::Mat& lane_mask);
    void release();

private:
    unsigned char* load_model(const char* model_path, int& model_size);
    bool query_model_info();
    bool postprocess_segmentation(int8_t* seg_ptr,
                                  int seg_c,
                                  int seg_h,
                                  int seg_w,
                                  int orig_w,
                                  int orig_h,
                                  cv::Mat& lane_mask);

private:
    rknn_context ctx_ = 0;
    unsigned char* model_data_ = nullptr;
    int model_size_ = 0;
    bool loaded_ = false;

    rknn_input_output_num io_num_{};
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;

    int input_w_ = 640;
    int input_h_ = 480;
    int input_c_ = 3;
    rknn_tensor_format input_fmt_ = RKNN_TENSOR_NHWC;
};
