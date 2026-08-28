#ifndef YOLOV5S_H
#define YOLOV5S_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string.h>
#include <vector>
#include "3rdparty/librknn_api/include/rknn_api.h"

#include "RgaUtils.h"
#include "im2d.h"
#include "rga.h"

#include "post_process.h"
#include "lane_detector.h"

using namespace std;
using namespace cv;

class Yolov5s
{
private:
    rknn_context context;
    unsigned int model_size;
    rknn_tensor_attr input_tensor;
    rknn_tensor_attr output_tensor;
    rknn_input_output_num num_tensors;
    vector<rknn_tensor_attr> input_attrs;
    vector<rknn_tensor_attr> output_attrs;
    unsigned char *model_data;
    unsigned char * load_model(const char* model_path, unsigned int &model_size);

public:
    Yolov5s(const char* model_path, int npu_index);
    ~Yolov5s();

    int model_height;
    int model_width;
    int model_channel;
    int img_height;
    int img_width;
    int img_channel;

    int inference_image(Mat &origin_img, detect_result_group_t &result_group);
    int draw_result(cv::Mat &orig_img, detect_result_group_t &group);
};

#endif
