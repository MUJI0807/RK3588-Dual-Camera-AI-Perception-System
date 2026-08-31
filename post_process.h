#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <stdint.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include <map>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define OBJ_NUM_MAX_SIZE        64
#define OBJ_CLASS_NUM 80
#define LABLE_PATH "../model/coco_80_labels_list.txt"
#define BOX_NUM_SIZE (OBJ_CLASS_NUM+5)
#define MAX_OBJ_BOXS 60

#define BOX_THRESHOLD 0.5
#define NMS_THRESHOLD 0.5

using namespace std;

struct box_p
{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
};

struct detect_result_t
{
    char label[32];
    float box_conf;
    box_p box;
};

  // 定义结构体，表示检测结果组
struct detect_result_group_t
{
    int box_count;                          // 框的数量
    detect_result_t result[OBJ_NUM_MAX_SIZE];// 检测结果数组
};  


int post_process(int8_t *output0, int8_t *output1, int8_t *output2, int model_height, int model_width, float box_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales, detect_result_group_t& group);
#endif