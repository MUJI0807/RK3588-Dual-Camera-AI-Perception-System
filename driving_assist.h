#ifndef DRIVING_ASSIST_H
#define DRIVING_ASSIST_H

#include <opencv2/opencv.hpp>
#include <string>

// 你这里按自己工程里的检测结果结构体头文件来 include
#include "post_process.h"

// 左中右风险分数
struct RiskScore
{
    float left   = 0.0f;
    float center = 0.0f;
    float right  = 0.0f;
};

// 引导状态
enum GuidanceState
{
    GUIDE_KEEP_LANE = 0,
    GUIDE_CHANGE_LEFT,
    GUIDE_CHANGE_RIGHT,
    GUIDE_SLOW_DOWN
};

// 主接口：输入原图和检测结果，直接在图上叠加“区域风险图 + 箭头引导”
void draw_driving_assist(cv::Mat &img, const detect_result_group_t &group);

#endif