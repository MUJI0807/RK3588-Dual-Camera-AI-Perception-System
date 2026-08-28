#pragma once

#include <opencv2/opencv.hpp>

class LaneDetector
{
public:
    LaneDetector();
    ~LaneDetector();

    bool inference(const cv::Mat &img, cv::Mat &lane_mask);
};
