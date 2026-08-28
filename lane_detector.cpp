#include "lane_detector.h"

using namespace cv;

LaneDetector::LaneDetector()
{
}

LaneDetector::~LaneDetector()
{
}

bool LaneDetector::inference(const cv::Mat &img, cv::Mat &lane_mask)
{
    if (img.empty()) return false;

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    // 简单白线 mask（先验证整体流程）
    Mat white_mask;
    inRange(hsv,
            Scalar(0, 0, 180),
            Scalar(180, 60, 255),
            white_mask);

    // 只保留近场
    Mat roi_mask = Mat::zeros(img.size(), CV_8UC1);

    std::vector<Point> roi = {
        Point(img.cols * 0.15, img.rows),
        Point(img.cols * 0.40, img.rows * 0.65),
        Point(img.cols * 0.60, img.rows * 0.65),
        Point(img.cols * 0.85, img.rows)
    };

    fillConvexPoly(roi_mask, roi, 255);

    lane_mask = white_mask & roi_mask;

    morphologyEx(lane_mask, lane_mask,
                 MORPH_OPEN,
                 getStructuringElement(MORPH_RECT, Size(3,3)));

    morphologyEx(lane_mask, lane_mask,
                 MORPH_CLOSE,
                 getStructuringElement(MORPH_RECT, Size(5,5)));

    return true;
}
