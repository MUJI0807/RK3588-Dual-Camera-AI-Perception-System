#include "driving_assist.h"
#include <algorithm>
#include <cmath>

// ==================== 可调参数区 ====================

// 只对这些类别做风险评估
static bool is_vehicle_label(const std::string &name)
{
    return (name == "car" ||
            name == "truck" ||
            name == "bus" ||
            name == "vehicle");
}

// 限幅函数
static float clampf(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// 风险平滑缓存，避免每帧闪烁
static RiskScore g_smooth_score = {0.0f, 0.0f, 0.0f};
static GuidanceState g_last_guidance = GUIDE_KEEP_LANE;

// 计算当前帧风险分数
static RiskScore compute_risk_score(const cv::Mat &img, const detect_result_group_t &group)
{
    RiskScore rs;

    const int img_w = img.cols;
    const int img_h = img.rows;
    const float img_area = static_cast<float>(img_w * img_h);

    for (int i = 0; i < group.count; ++i)
    {
        const detect_result_t &det = group.results[i];

        std::string cls_name = det.name;
        if (!is_vehicle_label(cls_name))
            continue;

        int xmin = det.box.left;
        int ymin = det.box.top;
        int xmax = det.box.right;
        int ymax = det.box.bottom;

        xmin = std::max(0, xmin);
        ymin = std::max(0, ymin);
        xmax = std::min(img_w - 1, xmax);
        ymax = std::min(img_h - 1, ymax);

        int w = std::max(0, xmax - xmin);
        int h = std::max(0, ymax - ymin);
        if (w <= 0 || h <= 0)
            continue;

        float area_ratio = (w * h) / img_area;              // 框越大，风险越高
        float bottom_ratio = static_cast<float>(ymax) / img_h; // 框底越靠下，风险越高
        float cx = 0.5f * (xmin + xmax);

        // 距离中心越近，可额外增加风险
        float center_dist = std::fabs(cx - img_w * 0.5f) / (img_w * 0.5f);
        float center_bonus = 1.0f - center_dist;  // 越接近中间越大

        // 只关注画面下方区域的目标，远处小车影响减弱
        if (bottom_ratio < 0.35f)
            continue;

        // 基础风险公式：可自行调权重
        float risk = 0.0f;
        risk += 3.0f * area_ratio;      // 面积项
        risk += 1.5f * bottom_ratio;    // 垂直位置项
        risk += 0.8f * center_bonus;    // 中心偏置项

        // 若目标非常大且非常靠下，额外加权
        if (area_ratio > 0.02f && bottom_ratio > 0.65f)
            risk *= 1.3f;

        // 分配到左 / 中 / 右区域
        if (cx < img_w / 3.0f)
            rs.left += risk;
        else if (cx < img_w * 2.0f / 3.0f)
            rs.center += risk;
        else
            rs.right += risk;
    }

    return rs;
}

// 平滑，避免提示乱跳
static RiskScore smooth_risk_score(const RiskScore &cur)
{
    const float alpha = 0.30f;  // 当前帧占比
    g_smooth_score.left   = (1.0f - alpha) * g_smooth_score.left   + alpha * cur.left;
    g_smooth_score.center = (1.0f - alpha) * g_smooth_score.center + alpha * cur.center;
    g_smooth_score.right  = (1.0f - alpha) * g_smooth_score.right  + alpha * cur.right;
    return g_smooth_score;
}

// 根据风险分数做决策
static GuidanceState decide_guidance(const RiskScore &rs)
{
    const float center_danger_th = 2.6f;   // 中间区域危险阈值
    const float safe_margin = 0.8f;        // 左右安全裕量
    const float all_danger_th = 2.2f;      // 全部较危险时直接减速

    // 中间不危险，默认保持
    if (rs.center < center_danger_th)
        return GUIDE_KEEP_LANE;

    // 三个区域都不太安全，优先减速
    if (rs.left > all_danger_th && rs.center > all_danger_th && rs.right > all_danger_th)
        return GUIDE_SLOW_DOWN;

    // 中间危险，且左边更安全
    if (rs.left + safe_margin < rs.center && rs.left < rs.right)
        return GUIDE_CHANGE_LEFT;

    // 中间危险，且右边更安全
    if (rs.right + safe_margin < rs.center && rs.right < rs.left)
        return GUIDE_CHANGE_RIGHT;

    return GUIDE_SLOW_DOWN;
}

// 根据分数选择区域颜色强度
static cv::Scalar risk_to_color(float score)
{
    // 低风险：绿；中风险：黄；高风险：红
    if (score < 1.2f)
        return cv::Scalar(0, 180, 0);
    else if (score < 2.5f)
        return cv::Scalar(0, 255, 255);
    else
        return cv::Scalar(0, 0, 255);
}

// 绘制半透明区域风险图
static void draw_risk_overlay(cv::Mat &img, const RiskScore &rs)
{
    cv::Mat overlay = img.clone();

    const int w = img.cols;
    const int h = img.rows;

    // 只覆盖下半部分，更像前方道路风险区
    int top_y = static_cast<int>(h * 0.45f);
    int bottom_y = h - 1;

    cv::rectangle(overlay,
                  cv::Point(0, top_y),
                  cv::Point(w / 3, bottom_y),
                  risk_to_color(rs.left),
                  -1);

    cv::rectangle(overlay,
                  cv::Point(w / 3, top_y),
                  cv::Point(w * 2 / 3, bottom_y),
                  risk_to_color(rs.center),
                  -1);

    cv::rectangle(overlay,
                  cv::Point(w * 2 / 3, top_y),
                  cv::Point(w - 1, bottom_y),
                  risk_to_color(rs.right),
                  -1);

    // 透明叠加
    cv::addWeighted(overlay, 0.16, img, 0.84, 0.0, img);

    // 画分区线
    cv::line(img, cv::Point(w / 3, top_y), cv::Point(w / 3, bottom_y),
             cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::line(img, cv::Point(w * 2 / 3, top_y), cv::Point(w * 2 / 3, bottom_y),
             cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    // 标注分数
    char buf[64];

    snprintf(buf, sizeof(buf), "L: %.2f", rs.left);
    cv::putText(img, buf, cv::Point(20, top_y + 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255,255,255), 2, cv::LINE_AA);

    snprintf(buf, sizeof(buf), "C: %.2f", rs.center);
    cv::putText(img, buf, cv::Point(w / 3 + 20, top_y + 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255,255,255), 2, cv::LINE_AA);

    snprintf(buf, sizeof(buf), "R: %.2f", rs.right);
    cv::putText(img, buf, cv::Point(w * 2 / 3 + 20, top_y + 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255,255,255), 2, cv::LINE_AA);
}

// 绘制箭头和引导文本
static void draw_guidance_arrow(cv::Mat &img, GuidanceState state)
{
    const int w = img.cols;

    // 顶部绘制区域
    int text_y = 50;
    int arrow_y = 90;

    // 先画一个顶部黑底条，避免文字不清楚
    cv::rectangle(img,
                  cv::Point(0, 0),
                  cv::Point(w - 1, 130),
                  cv::Scalar(0, 0, 0),
                  -1);

    cv::Mat top_overlay = img.clone();
    cv::addWeighted(top_overlay, 0.15, img, 0.85, 0.0, img);

    switch (state)
    {
        case GUIDE_CHANGE_LEFT:
            cv::putText(img, "CHANGE LEFT", cv::Point(w / 2 - 140, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);

            cv::arrowedLine(img,
                            cv::Point(w / 2 + 60, arrow_y),
                            cv::Point(w / 2 - 100, arrow_y),
                            cv::Scalar(0, 255, 0), 5, cv::LINE_AA, 0, 0.25);
            break;

        case GUIDE_CHANGE_RIGHT:
            cv::putText(img, "CHANGE RIGHT", cv::Point(w / 2 - 150, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);

            cv::arrowedLine(img,
                            cv::Point(w / 2 - 60, arrow_y),
                            cv::Point(w / 2 + 100, arrow_y),
                            cv::Scalar(0, 255, 0), 5, cv::LINE_AA, 0, 0.25);
            break;

        case GUIDE_SLOW_DOWN:
            cv::putText(img, "SLOW DOWN", cv::Point(w / 2 - 120, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);

            cv::putText(img, "!!!", cv::Point(w / 2 - 30, arrow_y + 10),
                        cv::FONT_HERSHEY_SIMPLEX, 1.8, cv::Scalar(0, 0, 255), 4, cv::LINE_AA);
            break;

        case GUIDE_KEEP_LANE:
        default:
            cv::putText(img, "KEEP LANE", cv::Point(w / 2 - 110, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 0), 3, cv::LINE_AA);

            cv::arrowedLine(img,
                            cv::Point(w / 2, arrow_y + 30),
                            cv::Point(w / 2, arrow_y - 30),
                            cv::Scalar(255, 255, 0), 5, cv::LINE_AA, 0, 0.25);
            break;
    }
}

// 主接口
void draw_driving_assist(cv::Mat &img, const detect_result_group_t &group)
{
    if (img.empty())
        return;

    // 1. 计算当前帧风险
    RiskScore cur = compute_risk_score(img, group);

    // 2. 时间平滑
    RiskScore smooth = smooth_risk_score(cur);

    // 3. 决策
    GuidanceState state = decide_guidance(smooth);

    // 再做一层简单防抖：和上次差太小就保持原状态
    if (std::fabs(smooth.left - smooth.right) < 0.25f &&
        smooth.center < 2.8f)
    {
        state = GUIDE_KEEP_LANE;
    }

    g_last_guidance = state;

    // 4. 画风险图
    draw_risk_overlay(img, smooth);

    // 5. 画箭头引导
    draw_guidance_arrow(img, state);
}