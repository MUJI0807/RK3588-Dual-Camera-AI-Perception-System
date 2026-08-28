#include "yolov5s.h"
#include "post_process.h"
#include "lane_detector.h"
#include "unet_lane_detector.h"

#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <vector>
#include <deque>
#include <string>
#include <cctype>


using namespace std;
using namespace cv;

static LaneDetector g_lane_detector;

// UNetMultiLane RKNN 车道线分割模块。
// 说明：
// 1) 优先尝试加载 ../model/lane.rknn；
// 2) 如果失败，再尝试 ./model/lane.rknn；
// 3) 如果模型不存在或推理失败，则自动回退到原来的传统 CV lane_mask。
static UNetLaneDetector g_unet_lane;
static bool g_unet_lane_try_init = false;

static void print_tensor_attr(rknn_tensor_attr *attr)
{
    string shape_str = attr->n_dims < 1 ? "" : to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; i++)
        shape_str += "," + to_string(attr->dims[i]);
}

// ============================================================================
//  RK3588 端侧驾驶辅助绘制模块：近端高权重 BEV + Sliding Window + Polynomial Fit 版本
// ----------------------------------------------------------------------------
//  这一版放弃 “Hough 直线拟合 + 固定梯形” 的主路线，改为：
//  1) YOLO 负责目标检测，并把车辆框从车道线 mask 中扣除；
//  2) OpenCV 提取白/黄车道线像素，而不是提取直线；
//  3) 通过 IPM 逆透视变换得到 BEV 鸟瞰图；
//  4) 在 BEV 中用滑动窗口搜索左右车道线像素；
//  5) 用二次多项式 x = ay^2 + by + c 拟合车道曲线；
//  6) 将拟合出的车道区域反投影回原图，生成更贴地的光毯。
// ----------------------------------------------------------------------------
//  这样做的核心原因：
//  - 弯道不是直线，HoughLinesP 天然会失真；
//  - 白色前车/电线杆虽然像白色直线，但它们不应该参与“路面车道像素”拟合；
//  - BEV 中车道线更接近平行/缓弯曲线，更适合用滑动窗口和多项式拟合。
// ============================================================================

enum GuidanceState { GUIDE_KEEP_LANE = 0, GUIDE_SLOW_DOWN };
enum LaneZoneType  { LANE_OUTSIDE = -1, LANE_LEFT = 0, LANE_CURRENT = 1, LANE_RIGHT = 2 };

struct MultiLaneRisk
{
    float left = 0.f;
    float current = 0.f;
    float right = 0.f;
};

struct QuadCurve
{
    bool valid = false;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    int pixel_count = 0;
    double confidence = 0.0;
};

struct BevLaneModel
{
    bool valid = false;
    bool from_fallback = true;
    float confidence = 0.0f;

    int img_w = 0;
    int img_h = 0;
    int bev_w = 0;
    int bev_h = 0;

    int top_y_img = 0;
    int bottom_y_img = 0;

    Mat M;     // 原图 -> BEV
    Mat Minv;  // BEV -> 原图

    QuadCurve left;
    QuadCurve right;

    // BEV 中当前车道宽度，单位像素。
    double lane_width_bev = 0.0;

    // 反投影后的三车道区域。
    vector<Point> left_poly;
    vector<Point> current_poly;
    vector<Point> right_poly;

    // 调试：反投影后的左右曲线点。
    vector<Point> left_curve_img;
    vector<Point> right_curve_img;

    // BEV 调试信息。
    int left_base_x = -1;
    int right_base_x = -1;
};

static BevLaneModel g_last_bev_lane;
static QuadCurve g_smooth_left_curve;
static QuadCurve g_smooth_right_curve;
static bool g_has_smooth_curve = false;
static float g_smooth_risk = 0.0f;
static GuidanceState g_last_stable = GUIDE_KEEP_LANE;
static GuidanceState g_candidate   = GUIDE_KEEP_LANE;
static int g_cand_count = 0;

// ============================================================================
//  基础工具
// ============================================================================
static float clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

static double clampd(double x, double lo, double hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

static bool point_in_polygon(const vector<Point> &poly, const Point2f &p)
{
    if (poly.size() < 3) return false;
    return pointPolygonTest(poly, p, false) >= 0.0;
}

static bool is_vehicle_label(const char *label)
{
    if (!label) return false;
    string name(label);
    transform(name.begin(), name.end(), name.begin(),
              [](unsigned char c){ return (char)tolower(c); });

    return name.find("car")     != string::npos ||
           name.find("bus")     != string::npos ||
           name.find("truck")   != string::npos ||
           name.find("van")     != string::npos ||
           name.find("motor")   != string::npos ||
           name.find("bike")    != string::npos ||
           name.find("bicycle") != string::npos;
}

static double curve_x_at_y(const QuadCurve &q, double y)
{
    return q.a * y * y + q.b * y + q.c;
}

static QuadCurve offset_curve(const QuadCurve &q, double offset_x)
{
    QuadCurve out = q;
    out.c += offset_x;
    return out;
}

static QuadCurve smooth_curve(const QuadCurve &old_q, const QuadCurve &new_q, double alpha)
{
    if (!old_q.valid) return new_q;
    if (!new_q.valid) return old_q;

    QuadCurve out;
    out.valid = true;
    out.a = alpha * new_q.a + (1.0 - alpha) * old_q.a;
    out.b = alpha * new_q.b + (1.0 - alpha) * old_q.b;
    out.c = alpha * new_q.c + (1.0 - alpha) * old_q.c;
    out.pixel_count = new_q.pixel_count;
    out.confidence = alpha * new_q.confidence + (1.0 - alpha) * old_q.confidence;
    return out;
}

// ============================================================================
//  车辆遮罩：白色车辆最容易污染车道线 mask，先从 lane mask 中扣掉。
// ============================================================================
static Mat build_vehicle_mask(const Mat &img, detect_result_group_t &result_group)
{
    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    if (img.empty()) return mask;

    const int iw = img.cols;
    const int ih = img.rows;

    for (int i = 0; i < result_group.box_count; ++i)
    {
        if (result_group.result[i].box_conf < 0.30f) continue;
        if (!is_vehicle_label(result_group.result[i].label)) continue;

        int xmin = max(0,    result_group.result[i].box.xmin);
        int ymin = max(0,    result_group.result[i].box.ymin);
        int xmax = min(iw-1, result_group.result[i].box.xmax);
        int ymax = min(ih-1, result_group.result[i].box.ymax);

        int bw = xmax - xmin;
        int bh = ymax - ymin;
        if (bw <= 0 || bh <= 0) continue;

        // 横向和纵向都适当膨胀，避免车身边缘/车牌/高亮反光进入车道线像素。
        // 短期优化重点：白色前车对 lane mask 污染很严重，遮罩宁可略大。
        // 横向多扩一点，底部也多扩一点，避免车身边缘、车牌和反光区域进入车道线拟合。
        int ex = (int)(bw * 0.28f);
        int ey_top = (int)(bh * 0.14f);
        int ey_bot = (int)(bh * 0.28f);

        Rect r(max(0, xmin - ex),
               max(0, ymin - ey_top),
               min(iw - 1, xmax + ex) - max(0, xmin - ex) + 1,
               min(ih - 1, ymax + ey_bot) - max(0, ymin - ey_top) + 1);

        rectangle(mask, r, Scalar(255), FILLED);
    }

    return mask;
}

// ============================================================================
//  路面颜色判断：车道线应该“长在灰色/暗灰色路面上”，而不是天空/草地/白车上。
// ============================================================================
static bool is_gray_road_pixel_bgr(const Vec3b &bgr)
{
    float b = (float)bgr[0];
    float g = (float)bgr[1];
    float r = (float)bgr[2];
    float brightness = (b + g + r) / 3.0f;
    float max_diff = max({fabs(b - g), fabs(g - r), fabs(b - r)});

    // 低饱和度 + 中等亮度：粗略覆盖沥青路面。
    return brightness >= 35.f && brightness <= 210.f && max_diff <= 55.f;
}

static Mat build_road_mask(const Mat &img)
{
    Mat road = Mat::zeros(img.size(), CV_8UC1);
    if (img.empty()) return road;

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    for (int y = 0; y < img.rows; ++y)
    {
        const Vec3b *bgr_row = img.ptr<Vec3b>(y);
        const Vec3b *hsv_row = hsv.ptr<Vec3b>(y);
        uchar *out = road.ptr<uchar>(y);

        for (int x = 0; x < img.cols; ++x)
        {
            const Vec3b &p = bgr_row[x];
            int H = hsv_row[x][0];
            int S = hsv_row[x][1];
            int V = hsv_row[x][2];

            bool gray_road = is_gray_road_pixel_bgr(p);

            // 排除明显绿色植被、蓝天、黄色路牌。
            bool green = (H >= 35 && H <= 95 && S > 45 && V > 45);
            bool sky   = (H >= 85 && H <= 125 && S > 35 && V > 80);
            bool yellow_sign = (H >= 15 && H <= 45 && S > 80 && V > 100);

            if (gray_road && !green && !sky && !yellow_sign)
                out[x] = 255;
        }
    }

    // 稍微闭运算，让路面区域更连续。
    morphologyEx(road, road, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(9, 9)));
    morphologyEx(road, road, MORPH_OPEN,  getStructuringElement(MORPH_RECT, Size(5, 5)));
    return road;
}

// ============================================================================
//  构建车道线像素 mask
//  注意：这一版不是为了找“白色直线”，而是找“在路面上的白/黄车道线像素”。
// ============================================================================
static Mat build_lane_pixel_mask(const Mat &img,
                                 const Mat &vehicle_mask,
                                 Mat &road_mask_out,
                                 vector<Point> &roi_poly_out)
{
    Mat lane_mask = Mat::zeros(img.size(), CV_8UC1);
    if (img.empty()) return lane_mask;

    const int w = img.cols;
    const int h = img.rows;

    Mat hsv, hls, gray;
    cvtColor(img, hsv,  COLOR_BGR2HSV);
    cvtColor(img, hls,  COLOR_BGR2HLS);
    cvtColor(img, gray, COLOR_BGR2GRAY);

    road_mask_out = build_road_mask(img);

    // 白色车道线：低饱和、高亮；黄线：黄色 hue 范围。
    Mat white_hsv, white_hls, yellow_hsv, bright_gray;
    inRange(hsv, Scalar(0, 0, 145), Scalar(180, 80, 255), white_hsv);
    inRange(hls, Scalar(0, 135, 0), Scalar(180, 255, 145), white_hls);
    inRange(hsv, Scalar(12, 55, 85), Scalar(42, 255, 255), yellow_hsv);
    threshold(gray, bright_gray, 155, 255, THRESH_BINARY);

    Mat white = white_hsv & white_hls & bright_gray;
    Mat yellow = yellow_hsv;

    lane_mask = white | yellow;

    // 近场 ROI：把远处天空/灯杆/路牌尽量排掉。
    // 注意这里不是固定最终光毯形状，只是限制“用于拟合的像素区域”。
    // 近端优先：不要再让远处白车、灯杆、路牌参与主车道拟合。
    // 这里只保留下半部分路面区域，远端只用于显示延展，不参与强锚定。
    vector<Point> roi_poly = {
        Point((int)(w * 0.16f), (int)(h * 0.985f)),
        Point((int)(w * 0.84f), (int)(h * 0.985f)),
        Point((int)(w * 0.60f), (int)(h * 0.68f)),
        Point((int)(w * 0.40f), (int)(h * 0.68f))
    };

    Mat roi_mask = Mat::zeros(img.size(), CV_8UC1);
    fillPoly(roi_mask, vector<vector<Point>>{roi_poly}, Scalar(255));

    // 车道线应该和路面相邻。把 road mask 膨胀后再与 lane mask 相交：
    // 允许车道线本身很白、不完全属于 road mask，但它附近必须有路面。
    Mat road_dilate;
    dilate(road_mask_out, road_dilate, getStructuringElement(MORPH_RECT, Size(19, 19)));

    lane_mask &= roi_mask;
    lane_mask &= road_dilate;

    // 扣除车辆区域，避免白车边缘/车牌/反光作为车道线像素。
    if (!vehicle_mask.empty())
    {
        Mat inv_vehicle;
        bitwise_not(vehicle_mask, inv_vehicle);
        lane_mask &= inv_vehicle;
    }

    // 去掉明显横向结构：护栏、阴影、字幕边缘。
    Mat horizontal;
    morphologyEx(lane_mask, horizontal, MORPH_OPEN,
                 getStructuringElement(MORPH_RECT, Size(60, 2)));
    dilate(horizontal, horizontal, getStructuringElement(MORPH_RECT, Size(5, 5)));
    lane_mask -= horizontal;

    // 连通增强：车道线可能是虚线，不能过度闭合；只做轻量形态学。
    morphologyEx(lane_mask, lane_mask, MORPH_OPEN,
                 getStructuringElement(MORPH_RECT, Size(2, 2)));
    morphologyEx(lane_mask, lane_mask, MORPH_CLOSE,
                 getStructuringElement(MORPH_RECT, Size(5, 9)));

    roi_poly_out = roi_poly;
    return lane_mask;
}

// ============================================================================
//  IPM 透视变换配置
//  这组点需要根据摄像头安装角度微调。
//  src 是原图中的近场道路梯形，dst 是 BEV 中的矩形道路区域。
// ============================================================================
static void build_ipm_transform(int w, int h, Mat &M, Mat &Minv, int &bev_w, int &bev_h)
{
    bev_w = w;
    bev_h = h;

    // 这组点比上一版 Hough 方案更“工程化”：
    // 只把下半部分路面拉成鸟瞰，不强行追很远的路面。
    // 只把近场路面拉成 BEV。远端对真实车道线信息贡献低、误检风险高，
    // 因此不再把 h*0.58 附近的远处区域强行纳入主拟合。
    vector<Point2f> src = {
        Point2f(w * 0.20f, h * 0.985f), // 左下
        Point2f(w * 0.80f, h * 0.985f), // 右下
        Point2f(w * 0.60f, h * 0.68f),  // 右上
        Point2f(w * 0.40f, h * 0.68f)   // 左上
    };

    vector<Point2f> dst = {
        Point2f(w * 0.24f, h * 0.985f),
        Point2f(w * 0.76f, h * 0.985f),
        Point2f(w * 0.76f, h * 0.18f),
        Point2f(w * 0.24f, h * 0.18f)
    };

    M = getPerspectiveTransform(src, dst);
    Minv = getPerspectiveTransform(dst, src);
}

// ============================================================================
//  二次多项式拟合 x = ay^2 + by + c
// ============================================================================
static QuadCurve fit_quadratic_curve(const vector<Point> &pts)
{
    QuadCurve q;
    q.pixel_count = (int)pts.size();

    if (pts.size() < 70)
        return q;

    // 近端高权重拟合：BEV 图像中 y 越大越靠近车头，可信度越高。
    // 远处白车/灯杆/路牌即使残留为白色像素，也只给很小权重，避免把整条光毯拉歪。
    Mat A((int)pts.size(), 3, CV_64F);
    Mat B((int)pts.size(), 1, CV_64F);

    int hmax = 1;
    for (const auto &p : pts) hmax = max(hmax, p.y);

    for (int i = 0; i < (int)pts.size(); ++i)
    {
        double y = (double)pts[i].y;
        double x = (double)pts[i].x;

        // y_norm 越接近 1 表示越靠近车头。
        double y_norm = clampd(y / (double)hmax, 0.0, 1.0);
        double weight = 0.10 + 0.90 * pow(y_norm, 3.0);
        double sw = sqrt(weight);

        A.at<double>(i, 0) = sw * y * y;
        A.at<double>(i, 1) = sw * y;
        A.at<double>(i, 2) = sw * 1.0;
        B.at<double>(i, 0) = sw * x;
    }

    Mat coeff;
    bool ok = solve(A, B, coeff, DECOMP_QR);
    if (!ok) return q;

    q.a = coeff.at<double>(0, 0);
    q.b = coeff.at<double>(1, 0);
    q.c = coeff.at<double>(2, 0);
    q.valid = true;

    // 加权残差评价：近端误差更重要，远端误差只做弱参考。
    double se = 0.0, sw_sum = 0.0;
    for (const auto &p : pts)
    {
        double y_norm = clampd((double)p.y / (double)hmax, 0.0, 1.0);
        double weight = 0.10 + 0.90 * pow(y_norm, 3.0);
        double pred = curve_x_at_y(q, p.y);
        double e = pred - p.x;
        se += weight * e * e;
        sw_sum += weight;
    }

    double rmse = sqrt(se / max(1e-6, sw_sum));
    double pix_score = clampd((double)pts.size() / 1000.0, 0.0, 1.0);
    double err_score = 1.0 - clampd(rmse / 38.0, 0.0, 1.0);
    q.confidence = 0.50 * pix_score + 0.50 * err_score;

    return q;
}

// ============================================================================
//  BEV 车道线候选特征增强：区分“间断车道线”和“连续路肩/边缘线”
// ----------------------------------------------------------------------------
//  当前问题的根因：最右侧连续白色路肩线在颜色和方向上很像车道线，
//  但它不是当前车道右边界。这里不再简单选择 histogram 最强的白线，
//  而是给每个候选 x 位置计算：
//  1) 近端白色像素强度；
//  2) 是否具有“间断/虚线”特征；
//  3) 是否太连续，像路肩实线；
//  4) 是否处在当前车道合理位置附近。
// ============================================================================
struct LaneBaseCandidate
{
    int x = -1;
    int hist = 0;
    double dash_score = 0.0;       // 越像间断车道线越高
    double continuous_score = 0.0; // 越像连续实线越高
    double position_score = 0.0;   // 越靠近当前车道预期边界越高
    double total_score = 0.0;
};

static void suppress_continuous_shoulder_lines_bev(Mat &bev_mask)
{
    if (bev_mask.empty()) return;

    const int w = bev_mask.cols;
    const int h = bev_mask.rows;
    const int y0 = (int)(h * 0.34f);
    const int y1 = (int)(h * 0.985f);
    const int min_run_for_edge = (int)((y1 - y0) * 0.42f);

    // 只重点压制左右边缘区域的连续线，不压制中间真实车道虚线。
    for (int x = 0; x < w; ++x)
    {
        int col_cnt = 0;
        int max_run = 0;
        int cur_run = 0;

        for (int y = y0; y < y1; ++y)
        {
            bool hit = bev_mask.at<uchar>(y, x) > 0;
            if (hit)
            {
                col_cnt++;
                cur_run++;
                if (cur_run > max_run) max_run = cur_run;
            }
            else
            {
                cur_run = 0;
            }
        }

        double occ = (double)col_cnt / max(1, y1 - y0);
        bool near_outer_edge = (x < (int)(w * 0.18f)) || (x > (int)(w * 0.72f));
        bool too_continuous = (max_run > min_run_for_edge && occ > 0.22);

        // 最右侧连续白线/路肩线通常在 BEV 中是一条长连续高占用线。
        // 一旦满足这个特征，直接在一个窄带内压掉，避免被当作右车道边界。
        if (near_outer_edge && too_continuous)
        {
            int x0 = max(0, x - 4);
            int x1 = min(w - 1, x + 4);
            rectangle(bev_mask, Point(x0, y0), Point(x1, y1), Scalar(0), FILLED);
        }
    }
}

static LaneBaseCandidate evaluate_lane_base_candidate(const Mat &bev_mask,
                                                      const vector<int> &hist,
                                                      int x,
                                                      int expected_x,
                                                      int side_sign)
{
    LaneBaseCandidate c;
    c.x = x;
    if (x < 0 || x >= bev_mask.cols) return c;

    const int w = bev_mask.cols;
    const int h = bev_mask.rows;
    const int y0 = (int)(h * 0.38f);
    const int y1 = (int)(h * 0.985f);
    const int band = max(4, (int)(w * 0.006f));

    int total_hits = 0;
    int max_run = 0;
    int cur_run = 0;
    int segments = 0;
    bool in_seg = false;

    for (int y = y0; y < y1; ++y)
    {
        bool hit = false;
        for (int xx = max(0, x - band); xx <= min(w - 1, x + band); ++xx)
        {
            if (bev_mask.at<uchar>(y, xx) > 0)
            {
                hit = true;
                break;
            }
        }

        if (hit)
        {
            total_hits++;
            cur_run++;
            if (cur_run > max_run) max_run = cur_run;
            if (!in_seg)
            {
                segments++;
                in_seg = true;
            }
        }
        else
        {
            cur_run = 0;
            in_seg = false;
        }
    }

    c.hist = hist[x];
    double occ = (double)total_hits / max(1, y1 - y0);
    double run_ratio = (double)max_run / max(1, y1 - y0);

    // 虚线/间断车道线：不是完全连续，同时有多个段；
    // 连续路肩线：run_ratio 很高，occ 也高。
    double segment_score = clampd((double)segments / 5.0, 0.0, 1.0);
    double not_too_continuous = 1.0 - clampd((run_ratio - 0.28) / 0.42, 0.0, 1.0);
    c.dash_score = clampd(0.55 * segment_score + 0.45 * not_too_continuous, 0.0, 1.0);
    c.continuous_score = clampd(0.55 * occ + 0.45 * run_ratio, 0.0, 1.0);

    double pos_err = fabs((double)x - (double)expected_x) / max(1.0, w * 0.11);
    c.position_score = 1.0 - clampd(pos_err, 0.0, 1.0);

    // 右侧候选额外惩罚最右侧连续路径线：它通常 x 很大，而且连续性强。
    double outer_penalty = 0.0;
    if (side_sign > 0 && x > (int)(w * 0.70f))
        outer_penalty = 0.55 + 0.45 * c.continuous_score;
    if (side_sign < 0 && x < (int)(w * 0.20f))
        outer_penalty = 0.35 + 0.35 * c.continuous_score;

    double hist_score = clampd((double)c.hist / max(12.0, h * 0.18), 0.0, 1.0);
    c.total_score = 1.15 * hist_score
                  + 1.40 * c.dash_score
                  + 1.35 * c.position_score
                  - 1.55 * c.continuous_score
                  - 1.75 * outer_penalty;

    return c;
}

static bool pick_lane_base_by_feature(const Mat &bev_mask,
                                      const vector<int> &hist,
                                      int x_min,
                                      int x_max,
                                      int expected_x,
                                      int side_sign,
                                      LaneBaseCandidate &best)
{
    best = LaneBaseCandidate();
    bool found = false;

    const int w = bev_mask.cols;
    x_min = max(0, x_min);
    x_max = min(w - 1, x_max);
    if (x_max <= x_min) return false;

    for (int x = x_min; x <= x_max; ++x)
    {
        if (hist[x] < 5) continue;
        LaneBaseCandidate c = evaluate_lane_base_candidate(bev_mask, hist, x, expected_x, side_sign);

        if (!found || c.total_score > best.total_score)
        {
            best = c;
            found = true;
        }
    }

    return found && best.total_score > 0.15;
}

// ============================================================================
//  BEV 滑动窗口搜索左右车道线像素
// ----------------------------------------------------------------------------
//  改进点：
//  1) 不再在右侧 0.50~0.85 全范围简单取 histogram 最大值；
//  2) 当前车道右边界被限制在合理范围，最右侧路肩连续线不允许作为右边界；
//  3) 使用“间断性/连续性/位置先验”综合评分选择左右基点；
//  4) 当前车道宽度以先验为主，检测结果只能小范围修正。
// ============================================================================
static bool sliding_window_search(const Mat &bev_mask,
                                  QuadCurve &left_curve,
                                  QuadCurve &right_curve,
                                  int &left_base_x,
                                  int &right_base_x)
{
    left_curve = QuadCurve();
    right_curve = QuadCurve();
    left_base_x = -1;
    right_base_x = -1;

    if (bev_mask.empty()) return false;

    const int w = bev_mask.cols;
    const int h = bev_mask.rows;

    vector<int> hist(w, 0);
    int y_start = (int)(h * 0.64f);
    for (int y = y_start; y < h; ++y)
    {
        const uchar *row = bev_mask.ptr<uchar>(y);
        for (int x = 0; x < w; ++x)
            if (row[x] > 0) hist[x]++;
    }

    // 平滑 histogram，避免单个噪声列成为峰值。
    vector<int> hist_smooth(w, 0);
    int smooth_r = max(3, (int)(w * 0.006f));
    for (int x = 0; x < w; ++x)
    {
        int s = 0;
        for (int xx = max(0, x - smooth_r); xx <= min(w - 1, x + smooth_r); ++xx)
            s += hist[xx];
        hist_smooth[x] = s;
    }

    // 当前车道左右边界的合理预期位置。这里核心是排除最右侧路肩连续白线。
    int expected_left  = (int)(w * 0.39f);
    int expected_right = (int)(w * 0.61f);

    LaneBaseCandidate left_cand, right_cand;
    bool has_left = pick_lane_base_by_feature(bev_mask, hist_smooth,
                                              (int)(w * 0.24f), (int)(w * 0.50f),
                                              expected_left, -1, left_cand);

    bool has_right = pick_lane_base_by_feature(bev_mask, hist_smooth,
                                               (int)(w * 0.50f), (int)(w * 0.70f),
                                               expected_right, +1, right_cand);

    if (!has_left && !has_right)
        return false;

    left_base_x = has_left ? left_cand.x : -1;
    right_base_x = has_right ? right_cand.x : -1;

    // 如果左右同时存在，但宽度明显不合理，优先用更可信的一侧 + 固定宽度外推另一侧。
    double expected_width = w * 0.25;
    if (has_left && has_right)
    {
        double width = right_base_x - left_base_x;
        if (width < expected_width * 0.72 || width > expected_width * 1.22)
        {
            if (left_cand.total_score >= right_cand.total_score)
            {
                right_base_x = (int)round(left_base_x + expected_width);
                has_right = false;
            }
            else
            {
                left_base_x = (int)round(right_base_x - expected_width);
                has_left = false;
            }
        }
    }
    else if (has_left && !has_right)
    {
        right_base_x = (int)round(left_base_x + expected_width);
    }
    else if (!has_left && has_right)
    {
        left_base_x = (int)round(right_base_x - expected_width);
    }

    left_base_x = (int)clampd(left_base_x, w * 0.22, w * 0.50);
    right_base_x = (int)clampd(right_base_x, w * 0.50, w * 0.70);

    vector<Point> nonzero;
    findNonZero(bev_mask, nonzero);
    if (nonzero.size() < 80)
        return false;

    // 近端优先：只从底部向上追踪到中近距离，不让远端白车/路牌/路肩线决定曲线。
    const int nwindows = 6;
    const int window_h = h / 9;
    const int margin = (int)(w * 0.050f);
    const int minpix = 18;

    vector<Point> left_pts;
    vector<Point> right_pts;

    int left_current = left_base_x;
    int right_current = right_base_x;

    for (int win = 0; win < nwindows; ++win)
    {
        int win_y_low  = h - (win + 1) * window_h;
        int win_y_high = h - win * window_h;
        if (win_y_low < (int)(h * 0.34f)) break;

        vector<Point> left_win_pts;
        vector<Point> right_win_pts;

        for (const Point &p : nonzero)
        {
            if (p.y < win_y_low || p.y >= win_y_high) continue;

            if (abs(p.x - left_current) <= margin)
                left_win_pts.push_back(p);

            if (abs(p.x - right_current) <= margin)
                right_win_pts.push_back(p);
        }

        if ((int)left_win_pts.size() > minpix)
        {
            int sx = 0;
            for (const auto &p : left_win_pts) sx += p.x;
            left_current = sx / (int)left_win_pts.size();
        }

        if ((int)right_win_pts.size() > minpix)
        {
            int sx = 0;
            for (const auto &p : right_win_pts) sx += p.x;
            right_current = sx / (int)right_win_pts.size();
        }

        // 再次防止右侧窗口被最右侧连续路肩线吸过去。
        right_current = (int)clampd(right_current, w * 0.50, w * 0.70);
        left_current  = (int)clampd(left_current,  w * 0.22, w * 0.50);

        left_pts.insert(left_pts.end(), left_win_pts.begin(), left_win_pts.end());
        right_pts.insert(right_pts.end(), right_win_pts.begin(), right_win_pts.end());
    }

    left_curve = fit_quadratic_curve(left_pts);
    right_curve = fit_quadratic_curve(right_pts);

    // 如果某侧点数不足，用另一侧 + 固定宽度兜底，避免误把最右侧路肩线拉进来。
    if (!right_curve.valid && left_curve.valid)
    {
        right_curve = offset_curve(left_curve, expected_width);
        right_curve.valid = true;
        right_curve.confidence = left_curve.confidence * 0.55;
        right_curve.pixel_count = left_curve.pixel_count;
    }
    if (!left_curve.valid && right_curve.valid)
    {
        left_curve = offset_curve(right_curve, -expected_width);
        left_curve.valid = true;
        left_curve.confidence = right_curve.confidence * 0.55;
        left_curve.pixel_count = right_curve.pixel_count;
    }

    return left_curve.valid || right_curve.valid;
}

// ============================================================================
//  曲线合理性检查：防止白车/灯杆残留导致曲线严重弯折或左右交叉。
// ============================================================================
static bool check_lane_geometry(const QuadCurve &left,
                                const QuadCurve &right,
                                int w, int h,
                                double expected_width,
                                double *avg_width_out = nullptr)
{
    if (!left.valid || !right.valid) return false;

    double widths[5];
    // 几何检查也偏近端：远端误检不应决定模型是否有效。
    double ys[5] = {
        h * 0.42, h * 0.56, h * 0.70, h * 0.84, h * 0.96
    };

    double sw = 0.0;
    for (int i = 0; i < 5; ++i)
    {
        double lx = curve_x_at_y(left, ys[i]);
        double rx = curve_x_at_y(right, ys[i]);
        double width = rx - lx;
        widths[i] = width;
        sw += width;

        if (lx < -w * 0.20 || lx > w * 1.20) return false;
        if (rx < -w * 0.20 || rx > w * 1.20) return false;
        if (width < expected_width * 0.55 || width > expected_width * 1.65) return false;
    }

    double avg_width = sw / 5.0;

    // 宽度沿 y 方向不能剧烈振荡，否则说明左右线不属于同一车道。
    for (int i = 0; i < 5; ++i)
    {
        if (fabs(widths[i] - avg_width) > expected_width * 0.48)
            return false;
    }

    if (avg_width_out) *avg_width_out = avg_width;
    return true;
}

// ============================================================================
//  用一侧曲线外推另一侧：车辆遮挡/虚线缺失时使用。
// ============================================================================
static void complete_missing_lane(QuadCurve &left,
                                  QuadCurve &right,
                                  int w, int h,
                                  double expected_width)
{
    if (left.valid && right.valid) return;

    if (left.valid && !right.valid)
    {
        right = offset_curve(left, expected_width);
        right.valid = true;
        right.confidence = left.confidence * 0.55;
        right.pixel_count = left.pixel_count;
    }
    else if (!left.valid && right.valid)
    {
        left = offset_curve(right, -expected_width);
        left.valid = true;
        left.confidence = right.confidence * 0.55;
        left.pixel_count = right.pixel_count;
    }
}

// ============================================================================
//  把 BEV 曲线区域反投影回原图
// ============================================================================
static vector<Point> project_bev_lane_polygon(const QuadCurve &left,
                                              const QuadCurve &right,
                                              const Mat &Minv,
                                              int bev_h,
                                              double y_top_ratio,
                                              double y_bottom_ratio)
{
    vector<Point2f> bev_pts;
    if (!left.valid || !right.valid || Minv.empty()) return {};

    int y_top = (int)(bev_h * y_top_ratio);
    int y_bottom = (int)(bev_h * y_bottom_ratio);
    int step = max(6, (y_bottom - y_top) / 28);

    vector<Point2f> left_side;
    vector<Point2f> right_side;

    for (int y = y_bottom; y >= y_top; y -= step)
    {
        float lx = (float)curve_x_at_y(left, y);
        left_side.push_back(Point2f(lx, (float)y));
    }

    for (int y = y_top; y <= y_bottom; y += step)
    {
        float rx = (float)curve_x_at_y(right, y);
        right_side.push_back(Point2f(rx, (float)y));
    }

    bev_pts.insert(bev_pts.end(), left_side.begin(), left_side.end());
    bev_pts.insert(bev_pts.end(), right_side.begin(), right_side.end());

    vector<Point2f> img_pts_f;
    perspectiveTransform(bev_pts, img_pts_f, Minv);

    vector<Point> img_pts;
    for (const auto &p : img_pts_f)
        img_pts.push_back(Point((int)round(p.x), (int)round(p.y)));

    return img_pts;
}

static vector<Point> project_bev_curve_points(const QuadCurve &curve,
                                              const Mat &Minv,
                                              int bev_h,
                                              double y_top_ratio,
                                              double y_bottom_ratio)
{
    vector<Point2f> bev_pts;
    if (!curve.valid || Minv.empty()) return {};

    int y_top = (int)(bev_h * y_top_ratio);
    int y_bottom = (int)(bev_h * y_bottom_ratio);
    int step = max(5, (y_bottom - y_top) / 36);

    for (int y = y_bottom; y >= y_top; y -= step)
    {
        float x = (float)curve_x_at_y(curve, y);
        bev_pts.push_back(Point2f(x, (float)y));
    }

    vector<Point2f> img_pts_f;
    perspectiveTransform(bev_pts, img_pts_f, Minv);

    vector<Point> img_pts;
    for (const auto &p : img_pts_f)
        img_pts.push_back(Point((int)round(p.x), (int)round(p.y)));

    return img_pts;
}

// ============================================================================
//  获取车道线 mask：优先使用 UNetMultiLane_seg.rknn，失败后回退传统 CV
// ----------------------------------------------------------------------------
//  这是本次改动的核心：
//  - 原来 lane_mask 来自 build_lane_pixel_mask()，本质还是颜色/亮度规则；
//  - 现在优先由 UNetLaneDetector 输出语义分割 lane_mask；
//  - 如果 lane.rknn 加载失败、推理失败或输出为空，则自动回退，不影响原程序运行。
// ============================================================================
static Mat get_lane_mask_with_unet_fallback(const Mat &img,
                                            const Mat &vehicle_mask)
{
    Mat lane_mask;

    if (img.empty())
        return lane_mask;

    // 延迟初始化：避免全局对象构造时就加载 RKNN。
    // build 目录运行时，../model/lane.rknn 通常对应项目根目录 model/lane.rknn。
    if (!g_unet_lane_try_init)
    {
        g_unet_lane_try_init = true;

        bool ok = false;
        ok = g_unet_lane.init("../model/lane.rknn");
        if (!ok)
            ok = g_unet_lane.init("./model/lane.rknn");

        if (ok)
            printf("[UNetLane] lane.rknn loaded successfully.\n");
        else
            printf("[UNetLane] lane.rknn not loaded, fallback to CV lane mask.\n");
    }

    bool unet_ok = false;
    if (g_unet_lane.is_loaded())
    {
        unet_ok = g_unet_lane.inference(img, lane_mask);
    }

    if (unet_ok && !lane_mask.empty())
    {
        // 保证输出为单通道二值图。
        if (lane_mask.channels() == 3)
            cvtColor(lane_mask, lane_mask, COLOR_BGR2GRAY);

        if (lane_mask.size() != img.size())
            resize(lane_mask, lane_mask, img.size(), 0, 0, cv::INTER_NEAREST);

        threshold(lane_mask, lane_mask, 80, 255, THRESH_BINARY);

        // 神经网络输出可能存在离散噪声，做轻量形态学即可。
        morphologyEx(lane_mask, lane_mask, MORPH_OPEN,
                     getStructuringElement(MORPH_RECT, Size(2, 2)));
        morphologyEx(lane_mask, lane_mask, MORPH_CLOSE,
                     getStructuringElement(MORPH_RECT, Size(5, 7)));

        putText(lane_mask, "", Point(0,0), FONT_HERSHEY_SIMPLEX, 0.1, Scalar(0), 1);
    }
    else
    {
        // 回退到传统 CV lane mask，确保没有 lane.rknn 时程序仍可运行。
        Mat road_mask;
        vector<Point> roi_poly;
        lane_mask = build_lane_pixel_mask(img, vehicle_mask, road_mask, roi_poly);
    }

    // 无论来自 UNet 还是 CV，都统一扣掉 YOLO 车辆区域，
    // 避免前车车身/反光/车牌区域参与后续 BEV 拟合。
    if (!vehicle_mask.empty() && !lane_mask.empty())
    {
        Mat inv_vehicle;
        bitwise_not(vehicle_mask, inv_vehicle);
        lane_mask &= inv_vehicle;
    }

    return lane_mask;
}

// ============================================================================
//  BEV 车道模型检测主函数
// ============================================================================
static BevLaneModel detect_bev_lane_model(const Mat &img, detect_result_group_t &result_group)
{
    BevLaneModel model;
    if (img.empty()) return model;

    const int w = img.cols;
    const int h = img.rows;

    model.img_w = w;
    model.img_h = h;
    model.top_y_img = (int)(h * 0.68f);
    model.bottom_y_img = (int)(h * 0.98f);

    build_ipm_transform(w, h, model.M, model.Minv, model.bev_w, model.bev_h);

    Mat vehicle_mask = build_vehicle_mask(img, result_group);

    // 优先使用 UNetMultiLane_seg.rknn 输出的语义车道线 mask；
    // 如果模型不存在、加载失败或推理失败，则自动回退传统 CV 方案。
    Mat lane_mask = get_lane_mask_with_unet_fallback(img, vehicle_mask);

    Mat bev_mask;
    warpPerspective(lane_mask, bev_mask, model.M, Size(model.bev_w, model.bev_h),
                    cv::INTER_NEAREST, cv::BORDER_CONSTANT, Scalar(0));

    // BEV 下只保留下方道路区域，进一步削弱远处灯杆/路牌。
    Mat bev_roi = Mat::zeros(bev_mask.size(), CV_8UC1);
    rectangle(bev_roi,
              Point((int)(model.bev_w * 0.08f), (int)(model.bev_h * 0.18f)),
              Point((int)(model.bev_w * 0.92f), (int)(model.bev_h * 0.985f)),
              Scalar(255), FILLED);
    bev_mask &= bev_roi;

    // 轻量形态学，把虚线局部连接，但不把车辆边缘大面积连进去。
    morphologyEx(bev_mask, bev_mask, MORPH_CLOSE,
                 getStructuringElement(MORPH_RECT, Size(5, 13)));
    morphologyEx(bev_mask, bev_mask, MORPH_OPEN,
                 getStructuringElement(MORPH_RECT, Size(3, 3)));

    // 关键增强：压制最右侧连续白色路肩/路径线。
    // 这类线不是“间断车道线”，不能作为当前车道右边界。
    suppress_continuous_shoulder_lines_bev(bev_mask);

    QuadCurve left, right;
    int left_base = -1, right_base = -1;
    bool found = sliding_window_search(bev_mask, left, right, left_base, right_base);

    model.left_base_x = left_base;
    model.right_base_x = right_base;

    // BEV 中预期车道宽度。这里约等于 dst 矩形宽度的一半。
    double expected_width = model.bev_w * 0.25;
    model.lane_width_bev = expected_width;

    if (!found)
    {
        // 完全没找到时，使用上一帧，避免画面突然飞走。
        if (g_last_bev_lane.valid)
            return g_last_bev_lane;
        return model;
    }

    // 单侧缺失时，用预期宽度外推另一侧。
    complete_missing_lane(left, right, model.bev_w, model.bev_h, expected_width);

    double avg_width = expected_width;
    bool geom_ok = check_lane_geometry(left, right, model.bev_w, model.bev_h,
                                       expected_width, &avg_width);

    // 几何不合理时优先回退上一帧，而不是相信当前帧错误锚点。
    if (!geom_ok)
    {
        if (g_last_bev_lane.valid)
            return g_last_bev_lane;

        // 没有上一帧时，构造一个保守 BEV 中央车道。
        left.valid = right.valid = true;
        left.a = right.a = 0.0;
        left.b = right.b = 0.0;
        left.c = model.bev_w * 0.375;
        right.c = model.bev_w * 0.625;
        left.confidence = right.confidence = 0.20;
        avg_width = expected_width;
    }

    // 时序平滑：曲线拟合要比 Hough 更容易抖，这里做温和平滑。
    double conf = (left.confidence + right.confidence) * 0.5;
    // 短期方案优先稳定，不让单帧误检大幅拉动光毯。
    double alpha = clampd(0.06 + 0.22 * conf, 0.06, 0.28);

    if (g_has_smooth_curve)
    {
        left = smooth_curve(g_smooth_left_curve, left, alpha);
        right = smooth_curve(g_smooth_right_curve, right, alpha);
    }

    g_smooth_left_curve = left;
    g_smooth_right_curve = right;
    g_has_smooth_curve = true;

    model.left = left;
    model.right = right;
    model.lane_width_bev = avg_width;
    model.valid = true;
    model.from_fallback = !geom_ok || conf < 0.45;
    model.confidence = (float)clampd(conf, 0.0, 1.0);

    // 当前车道。
    model.current_poly = project_bev_lane_polygon(left, right, model.Minv,
                                                  model.bev_h, 0.35, 0.985);

    // 左右邻车道：在 BEV 中按当前车道宽度平移曲线，再反投影。
    QuadCurve left_outer  = offset_curve(left,  -avg_width);
    QuadCurve right_outer = offset_curve(right,  avg_width);

    model.left_poly = project_bev_lane_polygon(left_outer, left, model.Minv,
                                               model.bev_h, 0.35, 0.985);
    model.right_poly = project_bev_lane_polygon(right, right_outer, model.Minv,
                                                model.bev_h, 0.35, 0.985);

    model.left_curve_img  = project_bev_curve_points(left, model.Minv,
                                                     model.bev_h, 0.35, 0.985);
    model.right_curve_img = project_bev_curve_points(right, model.Minv,
                                                     model.bev_h, 0.35, 0.985);

    g_last_bev_lane = model;
    return model;
}

// ============================================================================
//  目标风险计算：用反投影后的车道区域判断车辆归属
// ============================================================================
static LaneZoneType classify_vehicle_zone_by_poly(const BevLaneModel &lane,
                                                  int xmin, int xmax, int ymin, int ymax)
{
    if (!lane.valid) return LANE_OUTSIDE;

    int bw = xmax - xmin;
    int bh = ymax - ymin;
    if (bw <= 0 || bh <= 0) return LANE_OUTSIDE;

    // 底边 5 个脚点投票，降低远处车辆跨线误判。
    float y = (float)ymax + 0.06f * bh;
    int cnt_left = 0, cnt_cur = 0, cnt_right = 0;

    const float xs[5] = {0.12f, 0.32f, 0.50f, 0.68f, 0.88f};
    for (float a : xs)
    {
        float x = xmin * (1.f - a) + xmax * a;
        Point2f p(x, y);

        if (point_in_polygon(lane.current_poly, p)) cnt_cur++;
        else if (point_in_polygon(lane.left_poly, p)) cnt_left++;
        else if (point_in_polygon(lane.right_poly, p)) cnt_right++;
    }

    if (cnt_cur >= 2) return LANE_CURRENT;
    if (cnt_left >= cnt_right && cnt_left >= 2) return LANE_LEFT;
    if (cnt_right > cnt_left && cnt_right >= 2) return LANE_RIGHT;
    return LANE_OUTSIDE;
}

static const char* lane_zone_name(LaneZoneType z)
{
    if (z == LANE_LEFT) return "LEFT LANE";
    if (z == LANE_CURRENT) return "CURRENT LANE";
    if (z == LANE_RIGHT) return "RIGHT LANE";
    return "OUTSIDE";
}

static MultiLaneRisk compute_multi_lane_risk(const Mat &img,
                                             detect_result_group_t &result_group,
                                             const BevLaneModel &lane)
{
    MultiLaneRisk risk;
    if (img.empty() || !lane.valid) return risk;

    const int iw = img.cols;
    const int ih = img.rows;
    const float img_area = (float)(iw * ih);

    for (int i = 0; i < result_group.box_count; ++i)
    {
        int xmin = max(0,    result_group.result[i].box.xmin);
        int ymin = max(0,    result_group.result[i].box.ymin);
        int xmax = min(iw-1, result_group.result[i].box.xmax);
        int ymax = min(ih-1, result_group.result[i].box.ymax);

        int bw = xmax - xmin;
        int bh = ymax - ymin;
        if (bw <= 0 || bh <= 0) continue;

        float conf = result_group.result[i].box_conf;
        if (conf < 0.35f) continue;
        if (!is_vehicle_label(result_group.result[i].label)) continue;

        LaneZoneType zone = classify_vehicle_zone_by_poly(lane, xmin, xmax, ymin, ymax);
        if (zone == LANE_OUTSIDE) continue;

        float area_ratio = (float)(bw * bh) / img_area;
        float bottom_ratio = (float)ymax / ih;

        // 远处小目标不触发当前车道强报警，避免误报。
        if (zone == LANE_CURRENT && bottom_ratio < 0.68f && area_ratio < 0.010f)
            continue;

        float item_risk = (5.0f * area_ratio + 1.75f * bottom_ratio) * (0.55f + 0.45f * conf);
        if (area_ratio > 0.015f && bottom_ratio > 0.66f) item_risk *= 1.35f;

        if      (zone == LANE_LEFT)    risk.left    += item_risk * 0.82f;
        else if (zone == LANE_CURRENT) risk.current += item_risk;
        else if (zone == LANE_RIGHT)   risk.right   += item_risk * 0.82f;
    }

    return risk;
}

static void draw_vehicle_lane_tags(Mat &img,
                                   detect_result_group_t &result_group,
                                   const BevLaneModel &lane)
{
    if (img.empty() || !lane.valid) return;

    const int iw = img.cols;
    const int ih = img.rows;

    for (int i = 0; i < result_group.box_count; ++i)
    {
        if (result_group.result[i].box_conf < 0.35f) continue;
        if (!is_vehicle_label(result_group.result[i].label)) continue;

        int xmin = max(0,    result_group.result[i].box.xmin);
        int ymin = max(0,    result_group.result[i].box.ymin);
        int xmax = min(iw-1, result_group.result[i].box.xmax);
        int ymax = min(ih-1, result_group.result[i].box.ymax);

        LaneZoneType zone = classify_vehicle_zone_by_poly(lane, xmin, xmax, ymin, ymax);
        if (zone == LANE_OUTSIDE) continue;

        Scalar color = (zone == LANE_CURRENT) ? Scalar(0, 0, 255) : Scalar(255, 180, 0);
        string txt = lane_zone_name(zone);
        int ty = max(24, ymin - 8);
        putText(img, txt, Point(xmin, ty), FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, LINE_AA);
    }
}

// ============================================================================
//  稳定状态机
// ============================================================================
static GuidanceState stable_guidance(GuidanceState raw)
{
    if (raw == g_candidate) g_cand_count++;
    else
    {
        g_candidate = raw;
        g_cand_count = 1;
    }

    if (g_cand_count >= 3)
        g_last_stable = g_candidate;

    return g_last_stable;
}

// ============================================================================
//  绘制：车道区域、曲线和光毯
// ============================================================================
static void draw_zone(Mat &img,
                      const vector<Point> &poly,
                      Scalar fill_color,
                      Scalar line_color,
                      double alpha,
                      int thickness)
{
    if (poly.size() < 3) return;

    Mat overlay = img.clone();
    fillPoly(overlay, vector<vector<Point>>{poly}, fill_color);
    addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img);
    polylines(img, vector<vector<Point>>{poly}, true, line_color, thickness, LINE_AA);
}

static void draw_poly_label(Mat &img, const vector<Point> &poly,
                            const string &txt, Scalar color)
{
    if (poly.empty()) return;

    // 取多边形上方附近位置放文字。
    Point p = poly[min((size_t)poly.size() - 1, (size_t)poly.size() / 2)];
    putText(img, txt, p + Point(8, -8), FONT_HERSHEY_SIMPLEX,
            0.62, color, 2, LINE_AA);
}

static void draw_bev_lane_overlay(Mat &img,
                                  const BevLaneModel &lane,
                                  const MultiLaneRisk &risk,
                                  GuidanceState state)
{
    if (!lane.valid) return;

    bool cur_danger = (state == GUIDE_SLOW_DOWN);
    bool left_danger = risk.left > 1.85f;
    bool right_danger = risk.right > 1.85f;

    draw_zone(img, lane.left_poly,
              left_danger ? Scalar(0, 80, 255) : Scalar(60, 255, 120),
              left_danger ? Scalar(0, 0, 255) : Scalar(255, 255, 255),
              0.12, 1);

    draw_zone(img, lane.current_poly,
              cur_danger ? Scalar(0, 80, 255) : Scalar(60, 255, 120),
              cur_danger ? Scalar(0, 0, 255) : Scalar(255, 255, 255),
              0.24, 2);

    draw_zone(img, lane.right_poly,
              right_danger ? Scalar(0, 80, 255) : Scalar(60, 255, 120),
              right_danger ? Scalar(0, 0, 255) : Scalar(255, 255, 255),
              0.12, 1);

    draw_poly_label(img, lane.left_poly, left_danger ? "WATCH LEFT" : "LEFT",
                    left_danger ? Scalar(0,0,255) : Scalar(255,180,0));
    draw_poly_label(img, lane.current_poly, cur_danger ? "SLOW DOWN" : "CURRENT",
                    cur_danger ? Scalar(0,0,255) : Scalar(0,255,255));
    draw_poly_label(img, lane.right_poly, right_danger ? "WATCH RIGHT" : "RIGHT",
                    right_danger ? Scalar(0,0,255) : Scalar(255,180,0));

    // 调试曲线：蓝色是真实拟合出来的左右车道曲线。
    if (lane.left_curve_img.size() > 1)
        polylines(img, vector<vector<Point>>{lane.left_curve_img}, false,
                  Scalar(255, 80, 0), 3, LINE_AA);

    if (lane.right_curve_img.size() > 1)
        polylines(img, vector<vector<Point>>{lane.right_curve_img}, false,
                  Scalar(255, 80, 0), 3, LINE_AA);

    char buf[180];
    snprintf(buf, sizeof(buf), "BEV Lane conf:%.2f  Lpix:%d  Rpix:%d  %s  %s",
             lane.confidence,
             lane.left.pixel_count,
             lane.right.pixel_count,
             lane.from_fallback ? "SMOOTH/FALLBACK" : "PIXEL-FIT",
             g_unet_lane.is_loaded() ? "UNET" : "CV");

    putText(img, buf, Point(30, 48), FONT_HERSHEY_SIMPLEX, 0.68,
            Scalar(0, 255, 255), 2, LINE_AA);
}

static void draw_guidance_bar(Mat &img, GuidanceState state, const MultiLaneRisk &risk)
{
    int w = img.cols;

    Mat overlay = img.clone();
    rectangle(overlay, Point(0, 0), Point(w - 1, 95), Scalar(0, 0, 0), FILLED);
    addWeighted(overlay, 0.16, img, 0.84, 0.0, img);

    bool left_danger  = (risk.left  > 1.85f);
    bool right_danger = (risk.right > 1.85f);

    if (state == GUIDE_SLOW_DOWN)
    {
        putText(img, "SLOW DOWN", Point(w/2 - 115, 40), FONT_HERSHEY_SIMPLEX,
                1.05, Scalar(0, 0, 255), 3, LINE_AA);
        putText(img, "CURRENT LANE VEHICLE", Point(w/2 - 160, 78), FONT_HERSHEY_SIMPLEX,
                0.72, Scalar(0, 0, 255), 2, LINE_AA);
    }
    else if (left_danger || right_danger)
    {
        const char *txt = nullptr;
        int x_offset = 0;

        if (left_danger && right_danger)
        {
            txt = "WATCH BOTH";
            x_offset = 120;
        }
        else if (left_danger)
        {
            txt = "WATCH LEFT";
            x_offset = 125;
        }
        else
        {
            txt = "WATCH RIGHT";
            x_offset = 135;
        }

        putText(img, txt, Point(w/2 - x_offset, 42), FONT_HERSHEY_SIMPLEX,
                1.00, Scalar(0, 0, 255), 3, LINE_AA);
        putText(img, "SIDE VEHICLE", Point(w/2 - 115, 78), FONT_HERSHEY_SIMPLEX,
                0.72, Scalar(0, 0, 255), 2, LINE_AA);
    }
    else
    {
        putText(img, "KEEP LANE", Point(w/2 - 115, 40), FONT_HERSHEY_SIMPLEX,
                1.05, Scalar(0, 220, 255), 3, LINE_AA);
        arrowedLine(img, Point(w/2, 82), Point(w/2, 48),
                    Scalar(0, 220, 255), 4, LINE_AA, 0, 0.25);
    }
}


// ============================================================================
//  UNet Mask 驱动车道线可视化方案
// ----------------------------------------------------------------------------
//  这一版不再把 UNet lane_mask 强行转成 BEV 多项式光毯。
//  UNet 的输出直接作为最终车道线语义依据：
//  1) 过滤小连通域，保留稳定车道线区域；
//  2) 对车道线适度膨胀，形成弱背景提示带；
//  3) 车辆靠近中/左/右区域时，用弱红色提示风险；
//  4) 不再强制拟合 CURRENT 左右边界，避免车道线被路肩线/相邻线拉动。
// ============================================================================
static cv::Mat filter_lane_mask_components(const cv::Mat &mask, int min_area)
{
    if (mask.empty()) return cv::Mat();

    cv::Mat bin;
    if (mask.channels() == 3)
        cv::cvtColor(mask, bin, cv::COLOR_BGR2GRAY);
    else
        bin = mask.clone();

    cv::threshold(bin, bin, 80, 255, cv::THRESH_BINARY);

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);

    cv::Mat filtered = cv::Mat::zeros(bin.size(), CV_8UC1);
    for (int i = 1; i < n; ++i)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int top  = stats.at<int>(i, cv::CC_STAT_TOP);
        int h    = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // 面积太小的是噪声；完全处在远处上半区的小碎片也不要。
        if (area < min_area) continue;
        if (top + h < bin.rows * 0.42) continue;

        filtered.setTo(255, labels == i);
    }

    return filtered;
}

static cv::Mat build_near_field_roi_mask(const cv::Size &sz)
{
    int w = sz.width;
    int h = sz.height;

    cv::Mat roi = cv::Mat::zeros(sz, CV_8UC1);

    std::vector<cv::Point> poly = {
        cv::Point((int)(w * 0.06f), (int)(h * 0.99f)),
        cv::Point((int)(w * 0.94f), (int)(h * 0.99f)),
        cv::Point((int)(w * 0.72f), (int)(h * 0.42f)),
        cv::Point((int)(w * 0.28f), (int)(h * 0.42f))
    };

    cv::fillPoly(roi, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(255));
    return roi;
}

static void apply_vehicle_mask_to_lane(cv::Mat &lane_mask,
                                       const cv::Mat &vehicle_mask)
{
    if (lane_mask.empty() || vehicle_mask.empty()) return;

    cv::Mat inv_vehicle;
    cv::bitwise_not(vehicle_mask, inv_vehicle);
    lane_mask &= inv_vehicle;
}

static void build_lane_visual_masks(const cv::Mat &raw_lane_mask,
                                    const cv::Mat &vehicle_mask,
                                    cv::Mat &lane_line_mask,
                                    cv::Mat &lane_band_mask)
{
    lane_line_mask = cv::Mat();
    lane_band_mask = cv::Mat();

    if (raw_lane_mask.empty()) return;

    cv::Mat mask;
    if (raw_lane_mask.channels() == 3)
        cv::cvtColor(raw_lane_mask, mask, cv::COLOR_BGR2GRAY);
    else
        mask = raw_lane_mask.clone();

    cv::threshold(mask, mask, 80, 255, cv::THRESH_BINARY);

    // 只关注中近场区域，避免远方小噪声铺满画面。
    cv::Mat roi = build_near_field_roi_mask(mask.size());
    mask &= roi;

    apply_vehicle_mask_to_lane(mask, vehicle_mask);

    int min_area = std::max(60, (int)(mask.cols * mask.rows * 0.000025));
    lane_line_mask = filter_lane_mask_components(mask, min_area);

    // 轻量连接，让断续车道线视觉上更连贯，但不做强几何拟合。
    cv::morphologyEx(lane_line_mask, lane_line_mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 9)));
    cv::morphologyEx(lane_line_mask, lane_line_mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));

    // 车道线本身太细，膨胀成“弱提示带”，用于半透明可视化和风险区域。
    int kx = std::max(9, (int)(mask.cols * 0.012));
    int ky = std::max(9, (int)(mask.rows * 0.018));
    if (kx % 2 == 0) kx++;
    if (ky % 2 == 0) ky++;

    cv::dilate(lane_line_mask, lane_band_mask,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kx, ky)));

    // 进一步限制在近场道路区域内。
    lane_band_mask &= roi;
    apply_vehicle_mask_to_lane(lane_band_mask, vehicle_mask);
}

static void overlay_mask_color(cv::Mat &img,
                               const cv::Mat &mask,
                               const cv::Scalar &color,
                               double alpha)
{
    if (img.empty() || mask.empty()) return;

    cv::Mat overlay = img.clone();
    overlay.setTo(color, mask);
    cv::addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img);
}

static LaneZoneType classify_vehicle_simple_zone(int xmin, int xmax, int ymin, int ymax,
                                                 int img_w, int img_h)
{
    int bw = xmax - xmin;
    int bh = ymax - ymin;
    if (bw <= 0 || bh <= 0) return LANE_OUTSIDE;

    float foot_x = 0.5f * (xmin + xmax);
    float foot_y = ymax + 0.06f * bh;

    // 只处理中近场目标；远处目标只画框，不触发强车道提示。
    if (foot_y < img_h * 0.50f) return LANE_OUTSIDE;

    if (foot_x < img_w * 0.42f) return LANE_LEFT;
    if (foot_x > img_w * 0.58f) return LANE_RIGHT;
    return LANE_CURRENT;
}

static MultiLaneRisk compute_mask_lane_risk(const cv::Mat &img,
                                            detect_result_group_t &result_group,
                                            const cv::Mat &lane_band_mask)
{
    MultiLaneRisk risk;
    if (img.empty()) return risk;

    int iw = img.cols;
    int ih = img.rows;
    float img_area = (float)(iw * ih);

    for (int i = 0; i < result_group.box_count; ++i)
    {
        int xmin = std::max(0,    result_group.result[i].box.xmin);
        int ymin = std::max(0,    result_group.result[i].box.ymin);
        int xmax = std::min(iw-1, result_group.result[i].box.xmax);
        int ymax = std::min(ih-1, result_group.result[i].box.ymax);

        int bw = xmax - xmin;
        int bh = ymax - ymin;
        if (bw <= 0 || bh <= 0) continue;

        float conf = result_group.result[i].box_conf;
        if (conf < 0.35f) continue;
        if (!is_vehicle_label(result_group.result[i].label)) continue;

        LaneZoneType zone = classify_vehicle_simple_zone(xmin, xmax, ymin, ymax, iw, ih);
        if (zone == LANE_OUTSIDE) continue;

        float foot_x = 0.5f * (xmin + xmax);
        float foot_y = std::min((float)ih - 1.0f, (float)ymax + 0.08f * bh);

        // 如果有 lane_band，则要求车辆脚点附近和车道语义区域有一定关系；
        // 这样不会单纯因为 x 位置而把很外侧车辆强行算入。
        bool near_lane_semantics = true;
        if (!lane_band_mask.empty())
        {
            int cx = (int)std::round(foot_x);
            int cy = (int)std::round(foot_y);

            int x0 = std::max(0, cx - bw / 3);
            int x1 = std::min(iw - 1, cx + bw / 3);
            int y0 = std::max(0, cy - 12);
            int y1 = std::min(ih - 1, cy + 18);

            int hit = 0, total = 0;
            for (int y = y0; y <= y1; y += 3)
            {
                const uchar *row = lane_band_mask.ptr<uchar>(y);
                for (int x = x0; x <= x1; x += 3)
                {
                    total++;
                    if (row[x] > 0) hit++;
                }
            }

            float ratio = total > 0 ? (float)hit / (float)total : 0.0f;

            // 中央车道稍微严格一点，左右提示稍微宽松一点。
            near_lane_semantics = (zone == LANE_CURRENT) ? (ratio > 0.02f || ymax > ih * 0.68f)
                                                         : (ratio > 0.01f || ymax > ih * 0.60f);
        }

        if (!near_lane_semantics) continue;

        float area_ratio = (float)(bw * bh) / img_area;
        float bottom_ratio = (float)ymax / ih;

        float item_risk = (5.0f * area_ratio + 1.75f * bottom_ratio) * (0.55f + 0.45f * conf);

        if (zone == LANE_CURRENT)
            risk.current += item_risk;
        else if (zone == LANE_LEFT && bottom_ratio > 0.54f)
            risk.left += item_risk * 0.65f;
        else if (zone == LANE_RIGHT && bottom_ratio > 0.54f)
            risk.right += item_risk * 0.65f;
    }

    return risk;
}

static void draw_mask_side_warning_arrow(cv::Mat &img, bool left_side, const std::string &txt, bool danger)
{
    if (!danger) return;

    int w = img.cols;
    int h = img.rows;
    cv::Scalar color = cv::Scalar(0, 0, 255);

    cv::Point base = left_side ? cv::Point(45, h * 0.55) : cv::Point(w - 45, h * 0.55);
    cv::Point tip  = left_side ? cv::Point(120, h * 0.55) : cv::Point(w - 120, h * 0.55);

    cv::arrowedLine(img, base, tip, color, 5, cv::LINE_AA, 0, 0.35);
    cv::putText(img, txt,
                left_side ? cv::Point(35, h * 0.55 - 25) : cv::Point(w - 215, h * 0.55 - 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv::LINE_AA);
}

static void draw_unet_mask_lane_overlay(cv::Mat &img,
                                        const cv::Mat &lane_line_mask,
                                        const cv::Mat &lane_band_mask,
                                        const MultiLaneRisk &risk,
                                        GuidanceState state)
{
    if (img.empty()) return;

    bool cur_danger = state == GUIDE_SLOW_DOWN;
    bool left_danger = risk.left > 2.45f;
    bool right_danger = risk.right > 2.45f;

    // 弱绿色背景：表示 UNet 识别到的车道线附近区域，不再画强制规则光毯。
    if (!lane_band_mask.empty())
    {
        overlay_mask_color(img, lane_band_mask,
                           cur_danger ? cv::Scalar(0, 70, 255) : cv::Scalar(60, 255, 120),
                           cur_danger ? 0.22 : 0.16);
    }

    // UNet 原始车道线高亮：蓝青色，保留“模型识别结果”的直观性。
    if (!lane_line_mask.empty())
    {
        cv::Mat lane_colored = img.clone();
        lane_colored.setTo(cv::Scalar(255, 180, 0), lane_line_mask);
        cv::addWeighted(lane_colored, 0.75, img, 0.25, 0.0, img);
    }

    draw_mask_side_warning_arrow(img, true,  "WATCH LEFT",  left_danger);
    draw_mask_side_warning_arrow(img, false, "WATCH RIGHT", right_danger);

    const char *src = g_unet_lane.is_loaded() ? "UNET-MASK" : "CV-MASK";
    char buf[180];
    snprintf(buf, sizeof(buf), "Lane source:%s  mask-based overlay  risk L:%.1f C:%.1f R:%.1f",
             src, risk.left, risk.current, risk.right);

    cv::putText(img, buf, cv::Point(30, 48), cv::FONT_HERSHEY_SIMPLEX,
                0.68, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    if (cur_danger)
    {
        cv::putText(img, "SLOW DOWN", cv::Point(img.cols/2 - 115, 122),
                    cv::FONT_HERSHEY_SIMPLEX, 0.95, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }
}

// ============================================================================
//  主入口：draw_result() 内调用
// ============================================================================
static void draw_driving_assist(Mat &img, detect_result_group_t &result_group)
{
    if (img.empty()) return;

    // 1) 获取车辆遮罩，避免车辆框内部的反光、车牌、白色车身影响车道显示。
    Mat vehicle_mask = build_vehicle_mask(img, result_group);

    // 2) 获取 lane_mask：优先 UNetMultiLane_seg.rknn，失败则自动回退 CV。
    Mat raw_lane_mask = get_lane_mask_with_unet_fallback(img, vehicle_mask);

    // 3) 从 lane_mask 构建两类 mask：
    //    lane_line_mask：UNet 识别出的车道线本体；
    //    lane_band_mask：车道线附近膨胀出的弱背景提示区域。
    Mat lane_line_mask, lane_band_mask;
    build_lane_visual_masks(raw_lane_mask, vehicle_mask, lane_line_mask, lane_band_mask);

    // 4) 基于车辆位置 + lane_band 语义区域做风险判断。
    MultiLaneRisk cur_risk = compute_mask_lane_risk(img, result_group, lane_band_mask);
    g_smooth_risk = 0.78f * g_smooth_risk + 0.22f * cur_risk.current;

    GuidanceState raw = (g_smooth_risk > 3.15f) ? GUIDE_SLOW_DOWN : GUIDE_KEEP_LANE;
    GuidanceState stable = stable_guidance(raw);

    // 5) 直接可视化 UNet lane_mask，不再重建强几何光毯。
    draw_unet_mask_lane_overlay(img, lane_line_mask, lane_band_mask, cur_risk, stable);

    // 6) 车辆所属提示：这里不再依赖复杂多边形，只保留简洁区域标签。
    const int iw = img.cols;
    const int ih = img.rows;
    for (int i = 0; i < result_group.box_count; ++i)
    {
        if (result_group.result[i].box_conf < 0.35f) continue;
        if (!is_vehicle_label(result_group.result[i].label)) continue;

        int xmin = max(0,    result_group.result[i].box.xmin);
        int ymin = max(0,    result_group.result[i].box.ymin);
        int xmax = min(iw-1, result_group.result[i].box.xmax);
        int ymax = min(ih-1, result_group.result[i].box.ymax);

        LaneZoneType zone = classify_vehicle_simple_zone(xmin, xmax, ymin, ymax, iw, ih);
        if (zone == LANE_OUTSIDE) continue;

        const char *txt = lane_zone_name(zone);
        Scalar color = (zone == LANE_CURRENT) ? Scalar(0, 0, 255) : Scalar(255, 180, 0);
        putText(img, txt, Point(xmin, max(24, ymin - 8)),
                FONT_HERSHEY_SIMPLEX, 0.55, color, 2, LINE_AA);
    }

    draw_guidance_bar(img, stable, cur_risk);
}


// ============================================================================
//  YOLOv5s 类实现
// ============================================================================
Yolov5s::Yolov5s(const char* model_path, int npu_index)
{
    int ret;
    model_data = load_model(model_path, this->model_size);
    ret = rknn_init(&this->context, model_data, this->model_size, RKNN_FLAG_PRIOR_HIGH, NULL);

    if (ret != 0) printf("rknn init failed! %d\n", ret);
    else          printf("yolo %d init ok\n", npu_index);

    if      (npu_index % 4 == 0) ret = rknn_set_core_mask(context, RKNN_NPU_CORE_0);
    else if (npu_index % 4 == 1) ret = rknn_set_core_mask(context, RKNN_NPU_CORE_1);
    else                         ret = rknn_set_core_mask(context, RKNN_NPU_CORE_2);

    if (ret != 0) printf("npu set failed! %d\n", ret);

    rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &num_tensors, sizeof(num_tensors));

    input_attrs.resize(num_tensors.n_input);
    output_attrs.resize(num_tensors.n_output);

    for (int i = 0; i < num_tensors.n_input; i++)
    {
        input_attrs[i].index = i;
        rknn_query(context, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(input_attrs[i]));
        print_tensor_attr(&input_attrs[i]);
    }

    for (int i = 0; i < num_tensors.n_output; i++)
    {
        output_attrs[i].index = i;
        rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(output_attrs[i]));
        print_tensor_attr(&output_attrs[i]);
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        model_channel = input_attrs[0].dims[1];
        model_height  = input_attrs[0].dims[2];
        model_width   = input_attrs[0].dims[3];
    }
    else
    {
        model_height  = input_attrs[0].dims[1];
        model_width   = input_attrs[0].dims[2];
        model_channel = input_attrs[0].dims[3];
    }
}

Yolov5s::~Yolov5s()
{
    if (context) rknn_destroy(context);
    free(model_data);
}

unsigned char *Yolov5s::load_model(const char* model_path, unsigned int &model_size)
{
    FILE *fp = fopen(model_path, "rb");
    if (!fp)
    {
        printf("open model failed!\n");
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char *d = (unsigned char*)malloc(model_size);
    if (!d)
    {
        printf("malloc failed!\n");
        fclose(fp);
        return nullptr;
    }

    if (fread(d, 1, model_size, fp) != model_size)
    {
        printf("read failed!\n");
        free(d);
        fclose(fp);
        return nullptr;
    }

    fclose(fp);
    return d;
}

// ============================================================================
//  推理
// ============================================================================
int Yolov5s::inference_image(Mat &orig_img, detect_result_group_t &result_group)
{
    int ret = 0;

    this->img_height = orig_img.rows;
    this->img_width  = orig_img.cols;
    this->img_channel = orig_img.channels();

    Mat bkg;
    if (img_width % 16 != 0 || img_height % 16 != 0)
    {
        int bw = (img_width + 15) / 16 * 16;
        int bh = (img_height + 15) / 16 * 16;

        bkg = Mat(bh, bw, CV_8UC3, Scalar(0, 0, 0));
        orig_img.copyTo(bkg(Rect(0, 0, orig_img.cols, orig_img.rows)));

        img_width = bw;
        img_height = bh;
    }
    else
    {
        bkg = orig_img.clone();
    }

    int rh = model_height;
    int rw = model_width;
    int rc = model_channel;

    char *sb = (char*)malloc(img_height * img_width * img_channel);
    char *sc = (char*)malloc(img_height * img_width * img_channel);
    char *db = (char*)malloc(rh * rw * rc);

    if (!sb || !sc || !db)
    {
        if (sb) free(sb);
        if (sc) free(sc);
        if (db) free(db);
        return -1;
    }

    if (bkg.empty())
    {
        free(sb);
        free(sc);
        free(db);
        return -1;
    }

    memcpy(sb, bkg.data, img_height * img_width * img_channel);
    memset(sc, 0, img_height * img_width * img_channel);
    memset(db, 0, rh * rw * rc);

    rga_buffer_handle_t sh = importbuffer_virtualaddr(sb, img_height * img_width * img_channel);
    rga_buffer_handle_t ch = importbuffer_virtualaddr(sc, img_height * img_width * img_channel);
    rga_buffer_handle_t dh = importbuffer_virtualaddr(db, rh * rw * rc);

#define CLEANUP \
    if (sh) releasebuffer_handle(sh); \
    if (ch) releasebuffer_handle(ch); \
    if (dh) releasebuffer_handle(dh); \
    free(sb); \
    free(sc); \
    free(db);

    if (!sh || !ch || !dh)
    {
        CLEANUP;
        return -1;
    }

    rga_buffer_t src     = wrapbuffer_handle(sh, img_width, img_height, RK_FORMAT_BGR_888);
    rga_buffer_t src_cvt = wrapbuffer_handle(ch, img_width, img_height, RK_FORMAT_RGB_888);
    rga_buffer_t dst     = wrapbuffer_handle(dh, rw, rh, RK_FORMAT_RGB_888);

    if (imcheck(src, dst, {}, {}) != IM_STATUS_NOERROR)
    {
        CLEANUP;
        return -1;
    }

    if (imcvtcolor(src, src_cvt, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888) != IM_STATUS_SUCCESS)
    {
        CLEANUP;
        return -1;
    }

    if (imresize(src_cvt, dst) != IM_STATUS_SUCCESS)
    {
        CLEANUP;
        return -1;
    }

    int in_n = num_tensors.n_input;
    rknn_input inputs[in_n];
    memset(inputs, 0, sizeof(inputs));

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = rh * rw * rc;
    inputs[0].pass_through = false;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = db;

    if (rknn_inputs_set(context, in_n, inputs) != 0)
    {
        CLEANUP;
        return -1;
    }

    int on = num_tensors.n_output;
    rknn_output outputs[on];
    memset(outputs, 0, sizeof(outputs));

    for (int i = 0; i < on; i++)
        outputs[i].want_float = 0;

    if (rknn_run(context, NULL) != 0)
    {
        CLEANUP;
        return -1;
    }

    if (rknn_outputs_get(context, on, outputs, NULL) != 0)
    {
        CLEANUP;
        return -1;
    }

    float sw = (float)rw / img_width;
    float shh = (float)rh / img_height;

    vector<int32_t> qzps;
    vector<float> qscales;

    for (int i = 0; i < on; i++)
    {
        qzps.push_back(output_attrs[i].zp);
        qscales.push_back(output_attrs[i].scale);
    }

    post_process((int8_t*)outputs[0].buf,
                 (int8_t*)outputs[1].buf,
                 (int8_t*)outputs[2].buf,
                 rh, rw,
                 BOX_THRESHOLD,
                 NMS_THRESHOLD,
                 sw, shh,
                 qzps,
                 qscales,
                 result_group);

    draw_result(orig_img, result_group);

    rknn_outputs_release(context, on, outputs);
    CLEANUP;
#undef CLEANUP

    return 0;
}

// ============================================================================
//  绘制 YOLO 目标框 + BEV 车道光毯
// ============================================================================
int Yolov5s::draw_result(Mat &orig_img, detect_result_group_t &result_group)
{
    for (int i = 0; i < result_group.box_count; i++)
    {
        int xmin = max(0,               result_group.result[i].box.xmin);
        int ymin = max(0,               result_group.result[i].box.ymin);
        int xmax = min(orig_img.cols-1, result_group.result[i].box.xmax);
        int ymax = min(orig_img.rows-1, result_group.result[i].box.ymax);

        int bw = max(0, xmax - xmin);
        int bh = max(0, ymax - ymin);
        if (bw <= 0 || bh <= 0) continue;

        float conf = result_group.result[i].box_conf;

        rectangle(orig_img, Point(xmin, ymin), Point(xmax, ymax),
                  Scalar(255, 0, 0, 255), 3);

        if (conf < 0.45f || bw < 40 || bh < 25) continue;

        stringstream ss;
        ss << fixed << setprecision(2)
           << result_group.result[i].label << ":" << conf * 100 << " %";

        string lbl = ss.str();

        int bl = 0;
        Size ts = getTextSize(lbl, FONT_HERSHEY_SIMPLEX, 0.85, 2, &bl);

        int tx = xmin;
        int ty = (ymin - 10 < ts.height) ? ymin + ts.height + 8 : ymin - 10;

        rectangle(orig_img,
                  Point(tx, ty - ts.height - 4),
                  Point(tx + ts.width + 6, ty + bl),
                  Scalar(0, 0, 0), FILLED);

        putText(orig_img, lbl, Point(tx + 2, ty - 2),
                FONT_HERSHEY_SIMPLEX, 0.85,
                Scalar(0, 0, 255), 2, LINE_AA);
    }

    draw_driving_assist(orig_img, result_group);
    return 0;
}
