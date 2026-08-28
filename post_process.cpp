#include "post_process.h"

#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>
#include <cstring>
#include <mutex>
#include <array>
#include <unistd.h>

using namespace std;

float anchor0[6] = {10, 13, 16, 30, 33, 23};
float anchor1[6] = {30, 61, 62, 45, 59, 119};
float anchor2[6] = {116, 90, 156, 198, 373, 326};

struct ProbArray
{
    float conf;
    int index;
};

cv::Mat test_img;
static vector<string> labels;
static std::once_flag labels_once;

// ================================ 后处理优化参数 ================================
// Top-K：进入 NMS 前最多保留多少个候选框。
// 数值越小，CPU越省；数值太小，可能漏检。
// 推荐：300~600。当前先用 300，适合实时推流。
static const int YOLO_POST_TOPK = 100;

// 是否使用 obj_conf * cls_conf 作为最终置信度。
// YOLOv5 标准做法一般是 objectness * class confidence。
// 打开后可以减少低质量框进入 NMS，CPU 更省，检测也更稳。
static const bool USE_OBJ_CLASS_PRODUCT = true;

// ================================ 基础函数 ================================

static float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float unsigmoid(float y)
{
    if (y <= 0.0f) y = 1e-6f;
    if (y >= 1.0f) y = 1.0f - 1e-6f;
    return -1.0f * logf(1.0f / y - 1.0f);
}

static int sort_descending(vector<ProbArray>& p_arr)
{
    sort(p_arr.begin(), p_arr.end(),
         [](const ProbArray& a, const ProbArray& b)
         {
             return a.conf > b.conf;
         });
    return 0;
}

static float calculateIOU(float xmin0, float ymin0, float xmax0, float ymax0,
                          float xmin1, float ymin1, float xmax1, float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));

    float inter_area = w * h;

    float area0 = fmax(0.f, xmax0 - xmin0) * fmax(0.f, ymax0 - ymin0);
    float area1 = fmax(0.f, xmax1 - xmin1) * fmax(0.f, ymax1 - ymin1);

    float union_area = area0 + area1 - inter_area;
    if (union_area <= 0.f)
        return 0.f;

    return inter_area / union_area;
}

// ================================ 修复后的类别感知 NMS ================================
// 注意：
// indexArray[i] 保存的是排序后第 i 个框对应的原始候选框索引。
// 所以访问 classID / boxes 时必须用 indexArray[i] 得到的原始索引 n / m。
// 原代码的问题：
// 1) classID[i] / classID[j] 用错了，应为 classID[n] / classID[m]
// 2) IoU 超阈值时删除了 indexArray[i]，实际上应该删除低分框 indexArray[j]
static int nms(int validCount,
               vector<float> &boxes,
               vector<int> &classID,
               vector<int>& indexArray,
               int currentClass,
               float nms_threshold)
{
    for (int i = 0; i < validCount; i++)
    {
        int n = indexArray[i];

        if (n == -1 || classID[n] != currentClass)
        {
            continue;
        }

        float xmin0 = boxes[n * 4 + 0];
        float ymin0 = boxes[n * 4 + 1];
        float xmax0 = boxes[n * 4 + 2] + xmin0;
        float ymax0 = boxes[n * 4 + 3] + ymin0;

        for (int j = i + 1; j < validCount; j++)
        {
            int m = indexArray[j];

            if (m == -1 || classID[m] != currentClass)
            {
                continue;
            }

            float xmin1 = boxes[m * 4 + 0];
            float ymin1 = boxes[m * 4 + 1];
            float xmax1 = boxes[m * 4 + 2] + xmin1;
            float ymax1 = boxes[m * 4 + 3] + ymin1;

            float iou = calculateIOU(xmin0, ymin0, xmax0, ymax0,
                                     xmin1, ymin1, xmax1, ymax1);

            if (iou > nms_threshold)
            {
                // 删除后面的低分框，而不是删除当前高分框
                indexArray[j] = -1;
            }
        }
    }

    return 0;
}

int readLines(const char *LablePath, vector<string> &lable_vector, int maxLines)
{
    ifstream file(LablePath);
    if (!file.is_open())
    {
        std::cerr << "file " << LablePath << " can not open!" << endl;
        return 0;
    }

    string line;
    while (getline(file, line))
    {
        lable_vector.emplace_back(line);

        if (lable_vector.size() >= static_cast<size_t>(maxLines))
        {
            break;
        }
    }

    return (int)lable_vector.size();
}

int LoadLableName(const char *filepath, vector<string> &lable_vector, int num_labels)
{
    int line_num = readLines(filepath, lable_vector, num_labels);

    if (line_num <= 0)
    {
        std::cerr << "[post_process] load labels failed: " << filepath << std::endl;
    }
    else
    {
        std::cout << "[post_process] labels loaded: " << line_num << std::endl;
    }

    return line_num;
}

// ================================ 线程安全标签加载 ================================
// 通过 /proc/self/exe 动态定位可执行文件所在目录，拼接标签文件路径。
// 这样无论程序从哪个工作目录启动，都能找到同级 model/ 下的标签文件。
static std::string label_path_from_executable()
{
    std::array<char, 4096> executable_path{};
    const ssize_t length = readlink("/proc/self/exe",
                                    executable_path.data(),
                                    executable_path.size() - 1);
    if (length <= 0) return LABLE_PATH;
    const std::string full_path(executable_path.data(),
                                static_cast<size_t>(length));
    const size_t slash = full_path.find_last_of('/');
    if (slash == std::string::npos) return LABLE_PATH;
    return full_path.substr(0, slash) + "/../model/coco_80_labels_list.txt";
}

// std::call_once 回调：加载标签文件，失败时用 class_N 占位。
static void load_labels_once()
{
    const std::string path = label_path_from_executable();
    labels.clear();
    const int loaded = LoadLableName(path.c_str(), labels, OBJ_CLASS_NUM);
    if (loaded != OBJ_CLASS_NUM) {
        std::cerr << "[Labels] expected " << OBJ_CLASS_NUM << " labels, loaded "
                  << loaded << " from " << path << "; using class_N fallback\n";
        labels.clear();
        labels.reserve(OBJ_CLASS_NUM);
        for (int id = 0; id < OBJ_CLASS_NUM; ++id) {
            labels.emplace_back("class_" + std::to_string(id));
        }
    } else {
        std::cerr << "[Labels] loaded " << loaded << " labels from " << path << '\n';
    }
}

static float deqnt_int8_to_f32(int int_num, int32_t zp, float scale)
{
    return (float)(int_num - zp) * scale;
}

inline static int32_t __limit_num(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return static_cast<int32_t>(f);
}

static int8_t qnt_f32_to_int8(float float_num, int32_t zp, float scale)
{
    float float_qnt_num = (float_num / scale) + zp;
    int8_t int_num = static_cast<int8_t>(__limit_num(float_qnt_num, -128, 127));
    return int_num;
}

/*
参数：
1. input：要处理的 buffer
2. anchor：锚框的长宽参数地址
3. grid_h、grid_w：单元网格数
4. model_height、model_width：模型要求的输入尺寸
5. stride：单元格的步长
6. boxes：存放检测框坐标
7. objProbs：存放最终置信度
8. classID：存放类别索引
9. box_threshold：过滤阈值
10. zp、scale：零点和缩放比例
*/
int process(int8_t *input,
            float *anchor,
            int grid_h,
            int grid_w,
            int model_height,
            int model_width,
            int stride,
            vector<float> &boxes,
            vector<float> &objProbs,
            vector<int> &classID,
            float box_threshold,
            int32_t zp,
            float scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    float box_unsig = unsigmoid(box_threshold);
    int8_t box_int8 = qnt_f32_to_int8(box_unsig, zp, scale);

    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int8_t obj_conf_i8 = input[(a * BOX_NUM_SIZE + 4) * grid_len + i * grid_w + j];

                // 先用 int8 阈值快速过滤，避免大量低置信度框进入浮点计算
                if (obj_conf_i8 <= box_int8)
                {
                    continue;
                }

                int box_offset = (a * BOX_NUM_SIZE) * grid_len + i * grid_w + j;
                int8_t *box_p = input + box_offset;

                float obj_conf = sigmoid(deqnt_int8_to_f32(obj_conf_i8, zp, scale));

                float box_x = sigmoid(deqnt_int8_to_f32(*(box_p + 0 * grid_len), zp, scale)) * 2.0f - 0.5f;
                float box_y = sigmoid(deqnt_int8_to_f32(*(box_p + 1 * grid_len), zp, scale)) * 2.0f - 0.5f;
                float box_w = sigmoid(deqnt_int8_to_f32(*(box_p + 2 * grid_len), zp, scale)) * 2.0f;
                float box_h = sigmoid(deqnt_int8_to_f32(*(box_p + 3 * grid_len), zp, scale)) * 2.0f;

                box_x = (box_x + j) * (float)stride;
                box_y = (box_y + i) * (float)stride;
                box_w = box_w * box_w * (float)anchor[a * 2 + 0];
                box_h = box_h * box_h * (float)anchor[a * 2 + 1];

                box_x = box_x - (box_w / 2.0f);
                box_y = box_y - (box_h / 2.0f);

                int8_t maxClassProb_i8 = *(box_p + 5 * grid_len);
                int maxClassId = 0;

                for (int k = 1; k < OBJ_CLASS_NUM; k++)
                {
                    int8_t prob = *(box_p + (5 + k) * grid_len);

                    if (prob > maxClassProb_i8)
                    {
                        maxClassProb_i8 = prob;
                        maxClassId = k;
                    }
                }

                float cls_conf = sigmoid(deqnt_int8_to_f32(maxClassProb_i8, zp, scale));

                float final_conf = USE_OBJ_CLASS_PRODUCT ? (obj_conf * cls_conf) : cls_conf;

                // 第二次用最终置信度过滤，减少低质量候选框进入 NMS
                if (final_conf < box_threshold)
                {
                    continue;
                }

                boxes.emplace_back(box_x);
                boxes.emplace_back(box_y);
                boxes.emplace_back(box_w);
                boxes.emplace_back(box_h);

                objProbs.emplace_back(final_conf);
                classID.emplace_back(maxClassId);

                validCount++;
            }
        }
    }

    return validCount;
}

inline static int clamp(float val, int min, int max)
{
    return val > min ? (val < max ? (int)val : max) : min;
}

/*
参数：
1. output0, output1, output2：模型的三个输出（量化后的 int8 数据）
2. model_height, model_width：输入图像尺寸
3. box_threshold：锚框的置信度阈值
4. nms_threshold：NMS 的 IoU 阈值
5. scale_w, scale_h：宽和高的缩放比例（映射回原图用）
6. qnt_zps, qnt_scales：三个输出对应的量化零点和缩放系数
*/
int post_process(int8_t *output0,
                 int8_t *output1,
                 int8_t *output2,
                 int model_height,
                 int model_width,
                 float box_threshold,
                 float nms_threshold,
                 float scale_w,
                 float scale_h,
                 std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales,
                 detect_result_group_t &result_group)
{
    result_group.box_count = 0;

    std::call_once(labels_once, load_labels_once);

    if (!output0 || !output1 || !output2)
    {
        return -1;
    }

    if (qnt_zps.size() < 3 || qnt_scales.size() < 3)
    {
        return -1;
    }

    vector<float> detect_boxes;
    vector<float> objProbs;
    vector<int> classID;

    // 预留容量，减少 emplace_back 时动态扩容
    detect_boxes.reserve(2048 * 4);
    objProbs.reserve(2048);
    classID.reserve(2048);

    // stride = 8
    int stride0 = 8;
    int grid_h0 = model_height / stride0;
    int grid_w0 = model_width / stride0;
    int validCount0 = process(output0,
                              anchor0,
                              grid_h0,
                              grid_w0,
                              model_height,
                              model_width,
                              stride0,
                              detect_boxes,
                              objProbs,
                              classID,
                              box_threshold,
                              qnt_zps[0],
                              qnt_scales[0]);

    // stride = 16
    int stride1 = 16;
    int grid_h1 = model_height / stride1;
    int grid_w1 = model_width / stride1;
    int validCount1 = process(output1,
                              anchor1,
                              grid_h1,
                              grid_w1,
                              model_height,
                              model_width,
                              stride1,
                              detect_boxes,
                              objProbs,
                              classID,
                              box_threshold,
                              qnt_zps[1],
                              qnt_scales[1]);

    // stride = 32
    int stride2 = 32;
    int grid_h2 = model_height / stride2;
    int grid_w2 = model_width / stride2;
    int validCount2 = process(output2,
                              anchor2,
                              grid_h2,
                              grid_w2,
                              model_height,
                              model_width,
                              stride2,
                              detect_boxes,
                              objProbs,
                              classID,
                              box_threshold,
                              qnt_zps[2],
                              qnt_scales[2]);

    int validCount = validCount0 + validCount1 + validCount2;

    if (validCount <= 0)
    {
        result_group.box_count = 0;
        return 0;
    }

    std::vector<ProbArray> prob_arr;
    prob_arr.reserve(validCount);

    for (int i = 0; i < validCount; i++)
    {
        ProbArray temp;
        temp.conf = objProbs[i];
        temp.index = i;
        prob_arr.emplace_back(temp);
    }

    sort_descending(prob_arr);

    // Top-K 裁剪：只让置信度最高的一部分候选框进入 NMS
    // NMS 是 O(N^2)，这里对 CPU 占用影响非常明显
    int keepCount = std::min(validCount, YOLO_POST_TOPK);

    std::vector<int> indexArray;
    indexArray.reserve(keepCount);

    std::vector<float> sortedProbs;
    sortedProbs.reserve(keepCount);

    for (int i = 0; i < keepCount; i++)
    {
        sortedProbs.emplace_back(prob_arr[i].conf);
        indexArray.emplace_back(prob_arr[i].index);
    }

    // 收集 Top-K 中出现过的类别，避免遍历无关类别
    std::set<int> class_set;
    for (int i = 0; i < keepCount; i++)
    {
        int origin_idx = indexArray[i];
        if (origin_idx >= 0 && origin_idx < (int)classID.size())
        {
            class_set.insert(classID[origin_idx]);
        }
    }

    // 类别感知 NMS
    for (const int& id : class_set)
    {
        nms(keepCount, detect_boxes, classID, indexArray, id, nms_threshold);
    }

    int count = 0;

    for (int i = 0; i < keepCount; i++)
    {
        if (indexArray[i] == -1)
        {
            continue;
        }

        if (count >= MAX_OBJ_BOXS)
        {
            break;
        }

        int n = indexArray[i];

        if (n < 0 || n >= (int)classID.size())
        {
            continue;
        }

        int id = classID[n];

        if (id < 0 || id >= (int)labels.size())
        {
            continue;
        }

        float xmin = detect_boxes[4 * n + 0];
        float ymin = detect_boxes[4 * n + 1];
        float xmax = detect_boxes[4 * n + 2] + xmin;
        float ymax = detect_boxes[4 * n + 3] + ymin;

        float box_conf = sortedProbs[i];

        result_group.result[count].box.xmin = (int)(clamp(xmin, 0, model_width) / scale_w);
        result_group.result[count].box.ymin = (int)(clamp(ymin, 0, model_height) / scale_h);
        result_group.result[count].box.xmax = (int)(clamp(xmax, 0, model_width) / scale_w);
        result_group.result[count].box.ymax = (int)(clamp(ymax, 0, model_height) / scale_h);
        result_group.result[count].box_conf = box_conf;

        const char *label_temp = labels[id].c_str();
        strncpy(result_group.result[count].label, label_temp, 31);
        result_group.result[count].label[31] = '\0';

        count++;
    }

    result_group.box_count = count;

    return 0;
}