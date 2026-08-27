#include "post_process.h"
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <array>
#include <cstdio>
#include <mutex>
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

// Sigmoid 函数：计算输入值的 sigmoid 结果
static float sigmoid(float x)
{
    float y = 1 / (1 + expf(-x));
    return y;
}

// unsigmoid 函数：根据 sigmoid 结果反推输入值
static float unsigmoid(float y)
{
    float x = -1.0f * logf(1.0f / y - 1);
    return x;
}
static int sort_descending(vector<ProbArray>& p_arr)
{
    // 使用 lambda 表达式作为排序规则进行排序
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
    // 计算两个矩形框的交集宽度和高度
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    // 计算交集面积
    float i = w * h;
    // 计算并集面积
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    // 计算交并比，如果并集面积为 0，返回 0，否则返回交集面积除以并集面积
    float iou = u <= 0.f? 0.f : (i / u);
    return iou;
}
static int nms(int validCount, vector<float> &boxes, vector<int> &classID, 
                vector<int>& indexArray, int currentClass, float nms_threshold)
{
    for(int i = 0;i <validCount; i++)
    {
        if(indexArray[i] == -1 || classID[i] != currentClass)
        {
            continue;
        }
        
        int n = indexArray[i];
        for(int j = i+1; j < validCount; j++)
        {
            int m = indexArray[j];
            if(m == -1 || classID[j] != currentClass)
            {
                continue;
            }
            
            float xmin0 = boxes[n*4];
            float ymin0 = boxes[n*4+1];
            float xmax0 = boxes[n*4+2] + xmin0;
            float ymax0 = boxes[n*4+3] + ymin0;

            float xmin1 = boxes[m*4];
            float ymin1 = boxes[m*4+1];
            float xmax1 = boxes[m*4+2] + xmin1;
            float ymax1 = boxes[m*4+3] + ymin1;

            float iou = calculateIOU(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            if(iou > nms_threshold )
            {
                indexArray[i] = -1;
            }            
        }
    }
    return 0;
}

int readLines(const char * LablePath, vector<string> &lable_vector, int maxLines)
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
    return lable_vector.size();
}

int LoadLableName(const char * filepath, vector<string> &lable_vector, int num_labels)
{
    int line_num = readLines(filepath, lable_vector, num_labels);
    if(line_num > 0)
    {
        // cout << "标签数量是 " << line_num << endl;
    }
    return line_num;
}

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
    float float_num = (float)(int_num - zp) * scale;
    return float_num;
}

inline static int32_t __limit_num(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return static_cast<int32_t>(f);
}

// 量化：将浮点数转换为量化后的 int8_t 类型数据
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
7. objProbs：存放目标置信度
8. classID：存放类别索引
9. box_threshold：过滤阈值
10. zp、scale：零点和缩放比例
*/
int process(int8_t *input, float *anchor, int grid_h, int grid_w, int model_height, int model_width, int stride,
            vector<float> &boxes, vector<float> &objProbs, vector<int> &classID, float box_threshold, int32_t zp, float scale)
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
                int8_t box_anchor_conf = input[(a * BOX_NUM_SIZE + 4) * grid_len + i * grid_w + j];
                if (box_anchor_conf > box_int8)
                {
                    validCount++;
                    int box_offt = (a * BOX_NUM_SIZE) * grid_len + i * grid_w + j;
                    int8_t *box_p = input + box_offt;
                    
                    // 反量化和 sigmoid 操作获取框的坐标信息
                    float box_x = sigmoid(deqnt_int8_to_f32(*box_p, zp,scale)) * 2 - 0.5;
                    float box_y = sigmoid(deqnt_int8_to_f32(*(box_p + 1 * grid_len), zp,scale)) * 2 - 0.5;
                    float box_w = sigmoid(deqnt_int8_to_f32(*(box_p + 2 * grid_len), zp,scale)) * 2.0;
                    float box_h = sigmoid(deqnt_int8_to_f32(*(box_p + 3 * grid_len), zp,scale)) * 2.0;

                    // 计算框的坐标
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a*2];
                    box_h = box_h * box_h * (float)anchor[a*2 + 1];

                    box_x = box_x - (box_w / 2.0);
                    box_y = box_y - (box_h / 2.0);

                    boxes.emplace_back(box_x);
                    boxes.emplace_back(box_y);  
                    boxes.emplace_back(box_w);
                    boxes.emplace_back(box_h);
                    
                    // 获取最大类别概率及对应的类别 ID
                    int8_t maxClassProb = *(box_p + 5 * grid_len);
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; k++)
                    {
                        int8_t prob = *(box_p + (5 + k) * grid_len);
                        if (prob > maxClassProb)
                        {
                            maxClassProb = prob;
                            maxClassId = k;
                        }
                    }

                    objProbs.emplace_back(sigmoid(deqnt_int8_to_f32(maxClassProb, zp, scale)));
                    classID.emplace_back(maxClassId);
                }
            }
        }
    }
    return validCount;
}

static float IOU(const std::vector<float>& boxes, int idx1, int idx2)
{
    // 取出第 idx1 个检测框的 (x, y, w, h)
    float x1 = boxes[idx1 * 4 + 0];
    float y1 = boxes[idx1 * 4 + 1];
    float w1 = boxes[idx1 * 4 + 2];
    float h1 = boxes[idx1 * 4 + 3];

    // 取出第 idx2 个检测框的 (x, y, w, h)
    float x2 = boxes[idx2 * 4 + 0];
    float y2 = boxes[idx2 * 4 + 1];
    float w2 = boxes[idx2 * 4 + 2];
    float h2 = boxes[idx2 * 4 + 3];

    // 计算右下角坐标
    float x1max = x1 + w1;
    float y1max = y1 + h1;
    float x2max = x2 + w2;
    float y2max = y2 + h2;

    // 计算交集区域
    float inter_w = std::max(0.f, std::min(x1max, x2max) - std::max(x1, x2));
    float inter_h = std::max(0.f, std::min(y1max, y2max) - std::max(y1, y2));
    float inter_area = inter_w * inter_h;

    float area1 = w1 * h1;
    float area2 = w2 * h2;
    float union_area = area1 + area2 - inter_area;

    if (union_area <= 0.f) 
    {
        return 0.f;
    }
    return inter_area / union_area;
}

inline static int clamp(float val, int min, int max) { return val > min? (val < max? val : max) : min; }


/*
参数：
1. output0, output1, output2：模型的三个输出（量化后的 int8 数据）
2. model_height, model_width：输入图像尺寸
3. box_threshold：锚框的置信度阈值
4. nms_threshold：NMS 的 IoU 阈值
5. scale_w, scale_h：宽和高的缩放比例（映射回原图用）
6. qnt_zps, qnt_scales：三个输出对应的量化零点和缩放系数
*/
int post_process(int8_t *output0, int8_t *output1, int8_t *output2,
                 int model_height, int model_width, float box_threshold,
                 float nms_threshold, float scale_w, float scale_h,
                 std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales, detect_result_group_t &result_group)
{
    // 三个 NPU worker 可能同时进入后处理，只允许一个线程初始化全局标签表。
    std::call_once(labels_once, load_labels_once);
    // for (string &s : labels)
    // {
    //     cout << "lable name " << s << endl;
    // }
    
    // 示例：量化和反量化测试
    int8_t int8_num = qnt_f32_to_int8(1.5, 1, 8.0f/255.0f);
    // cout << static_cast<int>(int8_num) << endl;
    float f = deqnt_int8_to_f32(48, 1, 0.03137f);
    // cout << f << endl;

    vector<float> detect_boxes;
    vector<float> objProbs;
    vector<int> classID;

    // 处理第一个输出
    //cout << "第一个" << endl;
    int stride0 = 8;
    int grid_h0 = model_height / stride0;
    int grid_w0 = model_width / stride0;
    int validCount0 = process(output0, anchor0, grid_h0, grid_w0,
                              model_height, model_width, stride0,
                              detect_boxes, objProbs, classID,
                              box_threshold, qnt_zps[0], qnt_scales[0]);

    // 处理第二个输出
    //cout << "第二个" << endl;
    int stride1 = 16;
    int grid_h1 = model_height / stride1;
    int grid_w1 = model_width / stride1;
    int validCount1 = process(output1, anchor1, grid_h1, grid_w1,
                              model_height, model_width, stride1,
                              detect_boxes, objProbs, classID,
                              box_threshold, qnt_zps[1], qnt_scales[1]);

    // 处理第三个输出
    //cout << "第三个" << endl;
    int stride2 = 32;
    int grid_h2 = model_height / stride2;
    int grid_w2 = model_width / stride2;
    int validCount2 = process(output2, anchor2, grid_h2, grid_w2,
                              model_height, model_width, stride2,
                              detect_boxes, objProbs, classID,
                              box_threshold, qnt_zps[2], qnt_scales[2]);


    std::vector<int> indexArray;

    int validCount = validCount0 + validCount1 + validCount2;
    if (validCount < 0) {
        return 0;
    }
    //printf("jiacne:%d\n",validCount);
    
    std::vector<ProbArray> prob_arr;    
    for(int i = 0; i<validCount; i++)
    {
        ProbArray temp;
        temp.conf = objProbs[i];
        temp.index = i;
        prob_arr.emplace_back(temp);
    }
    sort_descending(prob_arr);

    objProbs.clear();
    indexArray.clear();

    for (int i = 0; i < validCount; i++) {
        objProbs.emplace_back(prob_arr[i].conf);
        indexArray.emplace_back(prob_arr[i].index);
    }


    std::set<int> class_set(begin(classID),end(classID));

    for(const int& id : class_set)
    {
        // printf("lable num:%d,name is %s\n",id,labels[id].c_str());
    }
    

    for(const int& id : class_set)
    {
        nms(validCount, detect_boxes, classID, indexArray, id, nms_threshold);
    }

    int count = 0;
    result_group.box_count = 0;
    
    for(int i = 0; i < validCount; i++)
    {
        if(indexArray[i] == -1 || count > MAX_OBJ_BOXS)
        {
            continue;
        }
        int n = indexArray[i];
        
        float xmin      = detect_boxes[4*n + 0];
        float ymin      = detect_boxes[4*n + 1];
        float xmax      = detect_boxes[4*n + 2] + xmin;
        float ymax      = detect_boxes[4*n + 3] + ymin;
        float box_conf  = objProbs[i];
        int id          = classID[n];

        result_group.result[count].box.xmin = (int)(clamp(xmin, 0, model_width) / scale_w);
        result_group.result[count].box.ymin = (int)(clamp(ymin, 0, model_height) / scale_h);
        result_group.result[count].box.xmax = (int)(clamp(xmax, 0, model_width) / scale_w);
        result_group.result[count].box.ymax = (int)(clamp(ymax, 0, model_height) / scale_h);
        result_group.result[count].box_conf = box_conf;

        const std::string fallback = "class_" + std::to_string(id);
        const std::string& label =
            (id >= 0 && static_cast<size_t>(id) < labels.size())
                ? labels[static_cast<size_t>(id)]
                : fallback;
        // snprintf 会确保 label 缓冲区以 NUL 结尾。
        snprintf(result_group.result[count].label,
                 sizeof(result_group.result[count].label),
                 "%s",
                 label.c_str());

        // printf("%s\n", labels[id].c_str());
        count++;
        result_group.box_count = count;
    }
   
    return 0;
}
