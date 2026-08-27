#ifndef YOLOV5S_H                 // 头文件保护宏：防止该头文件被重复包含导致重复定义
#define YOLOV5S_H                 // 与上面的 #ifndef 配对，只有第一次包含时才会编译下面内容

#include <iostream>               // 标准输入输出流
#include <opencv2/core.hpp>       // OpenCV 核心数据结构（如 cv::Mat 等）
#include <opencv2/highgui.hpp>    // OpenCV GUI/图像显示相关（如 imshow 等，可能在 draw_result 中使用）
#include <opencv2/imgproc.hpp>    // OpenCV 图像处理（resize/颜色空间转换/绘图等）

#include <string.h>               // C 风格字符串处理
#include <vector>                 // STL 动态数组容器 vector
#include "3rdparty/librknn_api/include/rknn_api.h" // RKNN NPU 推理 SDK 的核心头文件：rknn_context、tensor attr、推理接口等

#include "RgaUtils.h"             // RGA 工具封装：通常提供 buffer/格式转换辅助
#include "im2d.h"                 // RGA 2D 图像处理接口（如 resize / crop / format convert 的 API）
#include "rga.h"                  // RGA 底层相关定义/接口

#include "post_process.h"         // 后处理：通常包含 NMS、阈值过滤、框解码、结果结构体 detect_result_group_t 等

// #include "3rdparty/rga/RK3588/include/im2d_version.h"   
// #include "3rdparty/rga/RV110X/include/im2d_type.h"      
// #include "3rdparty/rga/RK3588/include/im2d_buffer.h"    

using namespace std;              
using namespace cv;               

class Yolov5s                     // Yolov5s 推理类：封装 RKNN 模型加载、推理、以及结果绘制等流程
{
private:
    rknn_context context;         // RKNN 上下文句柄：由 rknn_init 创建，用于后续推理
                                  // 关键点：此处必须与 rknn_api.h 中的定义一致

    unsigned int model_size;      // 模型文件大小（字节）：load_model 读取模型时会填充，用于 rknn_init 等

    rknn_tensor_attr input_tensor;    // 单个输入张量属性
    rknn_tensor_attr output_tensor;   // 单个输出张量属性
    rknn_input_output_num num_tensors;// 模型输入/输出个数

    vector<rknn_tensor_attr> input_attrs;  // 输入张量属性数组：支持模型有多个输入
    vector<rknn_tensor_attr> output_attrs; // 输出张量属性数组：支持模型有多个输出

    unsigned char *model_data;   // 指向“模型文件内容”的内存指针
                                 // 典型流程：load_model malloc/读取 -> rknn_init 使用 -> 析构时释放

    unsigned char * load_model(const char* model_path, unsigned int &model_size);
                                 // 读取模型文件到内存
                                 // 参数：
                                 //   model_path：rknn 模型文件路径（.rknn）
                                 //   model_size：引用输出，返回模型文件字节数
                                 // 返回值：
                                 //   指向模型数据的指针（失败通常返回 nullptr）

public:

    Yolov5s(const char* model_path, int npu_index);
                                 // 构造函数：创建 Yolov5s 实例并初始化 NPU 推理环境
                                 // 常见会做的事（看 .cpp 实现）：
                                 // 1) load_model(model_path)
                                 // 2) rknn_init(&context, model_data, model_size, flags, nullptr)
                                 // 3) 查询输入输出信息：rknn_query(context, RKNN_QUERY_IN_OUT_NUM, ...)
                                 // 4) 填充 input_attrs/output_attrs
                                 // 5) 记录模型输入分辨率/通道等
                                 // npu_index：
                                 //   多 NPU 核心/多设备时用于绑定某个核心（RK3588 常见有多核 NPU）
                                 //   具体绑定方式取决于你工程里用的 RKNN 接口/扩展接口

    ~Yolov5s();                   // 析构函数：释放资源

    
    // 模型的高、宽和通道数
    int model_height;             // 模型期望输入的高度
    int model_width;              // 模型期望输入的宽度
    int model_channel;            // 模型期望输入的通道数

    // 输入图像的高、宽和通道数
    int img_height;               // 当前待推理图像的高度
    int img_width;                // 当前待推理图像的宽度
    int img_channel;              // 当前待推理图像的通道数

    //模型推理函数
    int inference_image(const Mat &origin_img, detect_result_group_t &result_group);
                                 // 对一帧图像执行推理 + 后处理
                                 // 输入：
                                 //   origin_img：原始图像（OpenCV Mat），通常是 BGR 格式
                                 // 输出：
                                 //   result_group：检测结果集合（类别、置信度、bbox 等）

    int draw_result(const cv::Mat &orig_img, detect_result_group_t &group);
                                 // 将检测结果画到图像上（框、类别、置信度等）
                                 // 输入：
                                 //   orig_img：原图（一般会在其上画框；如果函数内部 clone 也可能返回新图）
                                 //   group：检测结果集合
                                 // 返回：
                                 //   0/非0：成功/失败（依你工程约定）

};

#endif                           // 结束头文件保护宏，与 #ifndef YOLOV5S_H 配对
