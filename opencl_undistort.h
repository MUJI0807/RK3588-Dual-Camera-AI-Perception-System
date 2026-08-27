// opencl_undistort.h
// OpenCL 去畸变模块：Host 端封装
//
// 功能：在 Mali GPU 上执行镜头去畸变
// 位置：V4L2 采集之后、RGA 预处理之前
// 适用：RK3588 双摄项目 v8.0

#ifndef OPENCL_UNDISTORT_H
#define OPENCL_UNDISTORT_H

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <opencv2/core.hpp>
#include <string>

// 相机标定参数（模拟值，后续可替换为真实标定结果）
struct CameraCalibration {
    float fx;  // x 方向焦距（像素单位）
    float fy;  // y 方向焦距（像素单位）
    float cx;  // 光心 x 坐标
    float cy;  // 光心 y 坐标
    float k1;  // 径向畸变系数
    float k2;
    float k3;
    float p1;  // 切向畸变系数
    float p2;

    // 默认构造函数：使用模拟参数（适用于 1280x720 分辨率的 MIPI 摄像头）
    CameraCalibration()
        : fx(800.0f), fy(800.0f)
        , cx(640.0f), cy(360.0f)
        , k1(-0.25f), k2(0.05f), k3(0.0f)
        , p1(0.001f), p2(-0.001f)
    {}
};

// OpenCL 去畸变上下文（封装 OpenCL 资源）
class OpenCLUndistortContext {
public:
    OpenCLUndistortContext();
    ~OpenCLUndistortContext();

    // 初始化 OpenCL 环境：平台、设备、上下文、命令队列、内核
    // 参数：
    //   calibration - 相机标定参数
    //   width       - 图像宽度
    //   height      - 图像高度
    // 返回：true=成功，false=失败
    bool initialize(const CameraCalibration& calibration,
                    int width, int height);

    // 执行去畸变
    // 参数：
    //   input_bgr  - 输入图像（BGR 格式，CV_8UC3）
    //   output_bgr - 输出图像（BGR 格式，CV_8UC3，由调用者分配）
    // 返回：true=成功，false=失败
    bool undistort(const cv::Mat& input_bgr, cv::Mat& output_bgr);

    // 释放 OpenCL 资源
    void release();

    // 检查是否已初始化
    bool is_initialized() const { return initialized_; }

private:
    // 加载并编译 OpenCL 内核程序
    bool load_kernel_program();

    // 创建 OpenCL 图像对象
    bool create_image_buffers();

private:
    bool initialized_;

    // OpenCL 核心资源
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    cl_program program_;
    cl_kernel kernel_;

    // 图像缓冲区
    cl_mem input_image_;
    cl_mem output_image_;

    // 相机参数
    CameraCalibration calibration_;
    int width_;
    int height_;
};

#endif // OPENCL_UNDISTORT_H
