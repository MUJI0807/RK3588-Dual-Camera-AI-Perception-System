// opencl_undistort.cpp
// OpenCL 去畸变模块：Host 端实现
//
// 功能：在 Mali GPU 上执行镜头去畸变
// 位置：V4L2 采集之后、RGA 预处理之前
// 适用：RK3588 双摄项目 v8.0

#include "opencl_undistort.h"

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>

// OpenCL 内核源代码文件路径
static const char* KERNEL_FILE_PATH = "undistort.cl";

// 辅助宏：检查 OpenCL 错误
#define CL_CHECK(err) \
    do { \
        if ((err) != CL_SUCCESS) { \
            std::cerr << "[OpenCL] Error " << (err) << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
            return false; \
        } \
    } while (0)

OpenCLUndistortContext::OpenCLUndistortContext()
    : initialized_(false)
    , platform_(nullptr)
    , device_(nullptr)
    , context_(nullptr)
    , queue_(nullptr)
    , program_(nullptr)
    , kernel_(nullptr)
    , input_image_(nullptr)
    , output_image_(nullptr)
    , width_(0)
    , height_(0)
{
}

OpenCLUndistortContext::~OpenCLUndistortContext()
{
    release();
}

bool OpenCLUndistortContext::initialize(const CameraCalibration& calibration,
                                        int width, int height)
{
    if (initialized_) {
        std::cerr << "[OpenCL] Already initialized\n";
        return true;
    }

    calibration_ = calibration;
    width_ = width;
    height_ = height;

    cl_int err = CL_SUCCESS;

    // 步骤 1：获取平台列表
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    CL_CHECK(err);
    if (num_platforms == 0) {
        std::cerr << "[OpenCL] No platforms found\n";
        return false;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    CL_CHECK(err);

    // 步骤 2：选择 Mali GPU 平台
    platform_ = nullptr;
    for (cl_uint i = 0; i < num_platforms; ++i) {
        char platform_name[256] = {0};
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                                sizeof(platform_name), platform_name, nullptr);
        if (err == CL_SUCCESS) {
            std::cout << "[OpenCL] Platform " << i << ": " << platform_name << '\n';
            // 选择包含 "ARM" 或 "Mali" 的平台
            if (strstr(platform_name, "ARM") != nullptr ||
                strstr(platform_name, "Mali") != nullptr) {
                platform_ = platforms[i];
                std::cout << "[OpenCL] Selected ARM/Mali platform\n";
                break;
            }
        }
    }

    if (platform_ == nullptr) {
        // 如果没有找到 ARM/Mali 平台，使用第一个平台
        platform_ = platforms[0];
        std::cout << "[OpenCL] Using first platform (fallback)\n";
    }

    // 步骤 3：获取设备列表
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    CL_CHECK(err);
    if (num_devices == 0) {
        std::cerr << "[OpenCL] No GPU devices found\n";
        return false;
    }

    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    CL_CHECK(err);

    device_ = devices[0];  // 使用第一个 GPU 设备

    // 打印设备信息
    char device_name[256] = {0};
    err = clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    if (err == CL_SUCCESS) {
        std::cout << "[OpenCL] Device: " << device_name << '\n';
    }

    // 步骤 4：创建上下文
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    CL_CHECK(err);

    // 步骤 5：创建命令队列
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    CL_CHECK(err);

    // 步骤 6：加载并编译内核程序
    if (!load_kernel_program()) {
        return false;
    }

    // 步骤 7：创建内核
    kernel_ = clCreateKernel(program_, "undistort_kernel", &err);
    CL_CHECK(err);

    // 步骤 8：创建图像缓冲区
    if (!create_image_buffers()) {
        return false;
    }

    initialized_ = true;
    std::cout << "[OpenCL] Initialization complete (" << width << "x" << height << ")\n";
    return true;
}

bool OpenCLUndistortContext::load_kernel_program()
{
    // 读取内核源代码文件
    std::ifstream file(KERNEL_FILE_PATH);
    if (!file.is_open()) {
        std::cerr << "[OpenCL] Failed to open kernel file: " << KERNEL_FILE_PATH << '\n';
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    const char* source_ptr = source.c_str();
    const size_t source_length = source.length();

    cl_int err = CL_SUCCESS;

    // 创建程序对象
    program_ = clCreateProgramWithSource(context_, 1, &source_ptr, &source_length, &err);
    CL_CHECK(err);

    // 编译程序
    err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCL] Kernel build failed\n";

        // 获取编译日志
        size_t log_length = 0;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_length);
        std::vector<char> log(log_length);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG,
                              log_length, log.data(), nullptr);
        std::cerr << "[OpenCL] Build log:\n" << log.data() << '\n';
        return false;
    }

    return true;
}

bool OpenCLUndistortContext::create_image_buffers()
{
    cl_int err = CL_SUCCESS;

    // 创建输入图像对象（只读）
    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNORM_INT8;

    input_image_ = clCreateImage2D(context_,
                                   CL_MEM_READ_ONLY,
                                   &format,
                                   width_, height_,
                                   0,  // 行对齐（0=自动）
                                   nullptr,
                                   &err);
    CL_CHECK(err);

    // 创建输出图像对象（只写）
    output_image_ = clCreateImage2D(context_,
                                    CL_MEM_WRITE_ONLY,
                                    &format,
                                    width_, height_,
                                    0,
                                    nullptr,
                                    &err);
    CL_CHECK(err);

    return true;
}

bool OpenCLUndistortContext::undistort(const cv::Mat& input_bgr, cv::Mat& output_bgr)
{
    if (!initialized_) {
        std::cerr << "[OpenCL] Not initialized\n";
        return false;
    }

    if (input_bgr.empty() || input_bgr.cols != width_ || input_bgr.rows != height_) {
        std::cerr << "[OpenCL] Invalid input image\n";
        return false;
    }

    cl_int err = CL_SUCCESS;

    // 步骤 1：将 BGR 转换为 RGBA（OpenCL 图像格式要求）
    cv::Mat rgba;
    cv::cvtColor(input_bgr, rgba, cv::COLOR_BGR2RGBA);

    // 步骤 2：将输入图像数据写入 OpenCL 图像对象
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {(size_t)width_, (size_t)height_, 1};
    size_t row_pitch = rgba.step[0];

    err = clEnqueueWriteImage(queue_, input_image_, CL_TRUE,
                              origin, region,
                              row_pitch, 0,
                              rgba.data,
                              0, nullptr, nullptr);
    CL_CHECK(err);

    // 步骤 3：设置内核参数
    err  = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &input_image_);
    err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &output_image_);
    err |= clSetKernelArg(kernel_, 2, sizeof(cl_int), &width_);
    err |= clSetKernelArg(kernel_, 3, sizeof(cl_int), &height_);
    err |= clSetKernelArg(kernel_, 4, sizeof(cl_float), &calibration_.fx);
    err |= clSetKernelArg(kernel_, 5, sizeof(cl_float), &calibration_.fy);
    err |= clSetKernelArg(kernel_, 6, sizeof(cl_float), &calibration_.cx);
    err |= clSetKernelArg(kernel_, 7, sizeof(cl_float), &calibration_.cy);
    err |= clSetKernelArg(kernel_, 8, sizeof(cl_float), &calibration_.k1);
    err |= clSetKernelArg(kernel_, 9, sizeof(cl_float), &calibration_.k2);
    err |= clSetKernelArg(kernel_, 10, sizeof(cl_float), &calibration_.k3);
    err |= clSetKernelArg(kernel_, 11, sizeof(cl_float), &calibration_.p1);
    err |= clSetKernelArg(kernel_, 12, sizeof(cl_float), &calibration_.p2);
    CL_CHECK(err);

    // 步骤 4：执行内核
    size_t global_work_size[2] = {(size_t)width_, (size_t)height_};
    size_t local_work_size[2] = {16, 16};  // Mali GPU 推荐 16x16

    err = clEnqueueNDRangeKernel(queue_, kernel_, 2, nullptr,
                                 global_work_size, local_work_size,
                                 0, nullptr, nullptr);
    CL_CHECK(err);

    // 步骤 5：等待内核执行完成
    err = clFinish(queue_);
    CL_CHECK(err);

    // 步骤 6：从 OpenCL 图像对象读取输出数据
    cv::Mat rgba_output(height_, width_, CV_8UC4);
    err = clEnqueueReadImage(queue_, output_image_, CL_TRUE,
                             origin, region,
                             rgba_output.step[0], 0,
                             rgba_output.data,
                             0, nullptr, nullptr);
    CL_CHECK(err);

    // 步骤 7：将 RGBA 转换回 BGR
    cv::cvtColor(rgba_output, output_bgr, cv::COLOR_RGBA2BGR);

    return true;
}

void OpenCLUndistortContext::release()
{
    if (!initialized_) return;

    if (input_image_) clReleaseMemObject(input_image_);
    if (output_image_) clReleaseMemObject(output_image_);
    if (kernel_) clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);

    input_image_ = nullptr;
    output_image_ = nullptr;
    kernel_ = nullptr;
    program_ = nullptr;
    queue_ = nullptr;
    context_ = nullptr;

    initialized_ = false;
    std::cout << "[OpenCL] Resources released\n";
}
