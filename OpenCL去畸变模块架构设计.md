# RK3588 双摄项目 v8.0 — OpenCL 去畸变模块架构设计

## 1. 模块定位与流水线集成

### 1.1 流水线位置

OpenCL 去畸变模块插入在 **V4L2 采集之后、RGA 预处理之前**，具体位于 `read_thread()` 函数内部：

```
V4L2 采集 (NV12 → BGR)
    ↓
OpenCL 去畸变 (仅 MIPI 摄像头)  ← 新增模块
    ↓
g_read_queue (线程安全队列)
    ↓
NPU 推理 (YOLOv5s + UNet)
    ↓
合成器 (左右拼接 1280x720)
    ↓
g_write_queue
    ↓
RGA (BGR → NV12)
    ↓
MPP (H.264 编码)
    ↓
RTMP 推流
```

### 1.2 设计决策

**为什么选择 OpenCL 而非 OpenGL？**
- RK3588 的 Mali-G610 GPU 支持 OpenCL 2.0，且 OpenCL 更适合通用图像处理任务
- OpenCL 提供显式内存控制，便于与 V4L2 MMAP/DMA-BUF 零拷贝流水线集成
- OpenCL C 内核可直接操作像素级计算，无需图形上下文

**为什么只对 MIPI 摄像头去畸变？**
- MIPI 摄像头（/dev/video11）通常使用广角镜头，畸变更明显
- USB 摄像头（/dev/video20）通常是标准镜头，畸变较小
- 后续可扩展为双路都去畸变

**为什么在 read_thread 中执行？**
- 保持流水线简洁：采集 → 去畸变 → 推理 → 编码
- 避免额外的线程同步开销
- 去畸变计算量相对较小（Mali GPU 并行处理），不会成为瓶颈

## 2. 数学模型

### 2.1 针孔相机模型

相机内参矩阵：

```
K = [fx  0  cx]
    [ 0 fy  cy]
    [ 0  0   1]
```

其中：
- `fx, fy`：x/y 方向焦距（像素单位）
- `cx, cy`：光心坐标（像素单位）

### 2.2 畸变模型

采用 **径向畸变 + 切向畸变** 组合模型：

**径向畸变**（由镜头曲面形状引起）：
```
x_distorted = x_norm * (1 + k1*r² + k2*r⁴ + k3*r⁶)
y_distorted = y_norm * (1 + k1*r² + k2*r⁴ + k3*r⁶)
```

**切向畸变**（由镜头与成像平面不平行引起）：
```
x_distorted += 2*p1*x_norm*y_norm + p2*(r² + 2*x_norm²)
y_distorted += p1*(r² + 2*y_norm²) + 2*p2*x_norm*y_norm
```

其中：
- `x_norm = (x - cx) / fx`，`y_norm = (y - cy) / fy`（归一化相机坐标）
- `r² = x_norm² + y_norm²`
- `k1, k2, k3`：径向畸变系数
- `p1, p2`：切向畸变系数

### 2.3 去畸变算法

对每个输出像素 `(x_out, y_out)`：

1. **归一化**：计算归一化相机坐标 `(x_norm, y_norm)`
2. **逆畸变**：应用畸变模型计算 `(x_distorted, y_distorted)`
3. **反投影**：映射回输入图像坐标 `(x_in, y_in)`
4. **采样**：使用双线性插值从输入图像采样

## 3. OpenCL 内核设计

### 3.1 内核函数签名

```c
__kernel void undistort_kernel(
    __read_only image2d_t input_img,      // 输入图像
    __write_only image2d_t output_img,    // 输出图像
    const int width,                       // 图像宽度
    const int height,                      // 图像高度
    const float fx, fy, cx, cy,           // 相机内参
    const float k1, k2, k3,               // 径向畸变系数
    const float p1, p2)                   // 切向畸变系数
```

### 3.2 工作项映射

- **NDRange 维度**：2D（对应图像宽高）
- **全局工作大小**：`(width, height)`
- **局部工作大小**：`(16, 16)`（Mali GPU 推荐）
- **每个工作项**：处理一个输出像素

### 3.3 双线性插值

内核内置双线性插值采样函数，支持亚像素精度：

```c
inline float4 sample_bilinear(__read_only image2d_t input_img,
                              float x, float y,
                              int width, int height,
                              sampler_t sampler)
{
    // 取整得到四个相邻像素
    int x0 = (int)x, y0 = (int)y;
    int x1 = x0 + 1, y1 = y0 + 1;
    
    // 计算小数部分（权重）
    float dx = x - (float)x0;
    float dy = y - (float)y0;
    
    // 读取四个相邻像素并插值
    float4 p00 = read_imagef(input_img, sampler, (int2)(x0, y0));
    float4 p10 = read_imagef(input_img, sampler, (int2)(x1, y0));
    float4 p01 = read_imagef(input_img, sampler, (int2)(x0, y1));
    float4 p11 = read_imagef(input_img, sampler, (int2)(x1, y1));
    
    float4 top = mix(p00, p10, dx);
    float4 bottom = mix(p01, p11, dx);
    return mix(top, bottom, dy);
}
```

## 4. Host 端封装

### 4.1 类设计

```cpp
class OpenCLUndistortContext {
public:
    bool initialize(const CameraCalibration& calibration,
                    int width, int height);
    bool undistort(const cv::Mat& input_bgr, cv::Mat& output_bgr);
    void release();
    
private:
    bool load_kernel_program();
    bool create_image_buffers();
    
    // OpenCL 资源
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    cl_program program_;
    cl_kernel kernel_;
    cl_mem input_image_;
    cl_mem output_image_;
    
    CameraCalibration calibration_;
    int width_, height_;
};
```

### 4.2 初始化流程

1. **平台选择**：遍历所有 OpenCL 平台，优先选择 ARM/Mali 平台
2. **设备选择**：选择第一个 GPU 设备
3. **上下文创建**：使用 `clCreateContext`
4. **命令队列创建**：使用 `clCreateCommandQueue`
5. **内核编译**：从 `undistort.cl` 文件加载源代码并编译
6. **图像缓冲区创建**：创建 `cl_mem` 图像对象（输入只读、输出只写）

### 4.3 执行流程

1. **BGR → RGBA**：OpenCV 颜色转换（OpenCL 图像格式要求）
2. **写入输入图像**：`clEnqueueWriteImage` 将 RGBA 数据上传到 GPU
3. **设置内核参数**：13 个参数（图像、尺寸、相机参数）
4. **执行内核**：`clEnqueueNDRangeKernel` 启动并行计算
5. **等待完成**：`clFinish` 同步
6. **读取输出图像**：`clEnqueueReadImage` 从 GPU 下载结果
7. **RGBA → BGR**：转换回 OpenCV 格式

## 5. 性能分析

### 5.1 计算复杂度

- **每个像素**：约 20 次浮点运算（归一化、畸变计算、双线性插值）
- **1280x720 图像**：约 92 万像素 × 20 = 1840 万次浮点运算

### 5.2 Mali-G610 性能估算

- **FP32 算力**：约 1 TFLOPS（理论峰值）
- **实际吞吐**：考虑内存带宽限制，约 100-200 GFLOPS
- **单帧延迟**：1840 万 / 100G = 0.18 ms（理论值）

### 5.3 瓶颈分析

- **内存带宽**：BGR→RGBA→OpenCL 图像→RGBA→BGR 多次拷贝
- **优化方向**：
  - 使用 DMA-BUF 实现 V4L2 → OpenCL 零拷贝
  - 直接在 NV12 格式上执行去畸变（避免颜色转换）
  - 使用 `clEnqueueAcquireExternalMemory` 共享缓冲区

### 5.4 当前实现的开销

- **颜色转换**：BGR ↔ RGBA（2 次 OpenCV 转换）
- **数据拷贝**：CPU → GPU → CPU（2 次 `clEnqueueWrite/Read`）
- **同步等待**：`clFinish` 阻塞 CPU

**预估实际延迟**：5-10 ms（含内存传输和同步）

## 6. 标定参数

### 6.1 模拟参数（默认值）

```cpp
CameraCalibration::CameraCalibration()
    : fx(800.0f), fy(800.0f)      // 焦距（像素）
    , cx(640.0f), cy(360.0f)      // 光心（1280x720 图像中心）
    , k1(-0.25f), k2(0.05f), k3(0.0f)  // 径向畸变
    , p1(0.001f), p2(-0.001f)     // 切向畸变
{}
```

### 6.2 真实标定流程（后续扩展）

1. **采集棋盘格图像**：10-20 张不同角度的棋盘格
2. **使用 OpenCV 标定**：`cv::calibrateCamera` 获取内参和畸变系数
3. **加载标定文件**：从 YAML/JSON 读取参数，替换模拟值

## 7. 文件结构

```
streamer_codev8.0/
├── main.cpp                  # 主程序（已集成去畸变调用）
├── opencl_undistort.h        # OpenCL 去畸变模块头文件
├── opencl_undistort.cpp      # OpenCL 去畸变模块实现
├── undistort.cl              # OpenCL 内核源代码
├── CMakeLists.txt            # 构建配置（已添加 OpenCL）
├── SafeQueue.h               # 线程安全队列
├── streamer.h / streamer.c   # MPP 编码 + RTMP 推流
├── thread_poll.h / .cpp      # NPU 线程池
├── yolov5s.h / .cpp          # YOLOv5s 推理
├── post_process.h / .cpp     # 后处理（NMS、解码）
├── mpp.h / mpp.c             # MPP 编码器封装
├── rtmp.h / rtmp.c           # RTMP 推流封装
└── 3rdparty/                 # 第三方库（RKNN、RGA）
```

## 8. 编译与运行

### 8.1 编译

```bash
cd streamer_codev8.0
mkdir build && cd build
cmake ..
make -j4
```

### 8.2 运行

```bash
./app /dev/video11 /dev/video20 rtmp://192.168.1.30:1935/live/app 1280 720 30
```

### 8.3 日志输出

```
[OpenCL] Platform 0: ARM Platform
[OpenCL] Selected ARM/Mali platform
[OpenCL] Device: Mali-G610
[OpenCL] Initialization complete (1280x720)
[Main] dual-camera detection started: /dev/video11 + /dev/video20 -> 1280x720
```

## 9. 后续优化方向

1. **零拷贝集成**：使用 DMA-BUF 共享 V4L2 和 OpenCL 缓冲区
2. **NV12 直接处理**：避免 BGR ↔ RGBA 颜色转换
3. **异步执行**：使用 OpenCL 事件机制，避免 `clFinish` 阻塞
4. **双路去畸变**：扩展到 USB 摄像头
5. **真实标定参数**：集成 OpenCV 标定流程
6. **性能监控**：添加 FPS 统计和延迟测量

## 10. 参考资料

- RK3588 OpenCL 知识归纳（飞书 Wiki）
- OpenCL 2.0 Specification
- Mali GPU OpenCL Developer Guide
- OpenCV Camera Calibration Documentation
