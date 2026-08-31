// main.cpp
#include <opencv2/opencv.hpp>  // OpenCV 头文件，包含核心功能和视频处理接口
#include <atomic>              // C++11 原子类型，用于线程安全的标志位
#include <thread>              // C++11 线程库，提供 std::thread 和相关功能
#include <iostream>
#include <string>
#include <chrono>              // C++11 时间库，用于计时和睡眠
#include <mutex>               // C++11 互斥锁，用于线程同步
#include <condition_variable>  // C++11 条件变量，用于线程间的等待和通知
#include <queue>               // C++ 标准队列容器
#include <future>              // C++11 异步任务和期约
#include <map>                 // C++ 标准映射容器，用于存储和管理键值对数据
#include <memory>              // C++11 智能指针，用于自动内存管理
#include <cstdlib>
#include <cstring>

#include "SafeQueue.h"         // 线程安全队列模板类
#include "yolov5s.h"
#include "thread_poll.h"
#include "streamer.h"          // RTMP 流媒体推送接口

#include "rga.h"               // RGA 头文件，包含 RGA 图像处理接口
#include "drmrga.h"            // RGA DRM 头文件，包含 DRM 相关接口
#include "im2d.h"              // RGA IM2D 头文件，包含图像转换接口
#include "RgaUtils.h"          // RGA 工具函数头文件

#include "perf_monitor.h"      // npu性能监控模块

#define ALIGN(x, a)   (((x)+(a)-1)&~((a)-1))   // 对齐宏：把 x 向上对齐到 a 的倍数（a 必须是 2 的幂）
#define ALIGN64(x)    ALIGN(x, 64)             // 向上对齐到64的倍数

int g_width = 0;             // 全局变量，图像宽度
int g_height = 0;            // 全局变量，图像高度
int g_hor_stride = 0;        // 全局变量，图像水平跨度
int g_ver_stride = 0;        // 全局变量，图像垂直跨度

static inline int calc_nv12_mpp_size(int w, int h)      // 计算 NV12 格式在 MPP 中的缓冲区大小
{
    int hs = ALIGN64(ALIGN(w, 16));            // 水平跨度，按16对齐后再按64对齐
    int vs = ALIGN64(ALIGN(h, 16));            // 垂直跨度，按16对齐后再按64对齐
    return hs * vs * 3 / 2;                    // NV12 格式大小计算公式
}

/**
 * @brief 使用 RGA（Rockchip Graphics Accelerator）把一帧 BGR888/RGB888 图像转换为 NV12(YUV420SP)。
 *
 * @param bgr    输入图像首地址（虚拟地址），像素格式期望为 24bit、3 字节/像素（BGR888 或 RGB888 的内存布局）。
 *               这里代码按 RK_FORMAT_RGB_888 来包装，所以调用者必须确保 bgr 的实际内存布局与 RK_FORMAT_RGB_888 匹配，
 *               否则颜色通道可能会颠倒（例如 BGR 与 RGB 顺序不一致会导致颜色异常）。
 * @param nv12   输出缓冲区首地址（虚拟地址），用于接收 NV12 数据（Y 平面 + UV 交错平面）。
 *               该函数会先把 nv12 清零，并写入转换结果。
 * @param width  输入图像宽度（像素）
 * @param height 输入图像高度（像素）
 *
 * @note 输出 NV12 的内存大小通常为 width*height*3/2，但这里通过 calc_nv12_mpp_size() 计算，
 *       可能包含对齐/stride 等额外空间（与 MPP/RGA 对齐规则相关）。
 */
void BGR_to_NV12_with_RGA(uint8_t *bgr, uint8_t *nv12, int width, int height)
{
    printf("开始BGR到NV12的转换，图像尺寸: %dx%d\n", width, height);

    rga_buffer_handle_t bgr_handle, yuv_handle;  // RGA 缓冲区句柄,bgr_handle 用于输入图像，yuv_handle用于输出图像

    // memset用于设置内存块的值，把 nv12 缓冲区的所有字节都设置为0x00，大小为calc_nv12_mpp_size(width, height)：NV12所需的字节数。
    memset(nv12, 0x00, calc_nv12_mpp_size(width, height));

    // 导入输入图像缓冲区，大小为 width*height*3 字节（BGR888/RGB888 的内存大小）
    bgr_handle = importbuffer_virtualaddr(bgr, width * height * 3);  // importbuffer_virtualaddr()获取的输入是虚拟地址和缓冲区大小
    // 导入输出图像缓冲区，大小为 calc_nv12_mpp_size(width, height) 字节（NV12 格式的内存大小）
    yuv_handle = importbuffer_virtualaddr(nv12, calc_nv12_mpp_size(width, height)); 

    if (bgr_handle == 0 || yuv_handle == 0) // 检查导入缓冲区句柄是否成功，如果失败则打印错误信息并释放已分配的缓冲区
        printf("import va failed.\n");

    // 包装输入图像缓冲区，指定格式为 RK_FORMAT_RGB_888
    rga_buffer_t bgr_src = wrapbuffer_handle(bgr_handle, width, height, RK_FORMAT_RGB_888); 
    // wrapbuffer_handle的输入是缓冲区句柄、图像宽度、高度和格式，返回一个 rga_buffer_t 结构体，表示输入图像缓冲区的信息

    // 包装输出图像缓冲区，指定格式为 RK_FORMAT_YCrCb_420_SP，获取的输入是缓冲区句柄、图像宽度、高度和格式，返回一个 rga_buffer_t 结构体，表示输出图像缓冲区的信息
    rga_buffer_t yuv_src = wrapbuffer_handle(yuv_handle,
                                             ALIGN(width, 16),
                                             ALIGN(height, 16),
                                             RK_FORMAT_YCrCb_420_SP);

    // 使用 RGA 的 imcvtcolor 接口进行颜色空间转换，输入包装输入图像缓冲区和输出图像缓冲区，要求输入格式为 RGB888，输出格式为 YUV420SP（NV12）
    int ret = imcvtcolor(bgr_src, yuv_src,
                         RK_FORMAT_RGB_888,
                         RK_FORMAT_YCrCb_420_SP);

    // 根据返回值判断转换是否成功，并打印相应的提示信息
    if (ret == IM_STATUS_SUCCESS)
        printf("BGR888 TO NV12 OK!\n");
    else
        printf("cvtColor error: %s\n", imStrError((IM_STATUS)ret));

    // 释放导入的缓冲区句柄，避免内存泄漏
    // releasebuffer_handle(*) 用于释放导入的缓冲区句柄，避免内存泄漏
    if (bgr_handle) releasebuffer_handle(bgr_handle);
    // 释放导入的缓冲区句柄，避免内存泄漏
    if (yuv_handle) releasebuffer_handle(yuv_handle);
}

const int MAX_CONCURRENT_FRAMES = 10;    // 最大并发处理帧数

// 定义用于在不同线程之间传递帧数据和索引的结构体
struct FrameData
{
    cv::Mat frame; // 存储视频帧的 OpenCV 矩阵
    int index;     // 帧的索引，用于保持处理顺序
};

// 定义线程安全队列，用于在不同线程之间传递帧数据
SafeQueue<FrameData> g_readQueue(50);   // 线程安全队列，用于存储从视频读取的帧数据，最大容量为50
SafeQueue<FrameData> g_writeQueue(50);  // 线程安全队列，用于存储处理完成的帧数据，最大容量为50

// 全局原子标志，用于指示读取线程和处理线程的完成状态
std::atomic<bool> g_readFinish(false);     // 原子标志，表示读取线程是否完成
std::atomic<bool> g_processFinish(false);  // 原子标志，表示处理线程是否完成

/**
 * @brief 读取线程函数：不断从 VideoCapture 中读取视频帧，并放入全局队列供后续线程处理。
 */
void readThreadFunc(cv::VideoCapture &cap) // cap是视频捕获对象，用于从视频文件或摄像头读取帧
{
    int idx = 0;
    while (true)
    {
        cv::Mat frame;
        if (!cap.read(frame)) // 从视频捕获对象中读取一帧，如果读取失败或到达文件末尾，则退出循环
        {
            std::cerr << "[ReadThread] read failed or EOF.\n";
            break;
        }

        FrameData data{ std::move(frame), idx++ }; // 创建 FrameData 对象，使用 std::move 将读取的帧移动到 data.frame 中，并设置帧索引为 idx，然后自增 idx
        g_readQueue.enqueue(data);  // 将读取到的帧数据放入线程安全队列 g_readQueue 中，供后续处理线程使用
    }
    // 设置读取完成标志，通知其他线程读取已完成
    g_readFinish = true;
    std::cerr << "[ReadThread] finished.\n";
}

/**
 * @brief 聚合线程：负责把读取到的帧送进 NPU 线程池异步处理，并按帧序号有序输出到写队列。
 */
void aggregatorThreadFunc(ThreadPoll &npu_pool)
{
    int nextWriteIndex = 0; // 记录下一个需要写入的帧索引，确保按顺序输出处理结果
    std::map<int, std::future<ProcessResult>> tasks_inflight;  // 用 Frame ID 作为索引，把所有尚未完成的异步 NPU 推理任务保存起来，并利用 map 的自动排序机制实现视频帧保序输出。

    // 循环处理读取到的帧数据，直到读取完成且所有任务处理完毕
    while (true)
    {
        FrameData inputFD; // 定义一个 FrameData 对象，用于存储从读取队列中取出的帧数据

        // 控制并发数量，避免同时挂太多任务
        if (!g_readQueue.empty() &&
            tasks_inflight.size() < MAX_CONCURRENT_FRAMES) // g_readQueue 不为空且当前正在处理的任务数小于最大并发帧数时，才从读取队列中取出一帧数据进行处理
        {  
            if (g_readQueue.dequeue(inputFD))              // 从读取队列中取出一帧数据，如果成功则提交给 NPU 线程池进行异步处理
            {
                auto fut = npu_pool.submit_task_async(inputFD.index, inputFD.frame);  // npu_pool.submit_task_async 提交异步任务给 NPU 线程池，返回一个 std::future 对象，用于获取处理结果
                tasks_inflight[inputFD.index] = std::move(fut); // 将异步任务的 future 对象存入 tasks_inflight map 中，以帧索引为键，方便后续按顺序获取处理结果
            }
        }
        // 检查 tasks_inflight 中是否有任务完成，如果有则按顺序输出到写队列
        auto it = tasks_inflight.find(nextWriteIndex);

        // 保证按序输出
        while (it != tasks_inflight.end())
        {
            auto status = it->second.wait_for(std::chrono::milliseconds(1)); // 等待异步任务完成，最多等待 1 毫秒，避免阻塞过久

            // 如果任务完成，则获取处理结果并放入写队列，否则跳出循环等待下一次检查
            if (status == std::future_status::ready)   
            {
                ProcessResult result = it->second.get(); // ProcessResult是一个结构体，包含处理后的图像和检测结果等信息

                FrameData outputFD;                     // 定义一个 FrameData 对象，用于存储处理完成的帧数据
                outputFD.index = nextWriteIndex;
                outputFD.frame = result.processed_img.clone();  // 带检测框的处理后图像
                g_writeQueue.enqueue(outputFD);        // 将处理完成的帧数据放入写队列 g_writeQueue 中，供写线程进行本地保存和推流

                tasks_inflight.erase(it);             // 从 tasks_inflight map 中移除已完成的任务，释放资源
                nextWriteIndex++;                     // 更新下一个需要写入的帧索引，确保按顺序输出处理结果
                it = tasks_inflight.find(nextWriteIndex); // 继续查找下一个需要写入的帧索引，确保按顺序输出处理结果
            }
            else
            {
                break;
            }
        }
        // 如果读取完成且所有任务处理完毕，则退出循环
        if (g_readFinish && g_readQueue.empty() && tasks_inflight.empty())
            break;
        // 避免 CPU 占用过高，稍微休眠一下
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // 设置处理完成标志，通知写线程所有帧都已处理完毕
    g_processFinish = true;
    std::cerr << "[AggregatorThread] finished.\n";
}

uint8_t *nv12_buffer = nullptr;   // 全局缓冲区指针，用于存储转换后的 NV12 数据，供推流线程使用

/**
 * @brief 写入线程：
 * 1. 从写队列中取出处理完成的帧
 * 2. 先把 BGR 帧写入本地视频文件
 * 3. 再把 BGR 转成 NV12
 * 4. 调用 process_frame() 推流
 */
void writeThreadFunc(cv::VideoWriter &writer)
{
    int nv12_size = calc_nv12_mpp_size(g_width, g_height);  // 计算 NV12 缓冲区大小，确保足够存储转换后的 NV12 数据

    nv12_buffer = (uint8_t *)malloc(nv12_size);  // 为 NV12 缓冲区分配内存，大小为 nv12_size 字节
    
    // 检查内存分配是否成功，如果失败则输出错误信息并返回
    if (!nv12_buffer)
    {
        std::cerr << "[WriteThread] malloc nv12_buffer failed.\n";
        return;
    }

    // 循环处理写队列中的帧数据，直到处理完成标志被设置且写队列为空
    while (true)
    {
        if (g_processFinish && g_writeQueue.empty()) // 如果处理完成标志被设置且写队列为空，则退出循环
            break;

        FrameData outputFD;
        if (!g_writeQueue.dequeue(outputFD))         // 从写队列中取出处理完成的帧数据
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 如果 outputFD.frame 不为空，则进行本地保存和推流处理
        if (!outputFD.frame.empty())
        {
            // ==================== 新增功能1：本地保存处理后的视频 ====================
            // outputFD.frame 是已经完成检测绘制后的 BGR 图像，直接写入本地视频即可
            if (writer.isOpened())
            {
                writer.write(outputFD.frame);
            }
            else
            {
                std::cerr << "[WriteThread] VideoWriter is not opened, skip local save.\n";
            }

            // ==================== 原有功能：继续推流 ====================
            BGR_to_NV12_with_RGA(outputFD.frame.data,
                                 nv12_buffer,
                                 outputFD.frame.cols,
                                 outputFD.frame.rows);

            process_frame(nv12_buffer, nv12_size);
        }
    }

    // 释放 NV12 缓冲区内存，避免内存泄漏
    free(nv12_buffer);
    nv12_buffer = nullptr;
    std::cerr << "[WriteThread] finished.\n";
}

int main()
{
    
    PerfMonitor::instance().start("../perf_log.csv", 500);  // 启动性能监控，日志保存到 perf_log.csv，每500ms记录一次性能数据
    auto start = std::chrono::high_resolution_clock::now();

    // 输入路径、本地输出路径和 RTMP 推流地址
    std::string inPath   = "../video.mp4";
    std::string outPath  = "../output.mp4";  // 本地保存处理后的视频
    std::string rtmpPath = "rtmp://10.24.10.119/live/app";

    // 打开输入视频文件
    cv::VideoCapture cap(inPath);
    if (!cap.isOpened())  // 如果输入视频文件无法打开，则输出错误信息并返回 -1
    {
        std::cerr << "[Main] Failed to open input video: " << inPath << "\n";
        return -1;
    }

    // 获取视频的宽度和高度，并设置全局变量 g_width 和 g_height
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 设置全局变量 g_width 和 g_height，用于后续的 NV12 转换和推流
    g_width = width;
    g_height = height;

    // 获取视频的帧率，并计算比特率
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps < 1.0) fps = 25.0;

    int bitrate = static_cast<int>(width * height / 8 * fps);

    // 初始化 RTMP 推流器
    if (init_streamer(width, height, static_cast<int>(fps), bitrate, rtmpPath.c_str()) != 0)
    {
        std::cerr << "[Main] init_streamer failed.\n";
        return -1;
    }

    // 创建本地视频输出对象
    // 这里优先使用 mp4v，更适合 OpenCV 本地保存；如果你的环境支持 H264，也可以改回 H264
    cv::VideoWriter writer(outPath,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps,
                           cv::Size(width, height));

    // 检查 VideoWriter 是否成功打开，如果失败则输出错误信息并关闭推流器
    if (!writer.isOpened())
    {
        std::cerr << "[Main] Failed to open VideoWriter: " << outPath << "\n";
        close_streamer();
        return -1;
    }

    // 创建 NPU 线程池，指定模型路径和线程数
    ThreadPoll npu_pool("../model/yolov5s.rknn", 3);

    // 启动读取线程、聚合线程和写入线程
    std::thread tRead(readThreadFunc, std::ref(cap));
    std::thread tAggregator(aggregatorThreadFunc, std::ref(npu_pool));
    std::thread tWrite(writeThreadFunc, std::ref(writer)); 

    // 等待所有线程完成
    tRead.join();
    tAggregator.join();
    tWrite.join();

    // 释放本地视频写入器
    writer.release();

    // 关闭推流器
    close_streamer();

    // 关闭输入视频
    cap.release();


    // 停止性能监控并记录总用时
    PerfMonitor::instance().stop();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "处理总用时：" << elapsed_ms.count() << " ms\n";
    std::cout << "本地保存文件：" << outPath << "\n";
    std::cerr << "[Main] All done.\n";


    return 0;
}

/*
systemctl start nginx
虚拟机端启用nginx服务器（确保已经配置好 nginx-rtmp-module，并且监听了1935端口）

要想执行推流，请确保已经启动了RTMP服务器（如nginx-rtmp-module）：
1. 虚拟机先开：
ffplay -fflags nobuffer -flags low_delay rtmp://127.0.0.1:1935/live/app

2. RK3588开发板再开：
./app

3. 程序运行后会同时完成两件事：
   - 将处理后的视频通过 RTMP 推到服务器
   - 将处理后的视频保存到本地 ../output.mp4

4. 监控NPU：watch -n 0.5 "sudo cat /sys/kernel/debug/rknpu/load"
   监控CPU：htop
*/