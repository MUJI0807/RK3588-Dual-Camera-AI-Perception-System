
#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <iostream>
#include <string>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>

#include "SafeQueue.h"
#include "yolov5s.h"
#include "thread_poll.h"

//-----------------------------------
// 2) 定义传递帧的数据结构
//-----------------------------------
struct FrameData {
    cv::Mat frame;
    int index;
};

// 全局队列 & 全局标志
SafeQueue<FrameData> g_readQueue(100);
SafeQueue<FrameData> g_writeQueue(100);
std::atomic<bool> g_readFinish(false);
std::atomic<bool> g_processFinish(false);

Yolov5s yolo("/home/orangepi/Desktop/model/yolov5s.rknn",1);
detect_result_group_t group;

const int PROCESS_THREAD_NUM = 1;
ThreadPoll npu_pool("/home/orangepi/Desktop/model/yolov5s.rknn", 3); // 使用3个NPU核心
//-----------------------------------
// 3) 读线程：不断从视频文件读取放入 g_readQueue
//-----------------------------------
void readThreadFunc(cv::VideoCapture &cap) {
    int idx = 0;
    while(true) 
    {
        cv::Mat frame;
        if(!cap.read(frame)) 
        {
            // 读不到帧了（说明到视频末尾或出错）
            std::cerr << "[ReadThread] read failed or EOF, break.\n";
            break;
        }
        FrameData data{ frame.clone(), idx++ };
        g_readQueue.enqueue(data);
        std::cout<<"读队列："<<g_readQueue.size()<<endl;
    }
    // 通知后续不再有新帧
    g_readFinish = true;
    std::cerr << "[ReadThread] finished reading.\n";
}

//-----------------------------------
// 4) 处理线程：从 g_readQueue 取出图像做简单操作后放入 g_writeQueue
//-----------------------------------
void processThreadFunc(ThreadPoll& npu_pool) 
{
    int thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 1000;

    while(true) 
    {
        // 若读完且队列也空了，就退出
        if(g_readFinish && g_readQueue.empty()) 
        {
            cout<<"完成处理线程"<<endl;
            break;
        }

        FrameData inputFD;
        if(!g_readQueue.dequeue(inputFD)) continue;
        // 如果队列在等，就阻塞；也可能拿到 false 表示stop

        // 使用NPU池异步处理（关键修改点）
        auto future = npu_pool.submit_task_async(thread_id, inputFD.frame);


        // 等待处理结果
        auto result = future.get(); //获得关联操作的返回值
        if(result.success)
        {
            inputFD.frame = result.processed_img; // 获取处理后的图像
            g_writeQueue.enqueue(inputFD);
            std::cout<<"写队列："<<g_writeQueue.size()<<endl;
        } 
        else
        {
            std::cerr << "NPU处理失败: " << result.error_msg << endl;
        }
    }
    // 最后一个退出的线程设置完成标志
    static std::atomic<int> exit_count(0);
    if(++exit_count == PROCESS_THREAD_NUM)
    {
        g_processFinish = true;
    }
}
//-----------------------------------
// 5) 写线程：从 g_writeQueue 中取出图像写到文件
//-----------------------------------
void writeThreadFunc(cv::VideoWriter &writer) 
{
    while(true) 
    {
        // 若处理完且队列也空了，就退出
        if(g_processFinish && g_writeQueue.empty()) 
        {
            cout<<"完成写文件"<<endl;
            break;
        }

        FrameData outputFD;
        bool ok = g_writeQueue.dequeue(outputFD);
        if(!ok) 
        {
            break;
        }

        if(!outputFD.frame.empty()) 
        {
            writer.write(outputFD.frame);
        }
    }
    std::cerr << "[WriteThread] finished writing.\n";
}

//-----------------------------------
// 6) main 函数，把上面三步串起来
//-----------------------------------
int main(void) 
{
    auto start = std::chrono::high_resolution_clock::now();  // 记录开始时间
    std::string inPath  = "/home/orangepi/Desktop/video.mp4";
    std::string outPath = "output.avi";

    // 打开输入视频
    cv::VideoCapture cap(inPath);
    if(!cap.isOpened()) 
    {
        std::cerr << "Fail to open input video: " << inPath << "\n";
        return -1;
    }

    // 获取属性
    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if(fps < 1.0) 
    {
        fps = 25.0; // 避免某些视频元数据不完整
    }
    int fourcc = cv::VideoWriter::fourcc('M','J','P','G');

    // 打开输出视频
    cv::VideoWriter writer(outPath, fourcc, fps, cv::Size(width, height));
    if(!writer.isOpened()) 
    {
        std::cerr << "Fail to create output video: " << outPath << "\n";
        return -1;
    }

    // 启动3个线程
    std::thread tRead(readThreadFunc, std::ref(cap));

    // std::vector<std::thread> tProcess;
    // for(int i = 0; i < PROCESS_THREAD_NUM; ++i)
    // {
    //     tProcess.emplace_back(processThreadFunc, std::ref(npu_pool)); 
    // }
    
    std::thread tProcess(processThreadFunc, std::ref(npu_pool));

    std::thread tWrite(writeThreadFunc, std::ref(writer));

    // 等它们退出
    tRead.join();
    // for(int i = 0; i < PROCESS_THREAD_NUM; ++i)
    // {
    //     tProcess[i].join();
    // }
    tProcess.join();
    tWrite.join();

    // 给队列发 stop 信号（一般到这儿已经空了）
    g_readQueue.stop();
    g_writeQueue.stop();

    writer.release();

    auto end = std::chrono::high_resolution_clock::now();  // 记录结束时间
    auto elapsed = end - start;  // 计算耗时
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed); // 转换为毫秒
    cout << "共计花费时间为：" << elapsed_ms.count() << " 毫秒" << endl; // 使用 .count() 获取毫秒数值
    std::cerr << "[Main] All done.\n";
    return 0;
}