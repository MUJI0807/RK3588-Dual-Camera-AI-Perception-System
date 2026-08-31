#ifndef THREADPOLL_H
#define THREADPOLL_H

#include <mutex>
#include <thread>
#include <condition_variable>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <queue>
#include <map>

#include "yolov5s.h"
#include <utility>
#include <exception>
#include <future>

using namespace std;
using namespace cv;

struct ProcessResult {
    cv::Mat processed_img;
    detect_result_group_t detection_results;
    bool success = false;
    std::string error_msg;
};
class ThreadPoll
{
public:
    // 构造：加载模型、创建指定数量的线程
    ThreadPoll(const char* model_path, int num_threads);
    // 析构：清理模型和工作线程
    ~ThreadPoll();

    // 提交异步推理任务（新的正确用法），返回 future 来获取结果
    std::future<ProcessResult> submit_task_async(int index, cv::Mat img);

private:
    // 工作线程函数：不断从 tasks 队列里取 std::packaged_task 并执行
    void worker(int id);

    // 初始化：创建 YOLO 实例 + 启动线程
    void init(const char* model_path, int num_threads);

private:
    // 下面这两个是新逻辑用到的核心队列与锁/信号量
    std::queue<std::packaged_task<ProcessResult()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;

    // 线程池线程
    std::vector<std::thread> threads;
    std::atomic<bool> run_flag{true};

    // 一个或多个 YOLO 模型实例，例如多线程使用
    std::vector<std::shared_ptr<Yolov5s>> yolo_group;
};
#endif