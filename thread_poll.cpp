#include "thread_poll.h"

#include <atomic>
#include <iostream>
#include <future>

// ================================ 调度参数 ================================
// 最大等待任务数：防止推理队列无限堆积导致延迟越来越大
static const size_t MAX_PENDING_TASKS = 4;

// 日志降频：每隔多少次打印一次状态，避免每帧 printf 拖慢系统
static const int LOG_INTERVAL = 100;

static std::atomic<int> g_submit_count{0};
static std::atomic<int> g_worker_count{0};
static std::atomic<int> g_drop_count{0};

// 生成一个已经完成的 future，用于队列满时安全丢帧
static std::future<ProcessResult> make_drop_future(int index)
{
    std::promise<ProcessResult> promise;
    ProcessResult result;

    result.success = false;
    result.error_msg = "drop frame because task queue is full, frame index = " + std::to_string(index);

    promise.set_value(result);
    return promise.get_future();
}

ThreadPoll::ThreadPoll(const char* model_path, int num_threads)
{
    run_flag = true;
    init(model_path, num_threads);
}

ThreadPoll::~ThreadPoll()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        std::cout << "[ThreadPoll] Remaining tasks before destroy: "
                  << tasks.size() << std::endl;
    }

    run_flag = false;
    condition.notify_all();

    for (auto& t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    std::cout << "[ThreadPoll] destroyed. submit="
              << g_submit_count.load()
              << ", worker="
              << g_worker_count.load()
              << ", drop="
              << g_drop_count.load()
              << std::endl;
}

void ThreadPoll::init(const char* model_path, int num_threads)
{
    if (num_threads <= 0)
        num_threads = 1;

    for (int i = 0; i < num_threads; i++)
    {
        auto yolo = std::make_shared<Yolov5s>(model_path, i % 3);
        yolo_group.emplace_back(yolo);
    }

    for (int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(&ThreadPoll::worker, this, i);
    }
}

void ThreadPoll::worker(int id)
{
    std::shared_ptr<Yolov5s> yolo = yolo_group[id];

    std::cout << "[ThreadPoll] worker start, id=" << id << std::endl;

    while (run_flag)
    {
        std::packaged_task<ProcessResult()> current_task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            condition.wait(lock, [this]
            {
                return (!tasks.empty() || !run_flag);
            });

            if (!run_flag)
            {
                break;
            }

            if (tasks.empty())
            {
                continue;
            }

            current_task = std::move(tasks.front());
            tasks.pop();
        }

        if (current_task.valid())
        {
            int cnt = ++g_worker_count;

            if (cnt % LOG_INTERVAL == 0)
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                std::cout << "[ThreadPoll] worker=" << id
                          << ", executed=" << cnt
                          << ", pending=" << tasks.size()
                          << ", dropped=" << g_drop_count.load()
                          << std::endl;
            }

            current_task();
        }
    }

    std::cout << "[ThreadPoll] worker exit, id=" << id << std::endl;
}

std::future<ProcessResult> ThreadPoll::submit_task_async(int index, cv::Mat img)
{
    int submit_cnt = ++g_submit_count;

    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // 队列满时直接丢弃当前新帧，避免延迟继续累积。
        // 这样不会破坏已经提交任务对应的 future，主线程不会因为 broken_promise 崩溃。
        if (tasks.size() >= MAX_PENDING_TASKS)
        {
            int drop_cnt = ++g_drop_count;

            if (drop_cnt % LOG_INTERVAL == 0)
            {
                std::cout << "[ThreadPoll] drop frame, index=" << index
                          << ", pending=" << tasks.size()
                          << ", total_drop=" << drop_cnt
                          << std::endl;
            }

            return make_drop_future(index);
        }
    }

    std::packaged_task<ProcessResult()> task([this, index, img]() mutable
    {
        ProcessResult result;

        try
        {
            auto yolo = yolo_group[index % yolo_group.size()];

            detect_result_group_t detections;
            yolo->inference_image(img, detections);

            result.processed_img = img;
            result.detection_results = detections;
            result.success = true;
        }
        catch (const std::exception& e)
        {
            result.error_msg = e.what();
            result.success = false;
        }
        catch (...)
        {
            result.error_msg = "unknown exception in ThreadPoll task";
            result.success = false;
        }

        return result;
    });

    std::future<ProcessResult> future = task.get_future();

    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.emplace(std::move(task));

        if (submit_cnt % LOG_INTERVAL == 0)
        {
            std::cout << "[ThreadPoll] submit=" << submit_cnt
                      << ", pending=" << tasks.size()
                      << ", dropped=" << g_drop_count.load()
                      << std::endl;
        }
    }

    condition.notify_one();
    return future;
}