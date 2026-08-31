#include "thread_poll.h"

ThreadPoll::ThreadPoll(const char* model_path, int num_threads)
{
    // 这里可以做一些通用初始化，比如 run_flag=true
    run_flag = true;
    // 初始化：加载模型，启动线程
    init(model_path, num_threads);
}

ThreadPoll::~ThreadPoll()
{
    // 在ThreadPoll析构函数添加
    // 修复：tasks 会被 worker 线程并发 pop，读取 size 必须先加锁，否则是数据竞争
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        std::cout << "Remaining tasks: " << tasks.size() << std::endl;
    }

    // 通知线程退出
    run_flag = false;
    // 唤醒所有等待条件，让 worker() 能跳出循环
    condition.notify_all();

    // 等待线程结束
    for(auto& t : threads)
    {
        if(t.joinable())
        {
            t.join();
        }
    }
    // 这里你也可以释放模型等资源
    std::cout << "ThreadPoll destroyed.\n";
}

void ThreadPoll::init(const char* model_path, int num_threads)
{

    if(num_threads <= 0) num_threads = 1; // 保底
    // 比如按照 num_threads 个 Yolov5s
    // 也可以根据需求只创建几个再共享
    for(int i = 0; i < num_threads; i++)
    {
        auto yolo = std::make_shared<Yolov5s>(model_path, i % 3);
        yolo_group.emplace_back(yolo);
        yolo_mutexes.emplace_back(std::make_shared<std::mutex>());
    }

    // 启动 num_threads 个工作线程
    for(int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(&ThreadPoll::worker, this, i);
    }
}

// worker：只对 tasks 这个队列做等待、取出、执行
void ThreadPoll::worker(int id)
{
    // 取到专属的yolo实例
    std::shared_ptr<Yolov5s> yolo = yolo_group[id];
    std::cout << "worker线程启动, id=" << id << "\n";
    while(run_flag)
    {
        std::packaged_task<ProcessResult()> current_task;
        {
            // 阻塞等待队列内有任务，或等到退出信号
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [this]
            {
                return (!tasks.empty() || !run_flag);
            });

            if(!run_flag)
            {
                // 收到退出命令
                std::cout << "worker " << id << " 下班！\n";
                break;
            }

            // 从 tasks 队列取出一个打包的任务
            current_task = std::move(tasks.front());
            tasks.pop();
        }

        // 离开大锁区后执行真正的推理任务
        if(current_task.valid())
        {
            // 如果任务有效，就调用 operator() 执行
            current_task();
        }
    }
    // 在worker线程退出时添加
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        std::cout << "Worker " << id << " exited, remaining tasks: " << tasks.size() << std::endl;
    }
}

// 新的方法：往 tasks 里塞任务，并用 std::future<ProcessResult> 返回结果
std::future<ProcessResult> ThreadPoll::submit_task_async(int index, cv::Mat img)
{
    // 1) 打包任务为 std::packaged_task<ProcessResult()>
    //    其中捕获 [this, index, img] 就可以在lambda里使用
    std::packaged_task<ProcessResult()> task([this, index, img]()
    {
        ProcessResult result;
        try
        {
            // 从 yolo_group 里选一个做推理，这里简单选 index % yolo_group.size()
            // 你也可以做负载均衡
            const size_t model_index = static_cast<size_t>(index) % yolo_group.size();
            auto yolo = yolo_group[model_index];
            std::lock_guard<std::mutex> model_lock(*yolo_mutexes[model_index]);

            // 推理
            detect_result_group_t detections;
            yolo->inference_image(img, detections);
            yolo->draw_result(const_cast<cv::Mat&>(img), detections);

            // 填充结果
            result.processed_img = img;
            result.detection_results = detections;
            result.success = true;
        }
        catch(const std::exception& e)
        {
            result.error_msg = e.what();
            result.success = false;
        }
        return result;
    });

    // 2) 先拿到 future，然后把 task 放进队列
    std::future<ProcessResult> future = task.get_future();
    {
        // 加锁操作队列
        std::unique_lock<std::mutex> lock(queue_mutex);

        // 把打包好的任务放到队列
        tasks.emplace(std::move(task));
    }
    // 3) 唤醒一个worker线程来执行
    condition.notify_one();
    // 4) 返回 future，后续可以 .get() 拿到结果
    return future;
}
