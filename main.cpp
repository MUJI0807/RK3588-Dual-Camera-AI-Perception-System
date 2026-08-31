// main.cpp  (V4L2 realtime input + auto stop after 120s)
// 只换输入源：从视频文件 -> V4L2 摄像头；其余 pipeline 不变

#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <iostream>
#include <string>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <map>
#include <memory>
#include <vector>

// ======= V4L2 headers (新增：用于摄像头实时采集) =======
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <poll.h>
#include <linux/videodev2.h>

#include "SafeQueue.h"
#include "yolov5s.h"
#include "thread_poll.h"
#include "streamer.h"

#include "rga.h"
#include "drmrga.h"
#include "im2d.h"
#include "RgaUtils.h"

#define ALIGN(x, a)   (((x)+(a)-1)&~((a)-1))
#define ALIGN64(x)    ALIGN(x, 64)

int g_width = 0;
int g_height = 0;
int g_hor_stride = 0;
int g_ver_stride = 0;

static inline int calc_nv12_mpp_size(int w, int h)
{
    int hs = ALIGN64(ALIGN(w, 16));
    int vs = ALIGN64(ALIGN(h, 16));
    return hs * vs * 3 / 2;
}

/**
 * @brief 使用 RGA 把一帧 BGR888/RGB888 转换为 NV12(YUV420SP)。
 */
void BGR_to_NV12_with_RGA(uint8_t *bgr, uint8_t *nv12, int width, int height)
{
    // 你的原代码保持不变
    printf("开始BGR到NV12的转换，图像尺寸: %dx%d\n", width, height);

    rga_buffer_handle_t bgr_handle, yuv_handle;

    memset(nv12, 0x00, calc_nv12_mpp_size(width, height));

    bgr_handle = importbuffer_virtualaddr(bgr, width * height * 3);
    yuv_handle = importbuffer_virtualaddr(nv12, calc_nv12_mpp_size(width, height));

    if(bgr_handle == 0 || yuv_handle == 0)
        printf("import va failed.\n");

    rga_buffer_t bgr_src = wrapbuffer_handle(bgr_handle, width, height, RK_FORMAT_RGB_888);
    rga_buffer_t yuv_src = wrapbuffer_handle(yuv_handle,
                                             ALIGN(width,16),
                                             ALIGN(height,16),
                                             RK_FORMAT_YCrCb_420_SP);

    int ret = imcvtcolor(bgr_src, yuv_src,
                         RK_FORMAT_RGB_888,
                         RK_FORMAT_YCrCb_420_SP);

    if(ret == IM_STATUS_SUCCESS)
        printf("BGR888 TO NV12 OK!\n");
    else
        printf("cvtColor error: %s\n", imStrError((IM_STATUS)ret));

    if(bgr_handle) releasebuffer_handle(bgr_handle);
    if(yuv_handle) releasebuffer_handle(yuv_handle);
}

const int MAX_CONCURRENT_FRAMES = 10;

struct FrameData {
    cv::Mat frame;
    int index;
};

SafeQueue<FrameData> g_readQueue(50);
SafeQueue<FrameData> g_writeQueue(50);
std::atomic<bool> g_readFinish(false);
std::atomic<bool> g_processFinish(false);

// ✅ 新增：120秒退出控制（只在 main.cpp 内部使用）
std::atomic<bool> g_stop(false);

// ============================================================================
// V4L2 摄像头采集封装：/dev/video0, YUYV, 640x480@30, MMAP + poll + DQBUF/QBUF
// ============================================================================
struct V4L2Buffer {
    void*  start = nullptr;
    size_t length = 0;
};

class V4L2Camera {
public:
    V4L2Camera() = default;
    ~V4L2Camera() { close_device(); }

    bool open_device(const char* dev, int w, int h, int fps) {
        dev_name_ = dev;
        width_ = w;
        height_ = h;
        fps_ = fps;

        fd_ = ::open(dev, O_RDWR | O_NONBLOCK, 0);
        if (fd_ < 0) {
            std::perror("[V4L2] open");
            return false;
        }

        // QUERYCAP
        v4l2_capability cap{};
        if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
            std::perror("[V4L2] VIDIOC_QUERYCAP");
            return false;
        }
        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
            !(cap.capabilities & V4L2_CAP_STREAMING)) {
            std::cerr << "[V4L2] device does not support capture/streaming.\n";
            return false;
        }

        // S_FMT : YUYV
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width_;
        fmt.fmt.pix.height = height_;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;

        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            std::perror("[V4L2] VIDIOC_S_FMT");
            return false;
        }

        // 驱动实际生效的尺寸（可能被调整）
        width_ = (int)fmt.fmt.pix.width;
        height_ = (int)fmt.fmt.pix.height;

        // S_PARM : fps（设备支持 timeperframe 时才会生效）
        v4l2_streamparm parm{};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_G_PARM, &parm) == 0) {
            if (parm.parm.capture.capability & V4L2_CAP_TIMEPERFRAME) {
                parm.parm.capture.timeperframe.numerator = 1;
                parm.parm.capture.timeperframe.denominator = fps_;
                if (ioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
                    std::perror("[V4L2] VIDIOC_S_PARM (fps)");
                    // fps 设不进去不致命
                }
            }
        }

        // REQBUFS
        v4l2_requestbuffers req{};
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
            std::perror("[V4L2] VIDIOC_REQBUFS");
            return false;
        }
        if (req.count < 2) {
            std::cerr << "[V4L2] insufficient buffer memory.\n";
            return false;
        }

        buffers_.resize(req.count);

        for (unsigned i = 0; i < req.count; ++i) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;

            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
                std::perror("[V4L2] VIDIOC_QUERYBUF");
                return false;
            }

            buffers_[i].length = buf.length;
            buffers_[i].start = mmap(nullptr, buf.length,
                                     PROT_READ | PROT_WRITE,
                                     MAP_SHARED,
                                     fd_, buf.m.offset);
            if (buffers_[i].start == MAP_FAILED) {
                std::perror("[V4L2] mmap");
                return false;
            }
        }

        // QBUF all
        for (unsigned i = 0; i < buffers_.size(); ++i) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
                std::perror("[V4L2] VIDIOC_QBUF");
                return false;
            }
        }

        // STREAMON
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            std::perror("[V4L2] VIDIOC_STREAMON");
            return false;
        }

        streaming_ = true;

        std::cerr << "[V4L2] opened " << dev_name_
                  << " (YUYV) " << width_ << "x" << height_
                  << " @" << fps_ << "fps\n";
        return true;
    }

    bool read_frame_yuyv(void*& out_ptr, size_t& out_bytes, int& out_buf_index, int timeout_ms = 2000) {
        if (fd_ < 0 || !streaming_) return false;

        pollfd pfd{};
        pfd.fd = fd_;
        pfd.events = POLLIN;

        int pret = poll(&pfd, 1, timeout_ms);
        if (pret == 0) return false; // timeout
        if (pret < 0) {
            if (errno == EINTR) return false;
            std::perror("[V4L2] poll");
            return false;
        }

        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) return false;
            std::perror("[V4L2] VIDIOC_DQBUF");
            return false;
        }

        out_buf_index = (int)buf.index;
        out_ptr = buffers_[buf.index].start;
        out_bytes = buf.bytesused;

        return true;
    }

    bool requeue(int buf_index) {
        if (fd_ < 0 || !streaming_) return false;
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = (unsigned)buf_index;
        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            std::perror("[V4L2] VIDIOC_QBUF (requeue)");
            return false;
        }
        return true;
    }

    int width() const { return width_; }
    int height() const { return height_; }
    int fps() const { return fps_; }

    void close_device() {
        if (fd_ < 0) return;

        if (streaming_) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(fd_, VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }

        for (auto &b : buffers_) {
            if (b.start && b.start != MAP_FAILED) {
                munmap(b.start, b.length);
            }
        }
        buffers_.clear();

        ::close(fd_);
        fd_ = -1;
    }

private:
    const char* dev_name_ = "/dev/video0";
    int fd_ = -1;
    int width_ = 640;
    int height_ = 480;
    int fps_ = 30;
    bool streaming_ = false;
    std::vector<V4L2Buffer> buffers_;
};

// ============================================================================
// 替换：读取线程（V4L2 摄像头取帧 -> 转 BGR -> 入队）
// ============================================================================
void readThreadFunc(V4L2Camera &cam)
{
    int idx = 0;
    const int w = cam.width();
    const int h = cam.height();

    while(!g_stop)
    {
        void* yuyv_ptr = nullptr;
        size_t bytesused = 0;
        int buf_index = -1;

        if (!cam.read_frame_yuyv(yuyv_ptr, bytesused, buf_index, 2000)) {
            continue;
        }

        // 注意：yuyv Mat 只是借用V4L2 buffer，不能跨越 requeue 使用
        cv::Mat yuyv(h, w, CV_8UC2, yuyv_ptr);
        cv::Mat bgr;
        cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUYV);

        cam.requeue(buf_index);

        FrameData data{ std::move(bgr), idx++ };
        g_readQueue.enqueue(data);
    }

    g_readFinish = true;
    std::cerr << "[ReadThread] finished.\n";
}

void aggregatorThreadFunc(ThreadPoll &npu_pool)
{
    int nextWriteIndex = 0;
    std::map<int, std::future<ProcessResult>> tasks_inflight;

    while(true)
    {
        FrameData inputFD;

        if(!g_readQueue.empty() &&
           tasks_inflight.size() < MAX_CONCURRENT_FRAMES)
        {
            if(g_readQueue.dequeue(inputFD)) {
                auto fut = npu_pool.submit_task_async(inputFD.index,
                                                      inputFD.frame);
                tasks_inflight[inputFD.index] = std::move(fut);
            }
        }

        auto it = tasks_inflight.find(nextWriteIndex);

        while(it != tasks_inflight.end())
        {
            auto status = it->second.wait_for(std::chrono::milliseconds(1));

            if(status == std::future_status::ready)
            {
                ProcessResult result = it->second.get();

                FrameData outputFD;
                outputFD.index = nextWriteIndex;
                outputFD.frame = result.processed_img.clone();
                g_writeQueue.enqueue(outputFD);

                tasks_inflight.erase(it);
                nextWriteIndex++;
                it = tasks_inflight.find(nextWriteIndex);
            }
            else break;
        }

        if(g_readFinish && g_readQueue.empty() && tasks_inflight.empty())
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    g_processFinish = true;
    std::cerr << "[AggregatorThread] finished.\n";
}

uint8_t *nv12_buffer = nullptr;

void writeThreadFunc(cv::VideoWriter &writer)
{
    int nv12_size = calc_nv12_mpp_size(g_width, g_height);
    nv12_buffer = (uint8_t*)malloc(nv12_size);

    while(true)
    {
        if(g_processFinish && g_writeQueue.empty())
            break;

        FrameData outputFD;
        if(!g_writeQueue.dequeue(outputFD)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if(!outputFD.frame.empty())
        {
            BGR_to_NV12_with_RGA(outputFD.frame.data,
                                 nv12_buffer,
                                 outputFD.frame.cols,
                                 outputFD.frame.rows);

            process_frame(nv12_buffer, nv12_size);
        }
    }

    free(nv12_buffer);
    std::cerr << "[WriteThread] finished.\n";
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    // 推流地址保持你的原配置
    std::string rtmpPath = "rtmp://192.168.1.30:1935/live/app";

    // 摄像头参数（来自你 v4l2-ctl 输出）
    const char* camDev = "/dev/video0";
    int width = 640;
    int height = 480;
    int fps = 30;

    V4L2Camera cam;
    if (!cam.open_device(camDev, width, height, fps)) {
        std::cerr << "[Main] Failed to open V4L2 camera.\n";
        return -1;
    }

    // 用驱动实际生效的尺寸
    width  = cam.width();
    height = cam.height();
    fps    = cam.fps();

    g_width = width;
    g_height = height;

    int bitrate = width * height / 8 * fps;

    init_streamer(width, height, fps, bitrate, rtmpPath.c_str());

    // 保持原结构：仍创建 VideoWriter（虽然不写文件）
    std::string outPath = "../output.avi";
    cv::VideoWriter writer(outPath,
                           cv::VideoWriter::fourcc('H','2','6','4'),
                           fps,
                           cv::Size(width,height));

    ThreadPoll npu_pool("../model/yolov5s.rknn", 3);

    std::thread tRead(readThreadFunc, std::ref(cam));
    std::thread tAggregator(aggregatorThreadFunc, std::ref(npu_pool));
    std::thread tWrite(writeThreadFunc, std::ref(writer));

    // ✅ 300秒后触发停止：让 readThread 退出，后面线程自然 drain 收尾
    auto run_start = std::chrono::steady_clock::now();
    while(!g_stop)
    {
        auto now = std::chrono::steady_clock::now();
        auto sec = std::chrono::duration_cast<std::chrono::seconds>(now - run_start).count();
        if(sec >= 300) {
            std::cerr << "[Main] 120 seconds reached, stopping...\n";
            g_stop = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    tRead.join();
    tAggregator.join();
    tWrite.join();

    close_streamer();
    cam.close_device();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "处理总用时：" << elapsed_ms.count() << " ms\n";
    std::cerr << "[Main] All done.\n";

    return 0;
}




/*
systemctl start nginx虚拟机端启用nginx服务器（确保已经配置好 nginx-rtmp-module，并且监听了1935端口）
要想执行推流，请确保已经启动了RTMP服务器（如nginx-rtmp-module），
1.虚拟机先开：
ffplay -fflags nobuffer -flags low_delay rtmp://127.0.0.1:1935/live/app
2.RK3588开发板再开：
ffmpeg -stream_loop -1 -re -i /home/cat/work/lesson/Chapter8/streamer_codev4.0/video.mp4 \
  -c:v libx264 -preset veryfast -tune zerolatency -g 50 -keyint_min 50 \
  -c:a aac -ar 44100 -b:a 128k \
  -f flv rtmp://192.168.1.30:1935/live/app
3.相应的端口的ffmpeg已经设定好，只需要更改推流视频名称即可
*/

/*
代码推流实现：
1.虚拟机端输入：ffplay -fflags nobuffer -flags low_delay rtmp://127.0.0.1:1935/live/app
2.RK3588开发板端运行：./app
3.效果：虚拟机端可以看到来自RK3588开发板的视频流
4.注意代码已经进行了修改，推理后的视频不再写入output.avi中，所以该文件的大小为0byte,推理后的视频会通过推流发送到虚拟机端
*/