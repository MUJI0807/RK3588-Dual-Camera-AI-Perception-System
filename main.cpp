#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <future>
#include <iostream>
#include <iomanip>
#include <memory>
#include <poll.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#include <linux/videodev2.h>

#include "SafeQueue.h"
#include "streamer.h"
#include "thread_poll.h"

#include "RgaUtils.h"
#include "drmrga.h"
#include "im2d.h"
#include "rga.h"

#include "opencl_undistort.h"
#include "qwen_analyzer.h"

#define ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#define ALIGN64(x) ALIGN(x, 64)

namespace {

constexpr int kCameraCount = 2;
constexpr int kMaxInflightFrames = 10;
constexpr int kReadQueueCapacity = 12;
constexpr int kWriteQueueCapacity = 4;

int g_output_width = 0;
int g_output_height = 0;

std::atomic<bool> g_stop{false};
std::atomic<bool> g_read_finish{false};
std::atomic<bool> g_process_finish{false};
std::atomic<int> g_active_readers{0};
std::atomic<uint64_t> g_task_index{0};
std::atomic<uint64_t> g_capture_frames[kCameraCount]{};
std::atomic<uint64_t> g_inference_frames[kCameraCount]{};
std::atomic<uint64_t> g_capture_drops[kCameraCount]{};
std::atomic<uint64_t> g_composite_drops{0};
std::atomic<uint64_t> g_streamed_frames{0};
std::atomic<uint64_t> g_stream_errors{0};

// OpenCL 去畸变上下文（仅用于 MIPI 摄像头）
OpenCLUndistortContext* g_undistort_context = nullptr;

struct FrameData {
    cv::Mat frame;
    uint64_t index = 0;
    uint64_t camera_sequence = 0;
    int camera_id = -1;
};

SafeQueue<FrameData> g_read_queue(kReadQueueCapacity);
SafeQueue<FrameData> g_write_queue(kWriteQueueCapacity);

#ifdef QWEN_ENABLED
// ============================================================================
//  Qwen3-VL 旁路风险分析（仅在 QWEN_ENABLED 下编译）
//  - 独立线程串行推理（RKLLM 单实例不可并发）
//  - "最新帧优先"单槽：Qwen 推理慢（单帧秒级），只保留最新一帧，旧的直接覆盖
//  - 事件驱动 + 定时调度：在 compositor 合成后触发投喂
// ============================================================================
QwenAnalyzer g_qwen;
std::atomic<bool> g_qwen_stop{false};

struct QwenSlot {
    std::mutex m;
    cv::Mat frame;
    std::string detection_context;
    bool has = false;
};
QwenSlot g_qwen_slot;

// 把一路检测结果文本化，作为注入给 Qwen 的底层感知上下文
std::string detection_group_to_text(const detect_result_group_t& group, const char* camera_name)
{
    std::string out;
    for (int i = 0; i < group.box_count; ++i) {
        const detect_result_t& r = group.result[i];
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "%s %s conf=%.2f box=[%d,%d,%d,%d]\n",
                      camera_name, r.label, r.box_conf,
                      r.box.xmin, r.box.ymin, r.box.xmax, r.box.ymax);
        out += buf;
    }
    return out;
}

// 投喂一帧给 Qwen 分析（最新帧覆盖旧帧，非阻塞）
void qwen_feed(const cv::Mat& frame, const std::string& ctx)
{
    std::lock_guard<std::mutex> lock(g_qwen_slot.m);
    g_qwen_slot.frame = frame.clone();
    g_qwen_slot.detection_context = ctx;
    g_qwen_slot.has = true;
}

// Qwen 分析线程：消费最新帧，输出风险 JSON
void qwen_analysis_thread()
{
    while (!g_qwen_stop.load()) {
        cv::Mat frame;
        std::string ctx;
        {
            std::lock_guard<std::mutex> lock(g_qwen_slot.m);
            if (g_qwen_slot.has) {
                frame = g_qwen_slot.frame.clone();
                ctx = g_qwen_slot.detection_context;
                g_qwen_slot.has = false;
            }
        }

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        std::string json;
        if (g_qwen.analyze(frame, ctx, json, 8000)) {
            std::cerr << "[Qwen] risk analysis:\n" << json << "\n";
        } else {
            std::cerr << "[Qwen] analyze failed or timeout\n";
        }
    }
}
#endif // QWEN_ENABLED

int calc_nv12_mpp_size(int width, int height)
{
    const int horizontal_stride = ALIGN64(ALIGN(width, 16));
    const int vertical_stride = ALIGN64(ALIGN(height, 16));
    return horizontal_stride * vertical_stride * 3 / 2;
}

void bgr_to_nv12_with_rga(const cv::Mat& bgr, uint8_t* nv12)
{
    const int width = bgr.cols;
    const int height = bgr.rows;
    std::memset(nv12, 0, calc_nv12_mpp_size(width, height));

    const rga_buffer_handle_t bgr_handle =
        importbuffer_virtualaddr(bgr.data, width * height * 3);
    const rga_buffer_handle_t yuv_handle =
        importbuffer_virtualaddr(nv12, calc_nv12_mpp_size(width, height));

    if (bgr_handle == 0 || yuv_handle == 0) {
        std::cerr << "[RGA] importbuffer_virtualaddr failed\n";
        if (bgr_handle) releasebuffer_handle(bgr_handle);
        if (yuv_handle) releasebuffer_handle(yuv_handle);
        return;
    }

    rga_buffer_t bgr_source =
        wrapbuffer_handle(bgr_handle, width, height, RK_FORMAT_BGR_888);
    rga_buffer_t nv12_destination =
        wrapbuffer_handle(yuv_handle,
                          ALIGN(width, 16),
                          ALIGN(height, 16),
                          RK_FORMAT_YCbCr_420_SP);

    const int ret = imcvtcolor(bgr_source,
                               nv12_destination,
                               RK_FORMAT_BGR_888,
                               RK_FORMAT_YCbCr_420_SP);
    if (ret != IM_STATUS_SUCCESS) {
        std::cerr << "[RGA] BGR to NV12 failed: "
                  << imStrError(static_cast<IM_STATUS>(ret)) << '\n';
    }

    releasebuffer_handle(bgr_handle);
    releasebuffer_handle(yuv_handle);
}

std::string fourcc_to_string(uint32_t value)
{
    std::string text(4, ' ');
    text[0] = static_cast<char>(value & 0xff);
    text[1] = static_cast<char>((value >> 8) & 0xff);
    text[2] = static_cast<char>((value >> 16) & 0xff);
    text[3] = static_cast<char>((value >> 24) & 0xff);
    return text;
}

struct MappedPlane {
    void* start = nullptr;
    size_t length = 0;
};

struct V4L2Buffer {
    std::vector<MappedPlane> planes;
};

class V4L2Camera {
public:
    V4L2Camera() = default;
    ~V4L2Camera() { close_device(); }

    V4L2Camera(const V4L2Camera&) = delete;
    V4L2Camera& operator=(const V4L2Camera&) = delete;

    bool open_device(const std::string& device,
                     int requested_width,
                     int requested_height,
                     int fallback_fps,
                     const std::vector<uint32_t>& preferred_formats)
    {
        device_name_ = device;
        width_ = requested_width;
        height_ = requested_height;
        // 只作为驱动没有报告帧率时的回退值，不向摄像头强制设置帧率。
        fps_ = fallback_fps;

        fd_ = ::open(device.c_str(), O_RDWR | O_NONBLOCK, 0);
        if (fd_ < 0) {
            std::perror(("[V4L2 " + device + "] open").c_str());
            return false;
        }

        v4l2_capability capability{};
        if (xioctl(VIDIOC_QUERYCAP, &capability) < 0) {
            std::perror(("[V4L2 " + device + "] VIDIOC_QUERYCAP").c_str());
            return false;
        }

        uint32_t capabilities = capability.capabilities;
        if (capabilities & V4L2_CAP_DEVICE_CAPS) {
            capabilities = capability.device_caps;
        }

        if (!(capabilities & V4L2_CAP_STREAMING)) {
            std::cerr << "[V4L2 " << device << "] streaming is not supported\n";
            return false;
        }

        if (capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) {
            buffer_type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            multi_planar_ = true;
        } else if (capabilities & V4L2_CAP_VIDEO_CAPTURE) {
            buffer_type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            multi_planar_ = false;
        } else {
            std::cerr << "[V4L2 " << device << "] capture is not supported\n";
            return false;
        }

        const std::vector<uint32_t> supported_formats = enumerate_formats();
        pixel_format_ = choose_format(preferred_formats, supported_formats);
        if (pixel_format_ == 0) {
            std::cerr << "[V4L2 " << device << "] no supported input format. Device offers:";
            for (uint32_t format : supported_formats) {
                std::cerr << ' ' << fourcc_to_string(format);
            }
            std::cerr << '\n';
            return false;
        }

        v4l2_format format{};
        format.type = buffer_type_;
        if (multi_planar_) {
            format.fmt.pix_mp.width = width_;
            format.fmt.pix_mp.height = height_;
            format.fmt.pix_mp.pixelformat = pixel_format_;
            format.fmt.pix_mp.field = V4L2_FIELD_ANY;
        } else {
            format.fmt.pix.width = width_;
            format.fmt.pix.height = height_;
            format.fmt.pix.pixelformat = pixel_format_;
            format.fmt.pix.field = V4L2_FIELD_ANY;
        }

        if (xioctl(VIDIOC_S_FMT, &format) < 0) {
            std::perror(("[V4L2 " + device + "] VIDIOC_S_FMT").c_str());
            return false;
        }

        if (multi_planar_) {
            width_ = static_cast<int>(format.fmt.pix_mp.width);
            height_ = static_cast<int>(format.fmt.pix_mp.height);
            pixel_format_ = format.fmt.pix_mp.pixelformat;
            plane_count_ = std::max<unsigned>(1, format.fmt.pix_mp.num_planes);
            bytes_per_line_.resize(plane_count_);
            for (unsigned i = 0; i < plane_count_; ++i) {
                bytes_per_line_[i] = format.fmt.pix_mp.plane_fmt[i].bytesperline;
            }
        } else {
            width_ = static_cast<int>(format.fmt.pix.width);
            height_ = static_cast<int>(format.fmt.pix.height);
            pixel_format_ = format.fmt.pix.pixelformat;
            plane_count_ = 1;
            bytes_per_line_ = {format.fmt.pix.bytesperline};
        }

        if (!can_convert(pixel_format_)) {
            std::cerr << "[V4L2 " << device << "] driver selected unsupported format "
                      << fourcc_to_string(pixel_format_) << '\n';
            return false;
        }

        read_frame_rate();
        if (!create_mmap_buffers()) return false;

        v4l2_buf_type type = buffer_type_;
        if (xioctl(VIDIOC_STREAMON, &type) < 0) {
            std::perror(("[V4L2 " + device + "] VIDIOC_STREAMON").c_str());
            return false;
        }
        streaming_ = true;

        std::cerr << "[V4L2] opened " << device_name_ << " ("
                  << fourcc_to_string(pixel_format_) << ") "
                  << width_ << 'x' << height_ << " @" << fps_ << "fps"
                  << (multi_planar_ ? " mplane" : " single-plane") << '\n';
        return true;
    }

    bool read_frame_bgr(cv::Mat& output, int timeout_ms = 2000)
    {
        output.release();
        if (fd_ < 0 || !streaming_) return false;

        pollfd descriptor{};
        descriptor.fd = fd_;
        descriptor.events = POLLIN;
        const int poll_result = ::poll(&descriptor, 1, timeout_ms);
        if (poll_result == 0) return false;
        if (poll_result < 0) {
            if (errno != EINTR) std::perror("[V4L2] poll");
            return false;
        }

        v4l2_buffer buffer{};
        std::array<v4l2_plane, VIDEO_MAX_PLANES> planes{};
        buffer.type = buffer_type_;
        buffer.memory = V4L2_MEMORY_MMAP;
        if (multi_planar_) {
            buffer.m.planes = planes.data();
            buffer.length = plane_count_;
        }

        if (xioctl(VIDIOC_DQBUF, &buffer) < 0) {
            if (errno != EAGAIN && errno != EINTR) {
                std::perror("[V4L2] VIDIOC_DQBUF");
            }
            return false;
        }

        bool converted = false;
        try {
            converted = convert_buffer(buffer, planes, output);
        } catch (const cv::Exception& error) {
            std::cerr << "[V4L2 " << device_name_
                      << "] OpenCV conversion failed: " << error.what() << '\n';
        }

        if (!queue_buffer(buffer.index)) {
            return false;
        }
        return converted && !output.empty();
    }

    int width() const { return width_; }
    int height() const { return height_; }
    int fps() const { return fps_; }
    const std::string& device_name() const { return device_name_; }

    void close_device()
    {
        if (fd_ < 0) return;

        if (streaming_) {
            v4l2_buf_type type = buffer_type_;
            xioctl(VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }

        for (V4L2Buffer& buffer : buffers_) {
            for (MappedPlane& plane : buffer.planes) {
                if (plane.start && plane.start != MAP_FAILED) {
                    ::munmap(plane.start, plane.length);
                }
            }
        }
        buffers_.clear();
        ::close(fd_);
        fd_ = -1;
    }

private:
    int xioctl(unsigned long request, void* argument) const
    {
        int result;
        do {
            result = ::ioctl(fd_, request, argument);
        } while (result < 0 && errno == EINTR);
        return result;
    }

    std::vector<uint32_t> enumerate_formats() const
    {
        std::vector<uint32_t> formats;
        for (uint32_t index = 0;; ++index) {
            v4l2_fmtdesc description{};
            description.type = buffer_type_;
            description.index = index;
            if (xioctl(VIDIOC_ENUM_FMT, &description) < 0) break;
            formats.push_back(description.pixelformat);
        }
        return formats;
    }

    static uint32_t choose_format(const std::vector<uint32_t>& preferred,
                                  const std::vector<uint32_t>& supported)
    {
        for (uint32_t wanted : preferred) {
            if (std::find(supported.begin(), supported.end(), wanted) != supported.end()) {
                return wanted;
            }
        }
        return 0;
    }

    static bool can_convert(uint32_t format)
    {
        switch (format) {
        case V4L2_PIX_FMT_MJPEG:
        case V4L2_PIX_FMT_YUYV:
        case V4L2_PIX_FMT_UYVY:
        case V4L2_PIX_FMT_NV12:
        case V4L2_PIX_FMT_NV21:
        case V4L2_PIX_FMT_YUV420:
        case V4L2_PIX_FMT_YVU420:
        case V4L2_PIX_FMT_RGB24:
        case V4L2_PIX_FMT_BGR24:
            return true;
#ifdef V4L2_PIX_FMT_NV12M
        case V4L2_PIX_FMT_NV12M:
            return true;
#endif
#ifdef V4L2_PIX_FMT_NV21M
        case V4L2_PIX_FMT_NV21M:
            return true;
#endif
        default:
            return false;
        }
    }

    void read_frame_rate()
    {
        v4l2_streamparm parameters{};
        parameters.type = buffer_type_;
        if (xioctl(VIDIOC_G_PARM, &parameters) != 0) return;

        const uint32_t numerator =
            parameters.parm.capture.timeperframe.numerator;
        const uint32_t denominator =
            parameters.parm.capture.timeperframe.denominator;
        if (numerator > 0 && denominator > 0) {
            const double reported_fps =
                static_cast<double>(denominator) / numerator;
            fps_ = std::max(1, static_cast<int>(reported_fps + 0.5));
        }
    }

    bool create_mmap_buffers()
    {
        v4l2_requestbuffers request{};
        request.count = 4;
        request.type = buffer_type_;
        request.memory = V4L2_MEMORY_MMAP;
        if (xioctl(VIDIOC_REQBUFS, &request) < 0) {
            std::perror("[V4L2] VIDIOC_REQBUFS");
            return false;
        }
        if (request.count < 2) {
            std::cerr << "[V4L2] insufficient MMAP buffers\n";
            return false;
        }

        buffers_.resize(request.count);
        for (uint32_t index = 0; index < request.count; ++index) {
            v4l2_buffer buffer{};
            std::array<v4l2_plane, VIDEO_MAX_PLANES> planes{};
            buffer.type = buffer_type_;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = index;
            if (multi_planar_) {
                buffer.m.planes = planes.data();
                buffer.length = plane_count_;
            }

            if (xioctl(VIDIOC_QUERYBUF, &buffer) < 0) {
                std::perror("[V4L2] VIDIOC_QUERYBUF");
                return false;
            }

            const unsigned actual_planes = multi_planar_ ? plane_count_ : 1;
            buffers_[index].planes.resize(actual_planes);
            for (unsigned plane_index = 0; plane_index < actual_planes; ++plane_index) {
                const size_t length = multi_planar_
                    ? planes[plane_index].length
                    : buffer.length;
                const off_t offset = multi_planar_
                    ? static_cast<off_t>(planes[plane_index].m.mem_offset)
                    : static_cast<off_t>(buffer.m.offset);
                void* address = ::mmap(nullptr,
                                       length,
                                       PROT_READ | PROT_WRITE,
                                       MAP_SHARED,
                                       fd_,
                                       offset);
                if (address == MAP_FAILED) {
                    std::perror("[V4L2] mmap");
                    return false;
                }
                buffers_[index].planes[plane_index] = {address, length};
            }

            if (!queue_buffer(index)) return false;
        }
        return true;
    }

    bool queue_buffer(uint32_t index)
    {
        v4l2_buffer buffer{};
        std::array<v4l2_plane, VIDEO_MAX_PLANES> planes{};
        buffer.type = buffer_type_;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        if (multi_planar_) {
            buffer.m.planes = planes.data();
            buffer.length = plane_count_;
            for (unsigned i = 0; i < plane_count_; ++i) {
                planes[i].length = buffers_[index].planes[i].length;
            }
        }
        if (xioctl(VIDIOC_QBUF, &buffer) < 0) {
            std::perror("[V4L2] VIDIOC_QBUF");
            return false;
        }
        return true;
    }

    bool convert_buffer(const v4l2_buffer& buffer,
                        const std::array<v4l2_plane, VIDEO_MAX_PLANES>& planes,
                        cv::Mat& output) const
    {
        if (buffer.index >= buffers_.size() || buffers_[buffer.index].planes.empty()) {
            return false;
        }

        const V4L2Buffer& mapped = buffers_[buffer.index];
        const size_t first_bytes = multi_planar_ ? planes[0].bytesused : buffer.bytesused;
        uint8_t* first = static_cast<uint8_t*>(mapped.planes[0].start);

        if (pixel_format_ == V4L2_PIX_FMT_MJPEG) {
            cv::Mat encoded(1, static_cast<int>(first_bytes), CV_8UC1, first);
            output = cv::imdecode(encoded, cv::IMREAD_COLOR);
            return !output.empty();
        }

        if (pixel_format_ == V4L2_PIX_FMT_YUYV ||
            pixel_format_ == V4L2_PIX_FMT_UYVY) {
            const size_t stride = bytes_per_line_[0] ? bytes_per_line_[0] : width_ * 2;
            cv::Mat packed(height_, width_, CV_8UC2, first, stride);
            const int conversion = pixel_format_ == V4L2_PIX_FMT_YUYV
                ? cv::COLOR_YUV2BGR_YUY2
                : cv::COLOR_YUV2BGR_UYVY;
            cv::cvtColor(packed, output, conversion);
            return true;
        }

        if (pixel_format_ == V4L2_PIX_FMT_RGB24 ||
            pixel_format_ == V4L2_PIX_FMT_BGR24) {
            const size_t stride = bytes_per_line_[0] ? bytes_per_line_[0] : width_ * 3;
            cv::Mat packed(height_, width_, CV_8UC3, first, stride);
            if (pixel_format_ == V4L2_PIX_FMT_RGB24) {
                cv::cvtColor(packed, output, cv::COLOR_RGB2BGR);
            } else {
                output = packed.clone();
            }
            return true;
        }

        if (pixel_format_ == V4L2_PIX_FMT_NV12 ||
            pixel_format_ == V4L2_PIX_FMT_NV21
#ifdef V4L2_PIX_FMT_NV12M
            || pixel_format_ == V4L2_PIX_FMT_NV12M
#endif
#ifdef V4L2_PIX_FMT_NV21M
            || pixel_format_ == V4L2_PIX_FMT_NV21M
#endif
        ) {
            cv::Mat yuv;
            if (mapped.planes.size() >= 2) {
                yuv.create(height_ * 3 / 2, width_, CV_8UC1);
                const size_t y_stride = bytes_per_line_[0] ? bytes_per_line_[0] : width_;
                const size_t uv_stride = bytes_per_line_.size() > 1 && bytes_per_line_[1]
                    ? bytes_per_line_[1]
                    : width_;
                const uint8_t* y = static_cast<const uint8_t*>(mapped.planes[0].start);
                const uint8_t* uv = static_cast<const uint8_t*>(mapped.planes[1].start);
                for (int row = 0; row < height_; ++row) {
                    std::memcpy(yuv.ptr(row), y + row * y_stride, width_);
                }
                for (int row = 0; row < height_ / 2; ++row) {
                    std::memcpy(yuv.ptr(height_ + row), uv + row * uv_stride, width_);
                }
            } else {
                const size_t stride = bytes_per_line_[0] ? bytes_per_line_[0] : width_;
                cv::Mat wrapped(height_ * 3 / 2, width_, CV_8UC1, first, stride);
                yuv = wrapped;
            }

            bool is_nv21 = pixel_format_ == V4L2_PIX_FMT_NV21;
#ifdef V4L2_PIX_FMT_NV21M
            is_nv21 = is_nv21 || pixel_format_ == V4L2_PIX_FMT_NV21M;
#endif
            cv::cvtColor(yuv,
                         output,
                         is_nv21 ? cv::COLOR_YUV2BGR_NV21 : cv::COLOR_YUV2BGR_NV12);
            return true;
        }

        if (pixel_format_ == V4L2_PIX_FMT_YUV420 ||
            pixel_format_ == V4L2_PIX_FMT_YVU420) {
            const size_t stride = bytes_per_line_[0] ? bytes_per_line_[0] : width_;
            cv::Mat yuv(height_ * 3 / 2, width_, CV_8UC1, first, stride);
            cv::cvtColor(yuv,
                         output,
                         pixel_format_ == V4L2_PIX_FMT_YUV420
                             ? cv::COLOR_YUV2BGR_I420
                             : cv::COLOR_YUV2BGR_YV12);
            return true;
        }

        return false;
    }

    std::string device_name_;
    int fd_ = -1;
    int width_ = 1280;
    int height_ = 720;
    int fps_ = 30;
    bool streaming_ = false;
    bool multi_planar_ = false;
    v4l2_buf_type buffer_type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    uint32_t pixel_format_ = 0;
    unsigned plane_count_ = 1;
    std::vector<uint32_t> bytes_per_line_;
    std::vector<V4L2Buffer> buffers_;
};

std::vector<uint32_t> mipi_format_preferences()
{
    std::vector<uint32_t> formats = {
        V4L2_PIX_FMT_NV12,
        V4L2_PIX_FMT_NV21,
        V4L2_PIX_FMT_YUYV,
        V4L2_PIX_FMT_UYVY,
        V4L2_PIX_FMT_YUV420,
        V4L2_PIX_FMT_YVU420,
        V4L2_PIX_FMT_MJPEG
    };
#ifdef V4L2_PIX_FMT_NV12M
    formats.insert(formats.begin() + 1, V4L2_PIX_FMT_NV12M);
#endif
#ifdef V4L2_PIX_FMT_NV21M
    formats.insert(formats.begin() + 3, V4L2_PIX_FMT_NV21M);
#endif
    return formats;
}

std::vector<uint32_t> usb_format_preferences()
{
    std::vector<uint32_t> formats = {
        V4L2_PIX_FMT_MJPEG,
        V4L2_PIX_FMT_YUYV,
        V4L2_PIX_FMT_UYVY,
        V4L2_PIX_FMT_NV12,
        V4L2_PIX_FMT_NV21
    };
#ifdef V4L2_PIX_FMT_NV12M
    formats.push_back(V4L2_PIX_FMT_NV12M);
#endif
#ifdef V4L2_PIX_FMT_NV21M
    formats.push_back(V4L2_PIX_FMT_NV21M);
#endif
    return formats;
}

// ============================================================================
//  帧源抽象：让 read_thread 既能读 V4L2 摄像头，也能读视频文件（调试用）
//  调试时用视频文件模拟摄像头输入（如 input1.mp4），无需真实摄像头。
// ============================================================================
struct IFrameSource {
    virtual ~IFrameSource() = default;
    virtual bool read_bgr(cv::Mat& out, int timeout_ms) = 0;
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual int fps() const = 0;
    virtual std::string name() const = 0;
};

// V4L2 摄像头帧源（适配现有 V4L2Camera）
struct V4L2Source : IFrameSource {
    V4L2Camera* cam = nullptr;
    explicit V4L2Source(V4L2Camera* c) : cam(c) {}
    bool read_bgr(cv::Mat& out, int timeout_ms) override {
        return cam->read_frame_bgr(out, timeout_ms);
    }
    int width() const override { return cam->width(); }
    int height() const override { return cam->height(); }
    int fps() const override { return cam->fps(); }
    std::string name() const override { return cam->device_name(); }
};

// 视频文件帧源（cv::VideoCapture 读文件，播完自动循环）
struct VideoSource : IFrameSource {
    cv::VideoCapture cap;
    std::string path_;
    int w_ = 0, h_ = 0, fps_ = 30;

    bool open(const std::string& path) {
        path_ = path;
        if (!cap.open(path)) return false;
        w_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        h_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double f = cap.get(cv::CAP_PROP_FPS);
        if (f > 1.0) fps_ = static_cast<int>(f);
        return w_ > 0 && h_ > 0;
    }
    bool read_bgr(cv::Mat& out, int) override {
        if (!cap.read(out) || out.empty()) {
            // 播完循环重播
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!cap.read(out) || out.empty()) return false;
        }
        return true;
    }
    int width() const override { return w_; }
    int height() const override { return h_; }
    int fps() const override { return fps_; }
    std::string name() const override { return path_; }
};

// 判断参数是"视频文件"还是"摄像头设备"：非 /dev/ 且文件可读 → 视频文件
bool is_video_source(const std::string& s)
{
    if (s.find("/dev/") != std::string::npos) return false;
    return ::access(s.c_str(), R_OK) == 0;
}

void read_thread(IFrameSource& source, int camera_id, bool do_undistort)
{
    uint64_t sequence = 0;
    uint64_t dropped = 0;

    while (!g_stop.load()) {
        cv::Mat bgr;
        if (!source.read_bgr(bgr, 1000)) continue;
        g_capture_frames[camera_id].fetch_add(1, std::memory_order_relaxed);

        // 仅对"真实 MIPI 摄像头"执行 OpenCL 去畸变（视频源已校正，跳过）
        if (do_undistort && camera_id == 0 && g_undistort_context != nullptr) {
            cv::Mat undistorted;
            if (g_undistort_context->undistort(bgr, undistorted)) {
                bgr = std::move(undistorted);
            } else {
                std::cerr << "[Undistort] Failed on frame from "
                          << source.name() << '\n';
            }
        }

        FrameData frame;
        frame.frame = std::move(bgr);
        frame.index = g_task_index.fetch_add(1);
        frame.camera_sequence = sequence++;
        frame.camera_id = camera_id;

        if (!g_read_queue.try_enqueue(std::move(frame))) {
            ++dropped;
            g_capture_drops[camera_id].fetch_add(1, std::memory_order_relaxed);
        }
    }

    if (g_active_readers.fetch_sub(1) == 1) {
        g_read_finish = true;
    }
    std::cerr << "[Capture " << source.name() << "] stopped\n";
}

cv::Mat fit_into_tile(const cv::Mat& source, int tile_width, int tile_height)
{
    cv::Mat tile = cv::Mat::zeros(tile_height, tile_width, CV_8UC3);
    if (source.empty()) return tile;

    const double scale = std::min(static_cast<double>(tile_width) / source.cols,
                                  static_cast<double>(tile_height) / source.rows);
    const int resized_width = std::max(1, static_cast<int>(source.cols * scale));
    const int resized_height = std::max(1, static_cast<int>(source.rows * scale));
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(resized_width, resized_height));

    const int x = (tile_width - resized_width) / 2;
    const int y = (tile_height - resized_height) / 2;
    resized.copyTo(tile(cv::Rect(x, y, resized_width, resized_height)));
    return tile;
}

cv::Mat compose_side_by_side(const std::array<cv::Mat, kCameraCount>& latest,
                             const std::array<std::string, kCameraCount>& labels,
                             int tile_width,
                             int tile_height)
{
    cv::Mat composite = cv::Mat::zeros(tile_height,
                                        tile_width * kCameraCount,
                                        CV_8UC3);
    for (int camera_id = 0; camera_id < kCameraCount; ++camera_id) {
        cv::Mat tile = fit_into_tile(latest[camera_id], tile_width, tile_height);
        cv::putText(tile,
                    labels[camera_id],
                    cv::Point(24, 44),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA);
        tile.copyTo(composite(cv::Rect(camera_id * tile_width,
                                       0,
                                       tile_width,
                                       tile_height)));
    }
    return composite;
}

struct PendingInference {
    int camera_id = -1;
    uint64_t sequence = 0;
    std::future<ProcessResult> future;
};

void inference_and_compositor_thread(
    ThreadPoll& npu_pool,
    const std::array<std::string, kCameraCount>& labels,
    int tile_width,
    int tile_height)
{
    std::vector<PendingInference> pending;
    std::array<cv::Mat, kCameraCount> latest;
    std::array<uint64_t, kCameraCount> latest_sequence{{0, 0}};
    std::array<bool, kCameraCount> have_frame{{false, false}};
    std::array<bool, kCameraCount> fresh_frame{{false, false}};
    auto last_output = std::chrono::steady_clock::now();
#ifdef QWEN_ENABLED
    std::array<std::string, kCameraCount> detection_text;
    auto last_qwen_feed = std::chrono::steady_clock::now();
#endif

    while (!g_read_finish.load() || !g_read_queue.empty() || !pending.empty()) {
        FrameData input;
        while (pending.size() < kMaxInflightFrames &&
               g_read_queue.try_dequeue(input)) {
            PendingInference task;
            task.camera_id = input.camera_id;
            task.sequence = input.camera_sequence;
            task.future = npu_pool.submit_task_async(static_cast<int>(input.index),
                                                     std::move(input.frame));
            pending.emplace_back(std::move(task));
        }

        bool made_progress = false;
        for (auto iterator = pending.begin(); iterator != pending.end();) {
            if (iterator->future.wait_for(std::chrono::milliseconds(0)) !=
                std::future_status::ready) {
                ++iterator;
                continue;
            }

            ProcessResult result;
            try {
                result = iterator->future.get();
            } catch (const std::exception& error) {
                result.success = false;
                result.error_msg = error.what();
            } catch (...) {
                result.success = false;
                result.error_msg = "unknown exception from NPU worker";
            }
            const int camera_id = iterator->camera_id;
            if (result.success && !result.processed_img.empty() &&
                (!have_frame[camera_id] || iterator->sequence >= latest_sequence[camera_id])) {
                latest[camera_id] = result.processed_img.clone();
                latest_sequence[camera_id] = iterator->sequence;
                have_frame[camera_id] = true;
                fresh_frame[camera_id] = true;
                g_inference_frames[camera_id].fetch_add(1, std::memory_order_relaxed);
#ifdef QWEN_ENABLED
                detection_text[camera_id] =
                    detection_group_to_text(result.detection_results, labels[camera_id].c_str());
#endif
            } else if (!result.success) {
                std::cerr << "[NPU camera " << camera_id << "] "
                          << result.error_msg << '\n';
            }
            iterator = pending.erase(iterator);
            made_progress = true;
        }

        const bool all_available =
            std::all_of(have_frame.begin(), have_frame.end(), [](bool value) { return value; });
        const bool all_fresh =
            std::all_of(fresh_frame.begin(), fresh_frame.end(), [](bool value) { return value; });
        const auto now = std::chrono::steady_clock::now();
        const bool refresh_timeout =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_output).count() >= 200;

        if (all_available && (all_fresh || refresh_timeout)) {
            FrameData output;
            output.frame = compose_side_by_side(latest,
                                                 labels,
                                                 tile_width,
                                                 tile_height);
            if (!g_write_queue.try_enqueue(std::move(output))) {
                g_composite_drops.fetch_add(1, std::memory_order_relaxed);
            }
            fresh_frame.fill(false);
            last_output = now;
            made_progress = true;

#ifdef QWEN_ENABLED
            // 关键帧调度：每 2 秒（或检测到目标时）投喂一帧给 Qwen 做风险分析。
            // Qwen 推理慢（秒级），用"最新帧覆盖"单槽避免任务堆积。
            const bool has_target =
                !detection_text[0].empty() || !detection_text[1].empty();
            const auto qwen_elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - last_qwen_feed).count();
            if (has_target && qwen_elapsed >= 2000) {
                std::string ctx = detection_text[0] + detection_text[1];
                qwen_feed(output.frame, ctx);
                last_qwen_feed = now;
            }
#endif
        }

        if (!made_progress) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    g_process_finish = true;
    std::cerr << "[Compositor] stopped\n";
}

void write_thread()
{
    const int nv12_size = calc_nv12_mpp_size(g_output_width, g_output_height);
    std::vector<uint8_t> nv12_buffer(nv12_size);

    while (!g_process_finish.load() || !g_write_queue.empty()) {
        FrameData output;
        if (!g_write_queue.try_dequeue(output)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (output.frame.empty()) continue;

        bgr_to_nv12_with_rga(output.frame, nv12_buffer.data());
        if (process_frame(nv12_buffer.data(), nv12_size) == 0) {
            g_streamed_frames.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_stream_errors.fetch_add(1, std::memory_order_relaxed);
        }
    }
    std::cerr << "[Encoder] stopped\n";
}

int parse_positive_int(const char* text, int fallback)
{
    if (!text) return fallback;
    try {
        const int value = std::stoi(text);
        return value > 0 ? value : fallback;
    } catch (...) {
        return fallback;
    }
}

// 解析运行时长（秒）：允许 0 表示无限运行；负数/非法值回退到 fallback。
int parse_run_seconds(const char* text, int fallback)
{
    if (!text) return fallback;
    try {
        const int value = std::stoi(text);
        return value >= 0 ? value : fallback;
    } catch (...) {
        return fallback;
    }
}

std::string default_model_path_from_executable()
{
    std::array<char, 4096> executable_path{};
    const ssize_t length = ::readlink("/proc/self/exe",
                                      executable_path.data(),
                                      executable_path.size() - 1);
    if (length <= 0) return "../model/yolov5s.rknn";

    const std::string full_path(executable_path.data(),
                                static_cast<size_t>(length));
    const size_t slash = full_path.find_last_of('/');
    if (slash == std::string::npos) return "../model/yolov5s.rknn";
    return full_path.substr(0, slash) + "/../model/yolov5s.rknn";
}

}  // namespace

int main(int argc, char** argv)
{
    const std::string mipi_device = argc > 1 ? argv[1] : "/dev/video11";
    const std::string usb_device = argc > 2 ? argv[2] : "/dev/video20";
    const std::string rtmp_url = argc > 3
        ? argv[3]
        : "rtmp://192.168.1.30:1935/live/app";
    const int capture_width = argc > 4 ? parse_positive_int(argv[4], 1280) : 1280;
    const int capture_height = argc > 5 ? parse_positive_int(argv[5], 720) : 720;
    // 仅在驱动不报告实际帧率时使用；不会通过 VIDIOC_S_PARM 强制设置。
    const int fallback_fps = argc > 6 ? parse_positive_int(argv[6], 30) : 30;
    const std::string model_path = argc > 7
        ? argv[7]
        : default_model_path_from_executable();
    // 运行时长（秒）：0 表示无限运行；默认 300 秒（兼容原行为）。
    const int run_seconds = argc > 8 ? parse_run_seconds(argv[8], 300) : 300;

    if (::access(model_path.c_str(), R_OK) != 0) {
        std::cerr << "[Main] RKNN model is not readable: " << model_path << '\n';
        return EXIT_FAILURE;
    }

    // 帧源选择：视频文件（调试用）或 V4L2 摄像头（真实采集）。
    // 当命令行参数是"可读文件且非 /dev/ 设备"时，按视频文件回放处理，
    // 方便在没有摄像头时用 input1.mp4 等视频验证识别/推流效果。
    std::array<V4L2Camera, kCameraCount> cameras;
    VideoSource video_sources[kCameraCount];
    V4L2Source v4l2_src0(&cameras[0]);
    V4L2Source v4l2_src1(&cameras[1]);
    IFrameSource* sources[kCameraCount] = {nullptr, nullptr};
    bool use_video[kCameraCount] = {false, false};

    // ---- camera 0（MIPI 路） ----
    if (is_video_source(mipi_device)) {
        if (!video_sources[0].open(mipi_device)) {
            std::cerr << "[Main] failed to open video " << mipi_device << '\n';
            return EXIT_FAILURE;
        }
        sources[0] = &video_sources[0];
        use_video[0] = true;
    } else {
        if (!cameras[0].open_device(mipi_device,
                                    capture_width,
                                    capture_height,
                                    fallback_fps,
                                    mipi_format_preferences())) {
            std::cerr << "[Main] failed to open MIPI camera " << mipi_device << '\n';
            return EXIT_FAILURE;
        }
        sources[0] = &v4l2_src0;
    }

    // ---- camera 1（USB 路） ----
    if (is_video_source(usb_device)) {
        if (!video_sources[1].open(usb_device)) {
            std::cerr << "[Main] failed to open video " << usb_device << '\n';
            return EXIT_FAILURE;
        }
        sources[1] = &video_sources[1];
        use_video[1] = true;
    } else {
        if (!cameras[1].open_device(usb_device,
                                    capture_width,
                                    capture_height,
                                    fallback_fps,
                                    usb_format_preferences())) {
            std::cerr << "[Main] failed to open USB camera " << usb_device << '\n';
            return EXIT_FAILURE;
        }
        sources[1] = &v4l2_src1;
    }

    // FFmpeg/RTMP 最终只输出一帧 1280x720：左右各一个 640x720 区域。
    // fit_into_tile 会等比缩放，因此每路 16:9 画面完整显示为 640x360，
    // 在各自区域内垂直居中，不拉伸、不裁剪。
    g_output_width = 1280;
    g_output_height = 720;
    const int tile_width = g_output_width / kCameraCount;
    const int tile_height = g_output_height;

    // 初始化 OpenCL 去畸变模块（仅用于"真实 MIPI 摄像头"，视频源跳过）
    // 使用模拟标定参数，后续可替换为真实标定结果
    CameraCalibration calibration;
    g_undistort_context = new OpenCLUndistortContext();
    if (!g_undistort_context->initialize(calibration,
                                         sources[0]->width(),
                                         sources[0]->height())) {
        std::cerr << "[Main] OpenCL undistortion initialization failed, "
                  << "continuing without undistortion\n";
        delete g_undistort_context;
        g_undistort_context = nullptr;
    }

    // 使用两路帧源报告值中较低的一路作为编码器参考帧率。
    // 实际 RTMP PTS/DTS 由单调时钟生成，可以接受可变间隔。
    const int stream_fps = std::max(1,
        std::min(sources[0]->fps(), sources[1]->fps()));

    std::unique_ptr<ThreadPoll> npu_pool;
    try {
        npu_pool.reset(new ThreadPoll(model_path.c_str(), 3));
    } catch (const std::exception& error) {
        std::cerr << "[Main] NPU initialization failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    const int bitrate = g_output_width * g_output_height / 8 * stream_fps;
    if (init_streamer(g_output_width,
                      g_output_height,
                      stream_fps,
                      bitrate,
                      rtmp_url.c_str()) != 0) {
        std::cerr << "[Main] streamer initialization failed\n";
        return EXIT_FAILURE;
    }

    const std::array<std::string, kCameraCount> labels = {{
        (use_video[0] ? "VIDEO " : "MIPI ") + sources[0]->name(),
        (use_video[1] ? "VIDEO " : "USB ") + sources[1]->name()
    }};

    g_active_readers = kCameraCount;
    // 仅真实 MIPI 摄像头执行去畸变（camera0 且非视频源）；USB 路从不去畸变
    std::thread mipi_reader(read_thread, std::ref(*sources[0]), 0, !use_video[0]);
    std::thread usb_reader(read_thread, std::ref(*sources[1]), 1, false);
    std::thread compositor(inference_and_compositor_thread,
                           std::ref(*npu_pool),
                           std::cref(labels),
                           tile_width,
                           tile_height);
    std::thread encoder(write_thread);

#ifdef QWEN_ENABLED
    // Qwen 旁路风险分析：模型路径可通过环境变量覆盖，默认相对 build 目录 ../model/
    const std::string qwen_llm = argc > 9 ? argv[9]
        : "../model/Qwen3-VL-2B_llm_w8a8_rk3588.rkllm";
    const std::string qwen_vision = argc > 10 ? argv[10]
        : "../model/Qwen3-VL-2B_vision_rk3588.rknn";
    std::thread qwen_worker;
    if (g_qwen.initialize(qwen_llm.c_str(), qwen_vision.c_str())) {
        qwen_worker = std::thread(qwen_analysis_thread);
        std::cerr << "[Main] Qwen risk analyzer started (旁路)\n";
    } else {
        std::cerr << "[Main] Qwen analyzer unavailable; continuing without it\n";
    }
#endif

    std::cerr << "[Main] dual-camera detection started: "
              << mipi_device << " + " << usb_device << " -> "
              << g_output_width << 'x' << g_output_height
              << ", nominal " << stream_fps << "fps (variable input timing)\n";

    const auto start = std::chrono::steady_clock::now();
    auto last_stats = start;
    while (!g_stop.load()) {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start);
        if (run_seconds > 0 && elapsed.count() >= run_seconds) {
            std::cerr << "[Main] " << run_seconds << " seconds reached; stopping\n";
            g_stop = true;
            break;
        }

        const auto stats_interval =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stats);
        if (stats_interval.count() >= 2000) {
            const double seconds = stats_interval.count() / 1000.0;
            const uint64_t cap0 = g_capture_frames[0].exchange(0);
            const uint64_t cap1 = g_capture_frames[1].exchange(0);
            const uint64_t infer0 = g_inference_frames[0].exchange(0);
            const uint64_t infer1 = g_inference_frames[1].exchange(0);
            const uint64_t drop0 = g_capture_drops[0].exchange(0);
            const uint64_t drop1 = g_capture_drops[1].exchange(0);
            const uint64_t composite_drop = g_composite_drops.exchange(0);
            const uint64_t streamed = g_streamed_frames.exchange(0);
            const uint64_t stream_errors = g_stream_errors.exchange(0);

            std::cerr << std::fixed << std::setprecision(1)
                      << "[FPS] capture=" << cap0 / seconds << '/' << cap1 / seconds
                      << " infer=" << infer0 / seconds << '/' << infer1 / seconds
                      << " push=" << streamed / seconds
                      << " drops=" << drop0 << '/' << drop1
                      << " composite=" << composite_drop
                      << " errors=" << stream_errors << '\n';
            last_stats = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    mipi_reader.join();
    usb_reader.join();
    compositor.join();
    encoder.join();

#ifdef QWEN_ENABLED
    g_qwen_stop = true;
    if (qwen_worker.joinable()) {
        qwen_worker.join();
    }
    g_qwen.release();
#endif

    close_streamer();

    // 释放 OpenCL 去畸变资源
    if (g_undistort_context != nullptr) {
        g_undistort_context->release();
        delete g_undistort_context;
        g_undistort_context = nullptr;
    }

    cameras[0].close_device();
    cameras[1].close_device();
    std::cerr << "[Main] all threads stopped\n";
    return EXIT_SUCCESS;
}
