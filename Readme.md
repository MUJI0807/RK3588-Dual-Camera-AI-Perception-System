# RK3588 Dual-Camera AI Perception System with OpenCL Undistortion

<div align="center">

**Real-time YOLOv5s + UNet on RK3588 with GPU-accelerated lens undistortion**

[![RK3588](https://img.shields.io/badge/Platform-RK3588-blue)](https://www.rockchip.com)
[![OpenCL](https://img.shields.io/badge/GPU%20Acceleration-OpenCL%202.0-orange)](https://www.opencl.org)
[![Mali-G610](https://img.shields.io/badge/GPU-Mali--G610-green)](https://developer.arm.com/ip-graphics/mali-gpu)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

## Overview

A production-grade **AI perception camera system** for autonomous driving assistance, deployed on the RK3588 edge computing platform. This project implements a complete pipeline from camera capture to AI inference to video streaming, with **OpenCL-accelerated lens undistortion** as a key preprocessing step.

### Key Features

- **Dual-Camera Support**: MIPI CSI (ov13855) + USB camera with synchronized capture
- **GPU-Accelerated Preprocessing**: Real-time lens undistortion on Mali-G610 GPU using OpenCL
- **AI Inference**: YOLOv5s vehicle detection + UNet lane line segmentation on 3-core NPU
- **Zero-Copy Pipeline**: V4L2 MMAP + DMA-BUF for minimal latency
- **Hardware Encoding**: H.265/HEVC via MPP with RTMP streaming
- **Multi-Threaded Architecture**: Asynchronous capture → preprocess → infer → encode pipeline

## System Architecture

```
┌─────────────┐
│  MIPI Camera│──────┐
│  (ov13855)  │      │
└─────────────┘      │
                     ▼
              ┌──────────────┐
              │ V4L2 Capture │  MMAP Zero-Copy
              │  (NV12→BGR)  │
              └──────┬───────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  OpenCL Undistortion  │  ← GPU Acceleration
         │  (Mali-G610, 5-10ms)  │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   g_read_queue        │  Thread-Safe Queue
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   NPU Inference       │  3-Core Parallel
         │  YOLOv5s + UNet       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │    Compositor         │  Side-by-Side
         │   (1280×720)          │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   g_write_queue       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  RGA (BGR→NV12)       │
         │  MPP H.265 Encoder    │
         │  RTMP Streaming       │
         └───────────────────────┘
```

## OpenCL Undistortion Module

### Why Undistortion Matters

Wide-angle lenses introduce radial and tangential distortion, causing straight lines to appear curved. For AI perception tasks like lane line detection and vehicle bounding box estimation, this geometric distortion directly impacts accuracy.

### Implementation Details

- **Algorithm**: Radial (k1, k2, k3) + Tangential (p1, p2) inverse distortion model
- **Parallelization**: NDRange 2D with 16×16 work groups (Mali GPU optimized)
- **Sampling**: Bilinear interpolation for sub-pixel accuracy
- **Performance**: 5-10ms per frame at 1280×720 resolution
- **Integration**: Inserted between V4L2 capture and NPU inference

### Distortion Model

```
Normalized coordinates:
  x_norm = (x - cx) / fx
  y_norm = (y - cy) / fy

Radial distortion:
  r² = x_norm² + y_norm²
  radial_dist = 1 + k1·r² + k2·r⁴ + k3·r⁶

Tangential distortion:
  x_distorted = x_norm·radial_dist + 2·p1·x_norm·y_norm + p2·(r² + 2·x_norm²)
  y_distorted = y_norm·radial_dist + p1·(r² + 2·y_norm²) + 2·p2·x_norm·y_norm

Inverse mapping:
  x_in = fx·x_distorted + cx
  y_in = fy·y_distorted + cy
```

## Project Structure

```
streamer_codev8.0/
├── main.cpp                  # Main pipeline (capture → undistort → infer → encode)
├── opencl_undistort.h        # OpenCL undistortion module header
├── opencl_undistort.cpp      # OpenCL undistortion implementation
├── undistort.cl              # OpenCL kernel (GPU code)
├── CMakeLists.txt            # Build configuration
├── SafeQueue.h               # Thread-safe bounded queue
├── streamer.h / streamer.c   # MPP encoder + RTMP push
├── thread_poll.h / .cpp      # NPU thread pool
├── yolov5s.h / .cpp          # YOLOv5s RKNN inference
├── post_process.h / .cpp     # NMS, decode, quantization
├── mpp.h / mpp.c             # Rockchip MPP wrapper
├── rtmp.h / rtmp.c           # RTMP streaming
├── 3rdparty/                 # RKNN API, RGA libraries
└── model/                    # RKNN model files
```

## Build & Run

### Prerequisites

- RK3588 development board (e.g., LubanCat 4)
- Rockchip SDK with OpenCL support
- OpenCV 4.x
- FFmpeg 4.x
- RKNN Toolkit 2.x

### Compilation

```bash
cd streamer_codev8.0
mkdir build && cd build
cmake ..
make -j4
```

### Execution

```bash
# Basic usage
./app /dev/video11 /dev/video20 rtmp://192.168.1.30:1935/live/app 1280 720 30

# Parameters:
#   $1: MIPI camera device (default: /dev/video11)
#   $2: USB camera device (default: /dev/video20)
#   $3: RTMP URL (default: rtmp://192.168.1.30:1935/live/app)
#   $4: Capture width (default: 1280)
#   $5: Capture height (default: 720)
#   $6: Fallback FPS (default: 30)
#   $7: Model path (default: ./model/yolov5s.rknn)
```

### Expected Output

```
[OpenCL] Platform 0: ARM Platform
[OpenCL] Selected ARM/Mali platform
[OpenCL] Device: Mali-G610
[OpenCL] Initialization complete (1280x720)
[Main] dual-camera detection started: /dev/video11 + /dev/video20 -> 1280x720
[FPS] capture=30.0/30.0 infer=28.5/28.5 push=29.0 drops=0/0 composite=0 errors=0
```

## Performance

| Stage | Latency | Notes |
|-------|---------|-------|
| V4L2 Capture | ~33ms | 30fps, MMAP zero-copy |
| OpenCL Undistortion | 5-10ms | Mali-G610 GPU, 1280×720 |
| NPU Inference | ~25ms | YOLOv5s + UNet parallel |
| Compositor | ~2ms | Side-by-side composition |
| RGA + MPP Encode | ~8ms | H.265 hardware encoding |
| **Total Pipeline** | **~70ms** | **~14fps end-to-end** |

**Streaming Performance**: Stable 29fps RTMP output (camera limited at 30fps)

## Calibration Parameters

Current implementation uses **simulated calibration parameters** for demonstration:

```cpp
CameraCalibration::CameraCalibration()
    : fx(800.0f), fy(800.0f)      // Focal length (pixels)
    , cx(640.0f), cy(360.0f)      // Optical center (1280×720)
    , k1(-0.25f), k2(0.05f), k3(0.0f)  // Radial distortion
    , p1(0.001f), p2(-0.001f)     // Tangential distortion
{}
```

For production deployment, replace with real calibration parameters obtained from chessboard calibration using OpenCV's `cv::calibrateCamera`.

## Future Enhancements

- [ ] Zero-copy integration: DMA-BUF sharing between V4L2 and OpenCL
- [ ] Direct NV12 processing: Eliminate BGR↔RGBA color conversion
- [ ] Asynchronous OpenCL execution: Use events instead of `clFinish`
- [ ] Dual-camera undistortion: Extend to USB camera
- [ ] Real calibration pipeline: Integrate OpenCV chessboard calibration
- [ ] ISP tuning: Auto-exposure, white balance, low-light enhancement
- [ ] Multi-camera synchronization: Hardware trigger support

## Technical Highlights

1. **OpenCL GPU Acceleration**: Leverages Mali-G610 for real-time image preprocessing
2. **Zero-Copy Design**: V4L2 MMAP + DMA-BUF minimizes memory bandwidth
3. **NPU Parallelism**: 3-core dynamic scheduling for YOLO + UNet
4. **Bounded Queue with Drop Policy**: Prevents memory bloat under load
5. **Graceful Degradation**: OpenCL initialization failure doesn't crash the pipeline

## Release History & Changelog

### v8.0 (2026-08-31) — Dual-Camera YOLO + OpenCL Undistortion, 15 bug fixes

Fixes in commit 9034a53:

#### P0 Critical (crash / corrupted stream / data race)

- Concurrent lane/UNet state race: 3 NPU worker threads shared file-scope globals (g_unet_lane, g_smooth_*, g_candidate, ...) without locking → added g_assist_mutex around draw_driving_assist; inference still runs in parallel.
- RTMP infinite retry: avio_open retried forever when server unreachable → bounded to 3 attempts.
- All packets marked KEY: every frame carried AV_PKT_FLAG_KEY (P-frames treated as keyframes → playback artifacts) → NALU type detection, KEY only for IDR(5)/SPS(7)/PPS(8).
- MPP init failure ignored: init_streamer continued with an uninitialized encoder → fail-fast with cleanup.
- RKNN load failure silent: model load/init errors now throw std::runtime_error instead of running with a dead context.
- Unhandled worker exception: compositor future.get() wrapped in try/catch.

#### P1 Medium (resource leaks / visual defects / hardcoded behavior)

- OpenCL release leak: release() now frees every resource pointer and all init failure paths go through unified cleanup.
- Undistort black border: out-of-range sampling returned black → clamped to image edge (removes the black frame around the undistorted image).
- Redundant clFinish: removed extra full-pipeline sync in undistort().
- 300s hardcoded exit: runtime duration now configurable via argv[8] (0 = run forever, default 300s keeps old behavior).
- Init failure leaks in streamer.c / rtmp.c: all failure paths now release mpp ctx / SPS-PPS / rtmp ctx / codec ctx safely.
- ThreadPoll size race: tasks.size() reads now locked.
- MPP packet leak: mpp_packet_init_with_buffer failure path deinits packet.
- Box offset for non-16-aligned resolutions: detection boxes are remapped from padded to original image coordinates (e.g. 1920x1080 cameras).

#### Minor

- print_tensor_attr now actually prints tensor info; printf format specifiers fixed (%u for uint32_t fields).

### Earlier Versions (feature evolution)

| Version | Description |
|---------|-------------|
| v4.0 | OTA upgrade package, single camera, YOLOv5s/YOLOv6s + UNet lane lines |
| v5.0 | Single-camera (/dev/video0 YUYV) YOLOv5s RKNN + RTMP streaming |
| v6.0 | Dual-camera retrofit: MIPI /dev/video11 + USB /dev/video20, 2 capture threads, 3-instance RKNN pool, side-by-side compositing |
| v7.0 | Same codebase as v6.0 (repackaged) |
| v8.0 | Dual-camera + OpenCL undistortion (Mali-G610) + YOLOv5s + UNet lane lines |

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- Rockchip for RK3588 platform and SDK
- OpenCL working group for GPU computing standards
- OpenCV community for computer vision tools

---

<div align="center">

**Built with RK3588 | OpenCL | YOLOv5s | UNet**

*Real-time AI perception on the edge*

</div>
