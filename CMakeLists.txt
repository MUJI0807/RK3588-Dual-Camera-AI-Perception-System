# 指定使用的最低 CMake 版本，以确保支持所有需要的功能和特性。
cmake_minimum_required(VERSION 3.5)
# 定义项目名称为 cv。CMake 将使用这个名称来创建相关构建目录和变量。
project(cv)
# 设置编译器标准为 C++14。这将确保代码在 C++14 标准下进行编译。
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 输出系统的基本信息，包括系统名称和版本号，便于调试和确认构建环境。
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")
# 查找并引入 OpenCV 库。如果找不到，将会给出错误提示。
find_package(OpenCV REQUIRED)
# 输出 OpenCV 库的信息，包括版本、库路径和头文件路径，帮助确认配置是否正确。
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")       # 输出 OpenCV 的版本
message(STATUS "    libraries: ${OpenCV_LIBS}")       # 输出 OpenCV 链接的库
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")  # 输出 OpenCV 包含文件的路径

# 查找并引入 FFmpeg 库
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(SWSCALE REQUIRED libswscale)
# 打印消息到 CMake 构建过程中
message("find FFmpeg Library")

# 查找 MPP 库
find_library(MPP_LIBRARY NAMES rockchip_mpp mpp)
if(NOT MPP_LIBRARY)
    message(FATAL_ERROR "MPP library not found")
endif()
message("find MPP library")

# 查找 OpenCL 库
find_package(OpenCL REQUIRED)
if(NOT OpenCL_FOUND)
    message(FATAL_ERROR "OpenCL not found")
endif()
message("find OpenCL library")

# 打印消息到 CMake 构建过程中
message(" this is meihua embedded!")

# 设置目标架构为 aarch64
set(LIB_ARCH "aarch64")

# 设置设备名称为 RK3588，这是 Rockchip 系列的某个型号
set(DEVICE_NAME "RK3588")

# 设置第三方库的路径
set(3DRPARTY_PATH       ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
# 设置 RKNN API 路径
set(RKNN_API_PATH       ${3DRPARTY_PATH}/librknn_api)
# 设置 RKNN 包含目录
set(RKNN_INCLUDE_DIRS   ${RKNN_API_PATH}/include)
# 设置 RKNN 运行时库的路径
set(RKNN_LIBS           ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

# 设置 RGA（Rockchip Graphic Accelerator）库的路径
set(RGA_PATH            ${3DRPARTY_PATH}/rga/${DEVICE_NAME})
# 设置 RGA 包含目录
set(RGA_INCLUDE_DIRS    ${RGA_PATH}/include)
# 设置 RGA 库的路径
set(RGA_LIBS            ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)

# 可选：Qwen3-VL 旁路风险分析（RKLLM，Rockchip 大模型运行时）
# 若板端装了 rknn-llm 库，指定 RKLLM_PATH 后会自动启用 QWEN_ENABLED；
# 否则 qwen_analyzer.cpp 编译为"空壳"，不影响现有双摄/推流功能。
set(RKLLM_PATH "" CACHE PATH "Path to RKLLM (rknn-llm) install directory")
find_path(RKLLM_INCLUDE_DIR rkllm.h
    HINTS ${RKLLM_PATH} ${RKLLM_PATH}/include)
find_library(RKLLM_LIBRARY NAMES rkllmrt
    HINTS ${RKLLM_PATH} ${RKLLM_PATH}/lib ${RKLLM_PATH}/lib/Linux/${LIB_ARCH})
if(RKLLM_INCLUDE_DIR AND RKLLM_LIBRARY)
    message(STATUS "RKLLM found: enable Qwen3-VL risk analyzer (QWEN_ENABLED)")
    add_definitions(-DQWEN_ENABLED)
    include_directories(${RKLLM_INCLUDE_DIR})
    set(RKLLM_LIBS ${RKLLM_LIBRARY})
    # API 版本选择：
    #   ON  = RKLLM 1.3.0 新版多模态 API（Qwen3.5-0.8B 等新模型，推荐）
    #   OFF = RKLLM 1.2.3 旧版 API（Qwen3-VL-2B / Qwen2-VL）
    option(QWEN_RKLLM_130 "Use RKLLM 1.3.0 new multimodal API (Qwen3.5-0.8B)" ON)
    if(QWEN_RKLLM_130)
        add_definitions(-DQWEN_RKLLM_130)
    endif()
else()
    message(STATUS "RKLLM not found: Qwen risk analyzer disabled (shell only)")
    set(RKLLM_LIBS "")
endif()


# 将 OpenCV 的头文件路径添加到编译器的搜索路径。
# 这使得源文件可以正确包含 OpenCV 的头文件。
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_INCLUDE_DIRS}
    ${RGA_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
)
# 定义一个名为 cv 的可执行目标，该目标将从 main.cpp 源文件生成。
add_executable(app 
    main.cpp 
    thread_poll.cpp
    yolov5s.cpp
    post_process.cpp
    lane_detector.cpp
    unet_lane_detector.cpp
    opencl_undistort.cpp
    qwen_analyzer.cpp
    streamer.c
    rtmp.c
    mpp.c
    )

# 将 OpenCV 的库与目标可执行文件 cv 链接，确保在程序运行时能够调用 OpenCV 函数。
target_link_libraries(app 
    ${OpenCV_LIBS}
    ${RKNN_LIBS}
    ${RGA_LIBS}
    ${OpenCL_LIBRARIES}
    # Qwen3-VL 旁路风险分析（RKLLM，未找到时为空）
    ${RKLLM_LIBS}
    # mpp的链接库
    ${MPP_LIBRARY}
    # FFmpeg的链接库
    ${AVCODEC_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    )

