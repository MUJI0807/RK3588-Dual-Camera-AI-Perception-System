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
find_package(PkgConfig REQUIRED)  # 确保 pkg-config 工具可用，用于查找和配置 FFmpeg 库
pkg_check_modules(AVCODEC REQUIRED libavcodec)  # 查找 libavcodec 库，这是 FFmpeg 的核心库之一，负责音视频编解码。
pkg_check_modules(AVFORMAT REQUIRED libavformat) # 查找 libavformat 库，这是 FFmpeg 的核心库之一，负责音视频封装格式的处理。
pkg_check_modules(AVUTIL REQUIRED libavutil)  # 查找 libavutil 库，这是 FFmpeg 的核心库之一，提供了各种实用工具函数。
pkg_check_modules(SWSCALE REQUIRED libswscale) # 查找 libswscale 库，这是 FFmpeg 的核心库之一，负责图像缩放和像素格式转换。
# 打印消息到 CMake 构建过程中
message(STATUS "find FFmpeg Library")

# 查找 MPP 库
find_library(MPP_LIBRARY NAMES rockchip_mpp mpp)  # 查找名为 rockchip_mpp 或 mpp 的库文件，并将其路径存储在 MPP_LIBRARY 变量中。
if(NOT MPP_LIBRARY) # 如果没有找到 MPP 库，则输出错误信息并终止构建过程。
    message(FATAL_ERROR "MPP library not found") # 输出错误信息并终止构建过程。
endif() # 结束 if 条件语句。
message("find MPP library")  # 输出消息，表示已经成功找到 MPP 库。

# 打印消息到 CMake 构建过程中
message(" this is meihua embedded!")

# 设置目标架构为 aarch64
set(LIB_ARCH "aarch64")

# 设置设备名称为 RK3588，这是 Rockchip 系列的某个型号
set(DEVICE_NAME "RK3588")

# 设置第三方库的路径
set(3DRPARTY_PATH       ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)  #CMAKE_CURRENT_SOURCE_DIR表示当前CMakeLists.txt所在的目录路径
# 设置 RKNN API 路径
set(RKNN_API_PATH       ${3DRPARTY_PATH}/librknn_api)  # 设置librknn_api的作用是指定RKNN API的路径，以便在构建过程中能够正确地找到和使用RKNN相关的头文件和库文件。
# 设置 RKNN 包含目录
set(RKNN_INCLUDE_DIRS   ${RKNN_API_PATH}/include)      # 设置 RKNN 包含目录的作用是指定RKNN API的头文件所在的路径，以便在编译过程中能够正确地找到和包含这些头文件，从而使用RKNN提供的功能。
# 设置 RKNN 运行时库的路径
set(RKNN_LIBS           ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so) # 设置 RKNN 运行时库的路径。以便在链接阶段能够正确地找到和链接该库，从而在运行时能够使用RKNN提供的功能。

# 设置 RGA（Rockchip Graphic Accelerator）库的路径
set(RGA_PATH            ${3DRPARTY_PATH}/rga/${DEVICE_NAME})  # 设置 RGA 库的路径在构建过程中能够正确地找到和使用 RGA 提供的图形加速功能。
# 设置 RGA 包含目录
set(RGA_INCLUDE_DIRS    ${RGA_PATH}/include)   # RGA包含目录提供了RGA库的头文件路径
# 设置 RGA 库的路径
set(RGA_LIBS            ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so) # 设置 RGA 库的路径在运行时能够使用 RGA 提供的图形加速功能.


# 将 OpenCV 的头文件路径添加到编译器的搜索路径。
# 这使得源文件可以正确包含 OpenCV 的头文件。
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_INCLUDE_DIRS}
    ${RGA_INCLUDE_DIRS}
)
# 定义一个名为 cv 的可执行目标，该目标将从 main.cpp 源文件生成。
add_executable(app 
    main.cpp 
    thread_poll.cpp
    yolov5s.cpp
    lane_detector.cpp
    post_process.cpp
    streamer.c
    rtmp.c
    mpp.c
    unet_lane_detector.cpp
    perf_monitor.cpp
    )
    #后续若有新增文件可以在这里进行添加让CMakelist进行编译

# 将 OpenCV 的库与目标可执行文件 cv 链接，确保在程序运行时能够调用 OpenCV 函数。
target_link_libraries(app 
    ${OpenCV_LIBS} # 
    ${RKNN_LIBS}
    ${RGA_LIBS}
    
    # mpp的链接库
    ${MPP_LIBRARY}
   
    # FFmpeg的链接库
    ${AVCODEC_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    )

