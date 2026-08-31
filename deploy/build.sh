#!/bin/bash
# ============================================================================
#  一键构建脚本（板端执行一次，生成可执行程序 app）
#  用途：把"源码 → 二进制"这一步自动化，部署工程师/售后在板子上跑一次
#  之后 build/app 可随部署包分发，客户无需再编译
# ============================================================================
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "========== [1/4] 依赖检查 =========="
command -v cmake >/dev/null 2>&1 || { echo "错误：缺少 cmake（apt install cmake）"; exit 1; }
command -v g++   >/dev/null 2>&1 || { echo "错误：缺少 g++（apt install g++）"; exit 1; }
echo "cmake / g++ 就绪"

# 加载配置
[ -f deploy/config.env ] && . deploy/config.env

# RKLLM（可选）：配置了 RKLLM_DIR 且库存在 → 编译时启用 Qwen
RKLLM_CMAKE_ARGS=""
if [ -n "$RKLLM_DIR" ] && [ -f "$RKLLM_DIR/include/rkllm.h" ] && ls "$RKLLM_DIR"/lib/librkllmrt.so >/dev/null 2>&1; then
    echo "检测到 RKLLM：Qwen3-VL 旁路风险分析将启用"
    RKLLM_CMAKE_ARGS="-DRKLLM_PATH=$RKLLM_DIR"
else
    echo "未配置/未找到 RKLLM：Qwen 编译为空壳（不影响 YOLO/UNet/推流）"
fi

echo "========== [2/4] YOLO 模型检查 =========="
if [ ! -f "$ROOT/model/yolov5s.rknn" ]; then
    echo "警告：缺少 model/yolov5s.rknn，请从开发环境拷贝"
fi

echo "========== [3/4] CMake 配置 =========="
mkdir -p "$ROOT/build"
cd "$ROOT/build"
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release $RKLLM_CMAKE_ARGS

echo "========== [4/4] 编译 =========="
cmake --build . -j"$(nproc)"

echo ""
echo "✅ 构建完成：$ROOT/build/app"
echo "   启动：./deploy/start.sh"
