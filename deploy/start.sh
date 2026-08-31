#!/bin/bash
# ============================================================================
#  一键启动脚本（客户入口）
#  用法： ./deploy/start.sh
#  ---------------------------------------------------------------------------
#  所有参数都在 deploy/config.env 中预设，客户无需改命令行
#  ============================================================================
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# 加载配置
[ -f deploy/config.env ] && . deploy/config.env

BIN="$ROOT/build/app"
if [ ! -x "$BIN" ]; then
    echo "错误：未找到可执行文件 $BIN"
    echo "请先在板子上执行一次 ./deploy/build.sh 完成编译"
    exit 1
fi

# Qwen 旁路开关：QWEN=1 传模型路径；否则传 off（不加载旁路）
if [ "$QWEN" = "1" ]; then
    QWEN_ARGS="$QWEN_LLM $QWEN_VISION"
else
    QWEN_ARGS="off off"
fi

echo "=============================================="
echo " 双摄YOLO识别 + RTMP推流  一键启动"
echo "  源1: $SOURCE0"
echo "  源2: $SOURCE1"
echo "  RTMP: $RTMP_URL"
echo "  Qwen旁路: $([ "$QWEN" = "1" ] && echo 启用 || echo 关闭)"
echo "=============================================="

exec "$BIN" "$SOURCE0" "$SOURCE1" "$RTMP_URL" "$WIDTH" "$HEIGHT" "$FPS" "$MODEL_YOLO" "$RUN_SECONDS" $QWEN_ARGS
