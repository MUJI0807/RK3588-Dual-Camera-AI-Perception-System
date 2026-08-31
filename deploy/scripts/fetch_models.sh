#!/bin/bash
# ============================================================================
#  模型自动获取脚本（联网环境/板子执行一次）
#  ---------------------------------------------------------------------------
#  1) YOLOv5s 模型检查（model/yolov5s.rknn）
#  2) Qwen3-VL-2B 的 RKLLM 转换模型自动下载到 model/
#     来源: huggingface.co/GatekeeperZA/Qwen3-VL-2B-Instruct-RKLLM-v1.2.3
#            （Qwen3-VL-2B 针对 RK3588 NPU 的官方转换版）
#  之后即可打包成离线部署包，客户无需自己下载模型
# ============================================================================
set -e
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
MODEL_DIR="$ROOT/model"
mkdir -p "$MODEL_DIR"

REPO="GatekeeperZA/Qwen3-VL-2B-Instruct-RKLLM-v1.2.3"
HF_BASE="https://huggingface.co/$REPO/resolve/main"

echo "========== [1/2] YOLOv5s 模型 =========="
if [ -f "$MODEL_DIR/yolov5s.rknn" ]; then
    echo "已存在: model/yolov5s.rknn"
else
    echo "model/yolov5s.rknn 缺失：请从开发环境拷贝（本脚本不自动转换 RKNN）"
fi

echo "========== [2/2] Qwen3-VL-2B RKLLM 模型 =========="
# 通过 HuggingFace API 获取仓库文件列表
echo "查询模型仓库文件列表 ..."
FILES="$(curl -s --max-time 30 "https://huggingface.co/api/models/$REPO" \
    | python3 -c "import sys,json;
try:
    d=json.load(sys.stdin)
    print('\n'.join(s['rfilename'] for s in d.get('siblings',[])))
except Exception as e:
    sys.stderr.write('解析失败: %s\n' % e)")"

if [ -z "$FILES" ]; then
    echo "警告：无法获取模型文件列表（网络或仓库问题），请手动下载后放入 model/"
    echo "  仓库地址: https://huggingface.co/$REPO"
    exit 0
fi

DL_COUNT=0
for f in $FILES; do
    case "$f" in
        *.rkllm|*.rknn)
            DEST="$MODEL_DIR/$f"
            if [ -f "$DEST" ]; then
                echo "已存在: model/$f"
            else
                echo "下载 model/$f ..."
                curl -L --retry 3 -C - -o "$DEST" "$HF_BASE/$f"
                DL_COUNT=$((DL_COUNT+1))
            fi
            ;;
    esac
done

echo ""
if [ "$DL_COUNT" -gt 0 ]; then
    echo "✅ 已下载 $DL_COUNT 个 Qwen 模型文件到 model/"
fi
echo "Qwen 模型准备完成。启用步骤："
echo "  1) deploy/config.env 中 QWEN=1"
echo "  2) 确认 RKLLM 库已装，deploy/build.sh 重新编译"
echo "  3) ./deploy/start.sh"
