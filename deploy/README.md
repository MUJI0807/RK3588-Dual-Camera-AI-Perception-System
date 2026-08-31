# 一键部署

工业交付标准：**客户解压部署包 → ./deploy/start.sh 即可运行**，不需要自己下载模型、配置路径。

## 目录结构

```
streamer_codev8.0/
├── deploy/                  ← 一键部署工程
│   ├── config.env           客户/部署工程师唯一需要改的配置文件
│   ├── build.sh             板端一键编译（售后/开发执行一次）
│   ├── start.sh             客户一键启动入口
│   ├── scripts/
│   │   └── fetch_models.sh  联网自动下载 Qwen3-VL 模型到 model/
│   └── README.md            本说明
├── main.cpp / qwen_analyzer.* / ...  源码
├── model/                   模型目录（yolov5s.rknn + Qwen rkllm/rknn）
└── build/                   （编译产物，app 可执行文件）
```

## 三步流程

### 第 1 步（联网机器/板子，一次性准备）：获取模型
```bash
./deploy/scripts/fetch_models.sh     # 自动下载 Qwen3-VL-2B RKLLM 模型到 model/
```
（若离线环境：把 model/ 整个目录拷进部署包即可）

### 第 2 步（板子，一次性编译）：构建可执行程序
```bash
sudo apt install -y cmake g++          # 板端基础工具（OpenCV/FFmpeg/MPP 等板级 SDK 自带）
./deploy/build.sh                      # 自动检测依赖 + 编译，生成 build/app
```
> 启用 Qwen 需在 `config.env` 配置 `RKLLM_DIR`（rknn-llm 安装路径），build.sh 会自动加 `-DRKLLM_PATH`

### 第 3 步（客户，每次运行）：一键启动
```bash
./deploy/start.sh
```

## 客户/部署工程师只需改 config.env

```bash
SOURCE0="/dev/video11"    # 输入源1：摄像头设备 或 视频文件(调试)
SOURCE1="/dev/video20"    # 输入源2
RTMP_URL="rtmp://..."     # 推流地址
QWEN="0"                  # 1=启用 Qwen 旁路风险分析，0=关闭
```

## 打成离线部署包（交付给客户）

在板子上完成第 1、2 步后（model/ 与 build/ 就位），打包整个目录即可：

```bash
cd .. && tar -czf streamer_codev8.0_deploy.tar.gz streamer_codev8.0 \
  --exclude=streamer_codev8.0/.git --exclude=streamer_codev8.0/build/CMakeFiles \
  --exclude=streamer_codev8.0/input1.mp4
```

客户拿到后：解压 → `./deploy/start.sh`。**无需下载任何模型，无需编译。**

## 说明
- `QWEN=0` 时程序不加载 Qwen 旁路，纯 YOLO/UNet 识别 + 推流，对算力零影响
- `QWEN=1` 时旁路做关键帧风险分析（事件驱动，秒级），输出结构化 JSON 到 stderr
- 视频回放调试：`SOURCE0/1` 填视频文件路径即可
