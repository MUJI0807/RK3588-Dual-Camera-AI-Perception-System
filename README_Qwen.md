# Qwen3-VL-2B 旁路风险分析器 —— 板端部署说明

## 1. 定位

在现有"双摄 + YOLOv5s + UNet + OpenCL 去畸变 + RTMP 推流"实时管线之外，
新增一个 **低频语义风险分析层**：

- 实时部分（YOLO/UNet/编码/推流，30fps）**完全不变**
- Qwen3-VL-2B 只对"事件驱动/定时"抽出的关键帧做多模态理解
- 输入：去畸变后的合成帧 + 底层检测结果文本
- 输出：结构化 JSON（风险等级/原因/目标/建议）

对应源码：`qwen_analyzer.h` / `qwen_analyzer.cpp`，集成点在 `main.cpp`
（全部用 `#ifdef QWEN_ENABLED` 包裹，未启用时编译为"空壳"，不影响原功能）。

## 2. 依赖

| 库 | 说明 |
|----|------|
| librkllmrt.so + rkllm.h | Rockchip RKLLM 运行时（语言模型 W8A8），来自 [rknn-llm](https://github.com/airockchip/rknn-llm) |
| librknnrt.so + rknn_api.h | 视觉编码器 RKNN（项目 3rdparty 已带） |

## 3. 模型文件

需提前转换并放置（默认相对 build 目录的 `../model/`）：

```
model/Qwen3-VL-2B_llm_w8a8_rk3588.rkllm     # 语言模型（W8A8 量化）
model/Qwen3-VL-2B_vision_rk3588.rknn        # 视觉编码器
```

可直接参考现成转换版：
- https://huggingface.co/GatekeeperZA/Qwen3-VL-2B-Instruct-RKLLM-v1.2.3

## 4. ★ 必须按实际模型核对/调整的参数（在 qwen_analyzer.cpp 顶部）

这些参数依赖你转换出的模型输出，**不是固定值**，首次部署务必核对：

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `kDefaultVisionSize` | 392 | 视觉编码 resize 目标尺寸；init 时会读取 rknn 模型实际输入尺寸并覆盖 |
| `kNImageTokens` | 196 | 每张图的 image token 数（由视觉编码器输出 shape 决定） |
| `kImageEmbedLen` | 1536 | 每个 image token 的 embedding 维度 |
| `kImgStart/kImgEnd/kImgContent` | `<|vision_start|>` 等 | 特殊 token，随模型/转换脚本可能不同 |
| `kMaxNewTokens` | 512 | 最大生成 token 数（图中项目实测 812 时单帧 5.4s，可按需下调提速） |

**如何获取正确值**：视觉编码器 RKNN 的输出 tensor 是
`[n_image_tokens, image_embed_len]`（或展平后 total_elems），打印
`run_vision` 里 `outputs[0].size / sizeof(float)` 即可得到总 float 数，
再结合模型转换脚本确定两维。

## 5. 编译

板端（aarch64）安装好 rknn-llm 后：

```bash
cmake -S . -B build_qwen \
  -DRKLLM_PATH=/path/to/rknn-llm/install \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_qwen -j4
```

- 找到 `rkllm.h` 与 `librkllmrt.so` 会自动 `add_definitions(-DQWEN_ENABLED)`
- 未找到则 Qwen 功能静默禁用，双摄/推流功能不受影响

## 6. 运行

```bash
./app /dev/video11 /dev/video20 rtmp://192.168.1.30:1935/live/app 1280 720 30 \
  ../model/yolov5s.rknn 0 \
  ../model/Qwen3-VL-2B_llm_w8a8_rk3588.rkllm \
  ../model/Qwen3-VL-2B_vision_rk3588.rknn
```

参数：argv[8]=运行秒数(0=无限)，argv[9]=RKLLM 语言模型路径，argv[10]=视觉编码器路径。

## 7. 关键帧调度策略（main.cpp 内，可调）

当前实现：**检测到目标 且 距上次投喂 ≥ 2 秒** 时投喂一帧。Qwen 推理慢（秒级），
用"最新帧覆盖"单槽避免任务堆积。可按需改成：

- 纯定时（不管有无目标）
- 事件驱动（如车辆框面积/位置触发、车道偏离触发）
- 调整间隔 `qwen_elapsed >= 2000` 的阈值

## 8. 风险 JSON 输出示例

```json
{"risk_level":"high","risk_reasons":["左前方车辆变道"],"objects":["car","person"],"actions":["减速","保持车道"]}
```

输出当前直接打印到 stderr（`[Qwen] risk analysis: ...`）。下游可改为写日志、
回注 OSD 到推流画面、或对接告警/上位机。

## 9. 已知边界

- Qwen 推理是 **同步串行** 的（RKLLM 单实例），一次只能分析一帧；这是刻意的，
  用"最新帧优先 + 丢弃旧帧"规避延迟累积
- 模型加载后内存占用约 3~4GB（W8A8），建议板载内存 ≥ 8GB
- 失败/超时静默降级，不影响主推流管线
