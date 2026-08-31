#include <stdio.h>      // printf 等标准IO
#include <stdlib.h>     // malloc/free 等内存管理
#include <string.h>    
#include "mpp.h"        // 封装的 MPP 编码器接口：MppContext / alloc_mpp_context / ctx->init_mpp 等
#include "rtmp.h"       // RTMP 推流接口：RtmpContext / init_rtmp_streamer / write_frame 等

/*
 * StreamerContext：把 “MPP编码器 + RTMP推流器 + SPS/PPS头” 打包成一个全局单例上下文
 */
typedef struct {
    MppContext *mpp_ctx;     // 指向 MPP 编码器上下文
    RtmpContext *rtmp_ctx;   // 指向 RTMP 推流配置上下文
    SpsHeader sps_header;    // 保存 SPS/PPS（header.data/header.size），用于给 RTMP/FFmpeg 的 extradata
    int is_initialized;      // 0=未初始化，1=已初始化，用作简单状态机
} StreamerContext;

/*
 * g_streamer_ctx：全局静态上下文，零初始化
 * {0} 会把所有指针设为 NULL，is_initialized=0，sps_header.data=NULL 等
 */
static StreamerContext g_streamer_ctx = {0};

/*
 * init_streamer：初始化“MPP编码 + RTMP推流”
 * width/height/fps/bitrate/rtmp_url 来自（main.cpp）
 */
int init_streamer(int width, int height, int fps, int bitrate, const char *rtmp_url) {
    if (g_streamer_ctx.is_initialized) { // 如果已经初始化过，就直接返回成功（避免重复初始化）
        return 0;
    }

    // 初始化MPP编码器
    g_streamer_ctx.mpp_ctx = alloc_mpp_context(); // 在堆上分配 MppContext，并把函数指针绑定好
    if (!g_streamer_ctx.mpp_ctx) {                // 分配失败直接返回
        printf("Failed to allocate MPP context\n");
        return -1;
    }

    // 配置MPP编码器参数
    g_streamer_ctx.mpp_ctx->width = width;        // 输入图像宽
    g_streamer_ctx.mpp_ctx->height = height;      // 输入图像高

    g_streamer_ctx.mpp_ctx->fps_in_flex = 1;      // 允许双摄推理后的输入间隔变化
    g_streamer_ctx.mpp_ctx->fps_in_num = fps;     // 输入帧率分子
    g_streamer_ctx.mpp_ctx->fps_in_den = 1;       // 输入帧率分母

    g_streamer_ctx.mpp_ctx->fps_out_flex = 1;      // 输出帧率使用参考值，不强制补帧
    g_streamer_ctx.mpp_ctx->fps_out_num = fps;    // 输出帧率分子
    g_streamer_ctx.mpp_ctx->fps_out_den = 1;      // 输出帧率分母
    
    g_streamer_ctx.mpp_ctx->bps = bitrate;        // 目标码率（单位 bit/s）
    g_streamer_ctx.mpp_ctx->gop_len = fps * 2;    // GOP 长度（2秒一个 IDR/I 帧）
    g_streamer_ctx.mpp_ctx->write_frame = write_frame; // 编码后输出回调：写到 RTMP（rtmp.c 里实现）
    g_streamer_ctx.mpp_ctx->type = MPP_VIDEO_CodingAVC; // 选择 H.264
    g_streamer_ctx.mpp_ctx->fmt = MPP_FMT_YUV420SP;     // 选择 NV12的 YUV420SP
    g_streamer_ctx.mpp_ctx->rc_mode = MPP_ENC_RC_MODE_CBR; // 码控：CBR

    printf("初始化流媒体推送器...\n");
    printf("视频参数: %dx%d, %d fps, %d bps\n", width, height, fps, bitrate);
    printf("RTMP地址: %s\n", rtmp_url);
    
    // 初始化MPP
    int ret = g_streamer_ctx.mpp_ctx->init_mpp(g_streamer_ctx.mpp_ctx); // 真正创建 MPP ctx/mpi/buf/cfg 等
    if(ret != 0) {                              // ret!=0 表示失败（你 mpp.c 的 init_mpp 返回 ret）
        printf("mpp init fail!\n");
        // 修复：失败必须立即返回并释放上下文，否则后续 get_header/process_image
        // 会在未初始化的编码器上调用，导致崩溃或非法码流。
        g_streamer_ctx.mpp_ctx->close(g_streamer_ctx.mpp_ctx);
        g_streamer_ctx.mpp_ctx = NULL;
        return -1;
    } else {
        printf("mpp init success!\n");
    }

    // 获取SPS/PPS信息
    if (!g_streamer_ctx.mpp_ctx->get_header(g_streamer_ctx.mpp_ctx, &g_streamer_ctx.sps_header)) {
        // get_header 返回 false 表示失败
        printf("Failed to get SPS/PPS header\n");
        g_streamer_ctx.mpp_ctx->close(g_streamer_ctx.mpp_ctx);
        g_streamer_ctx.mpp_ctx = NULL;
        return -1;
    }
    
    // 为 RTMP 上下文分配内存（保存推流需要的参数）
    g_streamer_ctx.rtmp_ctx = (RtmpContext *)malloc(sizeof(RtmpContext));
    // ⚠️ 风险：这里没判空；malloc 失败会导致后面解引用崩溃
    if (!g_streamer_ctx.rtmp_ctx) {
        printf("Failed to allocate RTMP context\n");
        return -1;
    }

    g_streamer_ctx.rtmp_ctx->codec_id = AV_CODEC_ID_H264;          // FFmpeg 编码器ID：H.264
    g_streamer_ctx.rtmp_ctx->pix_fmt = AV_PIX_FMT_NV12;            // FFmpeg 像素格式：NV12（注意：这只是描述，数据是真正由 MPP 给出）
    g_streamer_ctx.rtmp_ctx->width = width;                        // 宽
    g_streamer_ctx.rtmp_ctx->height = height;                      // 高
    g_streamer_ctx.rtmp_ctx->fps = fps;                            // 帧率
    g_streamer_ctx.rtmp_ctx->max_b_frames = 0;                     // 禁用B帧（低延迟）
    g_streamer_ctx.rtmp_ctx->profile = FF_PROFILE_H264_HIGH;       // H.264 profile：High
    // 双摄合成后的最终输出是 1280x720，H.264 Level 3.1 足够。
    g_streamer_ctx.rtmp_ctx->level = 31;
    g_streamer_ctx.rtmp_ctx->extradata = g_streamer_ctx.sps_header.data;      // SPS/PPS（由 MPP get_header 拷贝出来的 malloc 内存）
    g_streamer_ctx.rtmp_ctx->extradata_size = g_streamer_ctx.sps_header.size; // SPS/PPS 大小

    printf("初始化RTMP...\n");
    // 初始化RTMP
    if (init_rtmp_streamer((char*)rtmp_url, g_streamer_ctx.rtmp_ctx) < 0) {
        // init_rtmp_streamer 内部会 avformat_alloc_output_context2/avio_open/avformat_write_header 等
        printf("Failed to initialize RTMP streamer\n");
        // 修复：失败路径统一清理，避免 mpp_ctx / sps_header / rtmp_ctx 泄漏
        if (g_streamer_ctx.mpp_ctx) {
            g_streamer_ctx.mpp_ctx->close(g_streamer_ctx.mpp_ctx);
            g_streamer_ctx.mpp_ctx = NULL;
        }
        if (g_streamer_ctx.sps_header.data) {
            free(g_streamer_ctx.sps_header.data);
            g_streamer_ctx.sps_header.data = NULL;
            g_streamer_ctx.sps_header.size = 0;
        }
        if (g_streamer_ctx.rtmp_ctx) {
            free(g_streamer_ctx.rtmp_ctx);
            g_streamer_ctx.rtmp_ctx = NULL;
        }
        return -1;
    }
    printf("初始化RTMP成功\n");

    g_streamer_ctx.is_initialized = 1; // 标记初始化完成
    return 0;
}

/*
 * process_frame：喂一帧 NV12 给 MPP 编码器，并通过回调 write_frame 推到 RTMP
 * frame_data：NV12 数据指针
 * frame_size：NV12 数据大小（应当等于 stride*stride*3/2；至少不能为0）
 */
int process_frame(uint8_t *frame_data, int frame_size) {
    if (!g_streamer_ctx.is_initialized || !g_streamer_ctx.mpp_ctx) {
        // 未初始化或 MPP ctx 为空，直接失败
        return -1;
    }

    // 使用MPP进行编码
    if (!g_streamer_ctx.mpp_ctx->process_image(frame_data, frame_size, g_streamer_ctx.mpp_ctx)) {
        printf("Failed to process frame\n");
        return -1;
    }

    return 0;
}

/*
 * close_streamer：释放资源
 */
void close_streamer() {
    if (g_streamer_ctx.mpp_ctx) {
        // 调用 mpp_close：内部会 reset/destroy ctx，并 free(ctx)（即释放 mpp_ctx 本体）
        g_streamer_ctx.mpp_ctx->close(g_streamer_ctx.mpp_ctx);
        g_streamer_ctx.mpp_ctx = NULL; // 避免悬空指针
    }
    
    if (g_streamer_ctx.sps_header.data) {
        // 释放 get_header 里 malloc 的 SPS/PPS 缓冲
        free(g_streamer_ctx.sps_header.data);
        g_streamer_ctx.sps_header.data = NULL;  // 避免悬空指针
    }
    
    // 释放rtmp_ctx内存
    if (g_streamer_ctx.rtmp_ctx) {
        // 这里只 free rtmp_ctx 结构体本身
        free(g_streamer_ctx.rtmp_ctx);
        g_streamer_ctx.rtmp_ctx = NULL;      // 避免悬空指针
    }
    
    g_streamer_ctx.is_initialized = 0; // 置回未初始化状态
}
