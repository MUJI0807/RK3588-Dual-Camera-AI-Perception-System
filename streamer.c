#include <stdio.h>      // printf 等标准IO
#include <stdlib.h>     // malloc/free 等内存管理
#include <string.h>    
#include "mpp.h"        // 封装的 MPP 编码器接口：MppContext / alloc_mpp_context / ctx->init_mpp 等
#include "rtmp.h"       // RTMP 推流接口：RtmpContext / init_rtmp_streamer / write_frame 等

// ================================ 日志降频配置 ================================
// 说明：推流链路属于高频路径，每帧 printf 会明显拖慢 RK3588 端侧实时性。
// ENABLE_STREAMER_VERBOSE_LOG = 0：关闭每帧日志，只保留错误日志和初始化日志。
// ENABLE_STREAMER_VERBOSE_LOG = 1：每 STREAMER_LOG_INTERVAL 帧打印一次状态。
#define ENABLE_STREAMER_VERBOSE_LOG 1
#define STREAMER_LOG_INTERVAL 500


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

    g_streamer_ctx.mpp_ctx->fps_in_flex = 0;      // 固定输入帧率模式（0固定，1可变）
    g_streamer_ctx.mpp_ctx->fps_in_num = fps;     // 输入帧率分子
    g_streamer_ctx.mpp_ctx->fps_in_den = 1;       // 输入帧率分母

    g_streamer_ctx.mpp_ctx->fps_out_flex = 0;      // 固定输出帧率模式（0固定，1可变）
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
    } else {
        printf("mpp init success!\n");
    }

    // 获取SPS/PPS信息
    if (!g_streamer_ctx.mpp_ctx->get_header(g_streamer_ctx.mpp_ctx, &g_streamer_ctx.sps_header)) {
        // get_header 返回 false 表示失败
        printf("Failed to get SPS/PPS header\n");
        return -1;
    }
    
    // 为 RTMP 上下文分配内存（保存推流需要的参数）
    g_streamer_ctx.rtmp_ctx = (RtmpContext *)malloc(sizeof(RtmpContext));
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
    g_streamer_ctx.rtmp_ctx->level = 31;                           // H.264 level：31（大致对应 720p@30 的常见档位，仅作标记）
    g_streamer_ctx.rtmp_ctx->extradata = g_streamer_ctx.sps_header.data;      // SPS/PPS（由 MPP get_header 拷贝出来的 malloc 内存）
    g_streamer_ctx.rtmp_ctx->extradata_size = g_streamer_ctx.sps_header.size; // SPS/PPS 大小

    printf("初始化RTMP...\n");
    // 初始化RTMP
    if (init_rtmp_streamer((char*)rtmp_url, g_streamer_ctx.rtmp_ctx) < 0) {
        // init_rtmp_streamer 内部会 avformat_alloc_output_context2/avio_open/avformat_write_header 等
        printf("Failed to initialize RTMP streamer\n");
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
    static int frame_cnt = 0;
    frame_cnt++;

    if (!g_streamer_ctx.is_initialized || !g_streamer_ctx.mpp_ctx) {
        // 未初始化或 MPP ctx 为空，直接失败
        return -1;
    }

#if ENABLE_STREAMER_VERBOSE_LOG
    if (frame_cnt % STREAMER_LOG_INTERVAL == 0) {
        printf("[streamer] input frame=%d, size=%d bytes, fmt=nv12\n", frame_cnt, frame_size);
    }
#endif
    
    // 使用MPP进行编码
    if (!g_streamer_ctx.mpp_ctx->process_image(frame_data, frame_size, g_streamer_ctx.mpp_ctx)) {
        printf("Failed to process frame, frame=%d\n", frame_cnt);
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
