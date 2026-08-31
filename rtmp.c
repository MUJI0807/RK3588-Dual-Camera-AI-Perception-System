#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
#include <libavutil/time.h>
#include <libavcodec/avcodec.h>
#include <string.h>
#include <stdlib.h>
#include "rtmp.h"
#include <unistd.h>  // 用于sleep函数
                
static AVPacket pkt = {0};    // 全局AVPacket用于发送数据,AVPacket 是 FFmpeg 里“压缩后的媒体数据包”
static AVFormatContext *ofmt_ctx = NULL;  // 输出格式上下文，用于RTMP推送，ofmt_ctx为空指针表示未初始化
static int frame_index=0;       // 帧计数器初始化为0
static int64_t stream_start_us = AV_NOPTS_VALUE;
static int64_t last_pts_ms = AV_NOPTS_VALUE;
static int64_t nominal_frame_duration_ms = 33;

/*
 * 从 MPP 输出的 H.264 码流中解析 NALU 类型（跳过 start code）。
 * 返回 NALU 类型（1~12），无法识别返回 -1。
 * 用于正确设置 AVPacket 的 AV_PKT_FLAG_KEY：只有 IDR 帧（5）才真正
 * 是关键帧；SPS(7)/PPS(8) 属于序列头。把所有帧都标记为关键帧会导致
 * 播放端把 P 帧当 I 帧解码，出现花屏/卡顿。
 */
static int find_nalu_type(const uint8_t *data, int size)
{
    int offset = 0;
    if (data == NULL || size <= 0) return -1;
    if (size >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1)
        offset = 4;
    else if (size >= 3 && data[0] == 0 && data[1] == 0 && data[2] == 1)
        offset = 3;
    else
        return -1;
    if (size <= offset) return -1;
    return data[offset] & 0x1f;
}

/*
 * 发送一帧编码码流到 RTMP 输出（通过 FFmpeg muxer）
 * 将编码器输出的 H.264/H.265 码流数据封装为 AVPacket，
 * 设置时间戳（PTS/DTS）与持续时长，然后调用 av_write_frame()
 * 写入输出格式上下文（通常为 flv/rtmp）。
 *
 * @param data 编码后的码流数据指针（指向一帧/一个packet的码流数据，通常是NALU集合）
 * @param size 编码后的码流数据长度（单位：字节）
 * @return 成功返回0，失败返回-1
 */ 
int write_frame(uint8_t*data,int size)
{
        // 使用真实单调时钟，而不是把所有输入强行伪装成 30fps。
        const int64_t now_us = av_gettime_relative();
        if (stream_start_us == AV_NOPTS_VALUE) {
                stream_start_us = now_us;
        }
        int64_t pts = (now_us - stream_start_us) / 1000;
        if (last_pts_ms != AV_NOPTS_VALUE && pts <= last_pts_ms) {
                pts = last_pts_ms + 1;
        }
        const int64_t frame_duration = last_pts_ms == AV_NOPTS_VALUE
                ? nominal_frame_duration_ms
                : FFMAX((int64_t)1, pts - last_pts_ms);
        last_pts_ms = pts;

        pkt.size = size;      //设置 AVPacket 的数据长度
        pkt.data = data;      //设置 AVPacket 的数据指针
        // 关键帧标志必须按 NALU 类型设置：只有 IDR(5)/SPS(7)/PPS(8) 是关键帧相关，
        // 全部标 0x01 会把 P 帧当关键帧，导致播放端花屏。
        pkt.flags = 0;
        {
            const int nalu = find_nalu_type(data, size);
            if (nalu == 5 || nalu == 7 || nalu == 8) {
                pkt.flags = AV_PKT_FLAG_KEY;
            }
        }
        pkt.stream_index = 0; // 设置 AVPacket 标志位
        pkt.pts = pts;        // 设置 AVPacket 的显示时间戳
        pkt.dts = pts;        // 无 B 帧时 DTS 与 PTS 相同
        pkt.duration = frame_duration;    // 设置 AVPacket 的持续时长
        if (av_write_frame(ofmt_ctx, &pkt) < 0) {
                printf("Error muxing packet\n");  // 提示写入 AVPacket 失败
                return -1;
        }
        return 0;
}

/**init_rtmp_streamer
 * 初始化RTMP流媒体推送器
 * @param stream RTMP服务器地址
 * @param config RTMPStreamerConfig结构体，包含SPS/PPS数据和其他参数
 * @return 成功返回0，失败返回-1
 */
int init_rtmp_streamer(char* stream, RtmpContext *config)
{       
        printf("开始初始化RTMP流媒体推送...\n");
        printf("RTMP地址: %s\n", stream);
        printf("SPS/PPS数据大小: %d bytes\n", config->extradata_size);

        stream_start_us = AV_NOPTS_VALUE;
        last_pts_ms = AV_NOPTS_VALUE;
        nominal_frame_duration_ms = FFMAX((int64_t)1, 1000 / FFMAX(1, config->fps));

        int ret;  // 用于存储函数返回值并判断是否错误
        if((ret = avformat_network_init()) < 0)
        {
                fprintf(stderr, "avformat_network_init failed!");
                return -1;
        }

        printf("网络初始化成功\n");
        
        // 创建输出格式上下文，使用FLV格式（RTMP通常使用FLV封装）
        avformat_alloc_output_context2(&ofmt_ctx,NULL,"flv",stream);  // &ofmt_ctx是输出格式上下文的地址,NULL表示自动选择格式
        if(!ofmt_ctx)  // 如果创建失败
        {
                fprintf(stderr, "Could not create output context\n");
                return -1;
        }

        printf("创建输出上下文成功\n");

        // 创建新的输出流avformat_new_stream
        AVStream *out_stream = avformat_new_stream(ofmt_ctx,NULL); // NULL表示使用默认编码器
        if(! out_stream)
        {
                printf("Failed allocating output stream!\n");
                goto end;   // 跳转到清理代码
        }

        // 毫秒时间基配合 write_frame 的单调时钟 PTS，支持可变帧间隔。
        out_stream->time_base = av_make_q(1, 1000);

        // 创建编码器上下文avcodec_alloc_context3
        AVCodecContext *o_codec_ctx = avcodec_alloc_context3(NULL); // NULL表示使用默认编码器
        if (!o_codec_ctx) { // 如果创建失败
                printf("Failed to allocate codec context\n");
                goto end;
        }

        // 配置H.264编码器参数
        o_codec_ctx->codec_id = config->codec_id;       // 设置编码器ID
        o_codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;   // 设置编码器类型为视频
        o_codec_ctx->codec_tag = 0;                     // 设置编码器标签为0
        o_codec_ctx->pix_fmt = config->pix_fmt;         // 设置像素格式
        o_codec_ctx->width = config->width;             // 设置视频宽度
        o_codec_ctx->height = config->height;           // 设置视频高度
        o_codec_ctx->time_base = av_make_q(1, 1000);
        o_codec_ctx->framerate = av_make_q(config->fps, 1);  // 设置帧率
        o_codec_ctx->gop_size = config->fps * 2;             // 设置GOP大小为2秒
        o_codec_ctx->max_b_frames = config->max_b_frames;    // 设置最大B帧数,该参数决定 GOP（帧结构）中最多可以插入多少个 B 帧，
        o_codec_ctx->profile = config->profile;              // 设置H.264 profile
        o_codec_ctx->level = config->level;                  // 设置H.264 level
        o_codec_ctx->extradata = config->extradata;          // 设置SPS/PPS数据
        o_codec_ctx->extradata_size = config->extradata_size;   // 设置SPS/PPS数据大小

        printf("  输出格式: %s\n", ofmt_ctx->oformat->name);
        printf("  帧率: %d/%d\n", o_codec_ctx->framerate.num, o_codec_ctx->framerate.den);
        printf("设置编码器参数: 分辨率 %dx%d\n", o_codec_ctx->width, o_codec_ctx->height);
        
        if (!out_stream->codecpar) { // 如果输出流的编码参数为空
                printf("Error: out_stream->codecpar is NULL\n");
                goto end;
        }

        printf("开始复制编码器参数到流中...\n");
        ret = avcodec_parameters_from_context(out_stream->codecpar, o_codec_ctx); // 从编码器上下文复制参数到输出流,
        // out_stream->codecpar是输出流的编码参数结构体
        // o_codec_ctx是编码器上下文结构体
        if (ret < 0) {
                printf("Failed to copy codec parameters from encoder context.\n");
                goto end;
        }
        else
        {
                printf("ok to copy codec parameters from encoder context.\n");
        }
        out_stream->codecpar->codec_tag = 0; // FLV编码器要求codec_tag为0

        printf("打开RTMP URL %s\n", stream);
        if (!(ofmt_ctx->oformat->flags & AVFMT_NOFILE)) {  // 如果输出格式不包含 AVFMT_NOFILE 标志
                // 限制重试次数：RTMP 服务器不可达时不能无限阻塞（否则主线程永久挂起）。
                int retry_count = 0;
                const int max_retries = 3;
                while (1) {
                        ret = avio_open(&ofmt_ctx->pb, stream, AVIO_FLAG_WRITE);  // 打开RTMP输出URL进行写入
                        // &ofmt_ctx->pb是输出格式上下文的IO上下文指针, stream是RTMP服务器地址, AVIO_FLAG_WRITE表示以写入模式打开
                        if (ret >= 0) {
                                break;
                        }
                        retry_count++;  // 连接失败，增加重试计数
                        printf("无法连接到RTMP服务器 '%s'，5秒后重试... (第%d次尝试)\n", stream, retry_count);
                        if (retry_count >= max_retries) {
                                fprintf(stderr, "RTMP服务器连接失败，已重试 %d 次，放弃初始化\n", max_retries);
                                goto end;   // 跳转到清理代码
                        }
                        sleep(5);
                }
        }
        printf("打开输出URL成功\n");

        ret = avformat_write_header(ofmt_ctx, NULL);  // 写入文件头，NULL表示使用默认选项,avformat_write_header函数用于写入输出文件的头信息
        if (ret < 0) {
                printf( "Error occurred when opening output URL\n");
                goto end;
        }
        printf("写入文件头成功\n");

        avcodec_free_context(&o_codec_ctx);
        o_codec_ctx = NULL;

        printf("创建输出流成功\n");
        printf("创建编码器上下文成功\n");
        printf("复制编码器参数到流中成功\n");
        printf("格式信息打印完成\n");
        printf("RTMP流媒体推送初始化完成\n");
        printf("RTMP流媒体推送器关键参数:\n");

        return 0;

        end: // 清理代码
        // 失败路径释放编码器上下文（成功路径已在上面释放并置 NULL，这里判空保护）
        if (o_codec_ctx) {
                avcodec_free_context(&o_codec_ctx);
                o_codec_ctx = NULL;
        }
        if (ofmt_ctx && !(ofmt_ctx->oformat->flags & AVFMT_NOFILE)) { // 如果输出格式上下文存在且不包含 AVFMT_NOFILE 标志
                // avio_open 失败时 pb 可能为 NULL，avio_close(NULL) 会崩溃，必须判空
                if (ofmt_ctx->pb) {
                        avio_close(ofmt_ctx->pb); // 关闭IO上下文
                        ofmt_ctx->pb = NULL;
                }
        }
        if (ofmt_ctx) {
                avformat_free_context(ofmt_ctx);  // 释放输出格式上下文
                ofmt_ctx = NULL;
        }
        if (ret < 0 && ret != AVERROR_EOF) {  // 如果发生错误且不是文件结束错误
                printf( "Error occurred.\n");  
                printf("初始化失败，开始清理资源...\n");
                return -1;    // 返回错误码
        }
}
