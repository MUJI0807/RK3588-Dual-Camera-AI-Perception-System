#ifndef RTMP_H
#define RTMP_H

#include <libavcodec/avcodec.h>

/**
 * RTMP流媒体参数结构体
 */
typedef struct {
    // 视频基本参数
    int width;                  // 视频宽度
    int height;                 // 视频高度
    int max_b_frames;          // 最大B帧数
    
    int fps;                    // 帧率

    // 编码器参数
    enum AVCodecID codec_id;    // 编码器ID
    enum AVPixelFormat pix_fmt; // 像素格式
    int profile;               // H.264 profile, 如 baseline, main, high
    int level;                 // H.264 level, 如 3.1, 4.0
    
    // SPS/PPS数据
    uint8_t *extradata;        // SPS/PPS数据
    uint32_t extradata_size;   // SPS/PPS数据大小
    
} RtmpContext;

/**
 * 初始化RTMP流媒体推送器
 * @param stream RTMP服务器地址
 * @param config 流媒体配置参数
 * @return 成功返回0，失败返回-1
 */
int init_rtmp_streamer(char* stream, RtmpContext *config);      // 初始化RTMP流媒体推送器

/**
 * 写入一帧数据
 * @param data 编码后的数据
 * @param size 数据大小
 * @return 成功返回0，失败返回-1
 */
int write_frame(uint8_t* data, int size);

#endif // RTMP_H