#ifndef _STREAMER_H_
#define _STREAMER_H_

#ifdef __cplusplus
extern "C" {
#endif

// 初始化流媒体推送器
// width: 视频宽度
// height: 视频高度
// fps: 帧率
// bitrate: 比特率
// rtmp_url: RTMP推流地址
int init_streamer(int width, int height, int fps, int bitrate, const char *rtmp_url);

// 处理一帧图像数据
// frame_data: 图像数据
// frame_size: 图像数据大小
int process_frame(uint8_t *frame_data, int frame_size);

// 关闭流媒体推送器
void close_streamer();

#ifdef __cplusplus
}
#endif

#endif /* _STREAMER_H_ */ 