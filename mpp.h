#ifndef _MPP_H
#define _MPP_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <string.h>
#include <errno.h>
#include <rockchip/rk_mpi.h>

#ifdef __cplusplus
        extern "C"
        {
#endif

#define MPP_ALIGN(x, a)         (((x)+(a)-1)&~((a)-1))

// [MIN-FIX] 码率兜底范围（MPP 日志里提示 1K~200M）
#define MPP_BPS_MIN_LIMIT       (1000)
#define MPP_BPS_MAX_LIMIT       (200000000)

//rk系列芯片枚举
typedef enum RockchipSocType_e {
    ROCKCHIP_SOC_AUTO,
    ROCKCHIP_SOC_RK3036,
    ROCKCHIP_SOC_RK3066,
    ROCKCHIP_SOC_RK3188,
    ROCKCHIP_SOC_RK3288,
    ROCKCHIP_SOC_RK312X,
    ROCKCHIP_SOC_RK3368,
    ROCKCHIP_SOC_RK3399,
    ROCKCHIP_SOC_RK3228H,
    ROCKCHIP_SOC_RK3328,
    ROCKCHIP_SOC_RK3228,
    ROCKCHIP_SOC_RK3229,
    ROCKCHIP_SOC_RV1108,
    ROCKCHIP_SOC_RV1109,
    ROCKCHIP_SOC_RV1126,
    ROCKCHIP_SOC_RK3326,
    ROCKCHIP_SOC_RK3128H,
    ROCKCHIP_SOC_PX30,
    ROCKCHIP_SOC_RK1808,
    ROCKCHIP_SOC_RK3566,
    ROCKCHIP_SOC_RK3567,
    ROCKCHIP_SOC_RK3568,
    ROCKCHIP_SOC_RK3588,
    ROCKCHIP_SOC_RK3528,
    ROCKCHIP_SOC_RK3562,
    ROCKCHIP_SOC_RK3576,
    ROCKCHIP_SOC_RV1126B,
    ROCKCHIP_SOC_BUTT,
} RockchipSocType;

typedef struct {        /// SPS头信息结构体
        uint8_t *data;
        uint32_t size;
} SpsHeader;

/** 
 * @brief MPP编码器上下文结构体
 * 
 * 该结构体包含了MPP编码器运行所需的所有参数和状态信息
 */
typedef struct {
        // 基础MPP上下文
        MppCtx ctx;          ///< MPP上下文句柄
        MppApi *mpi;         ///< MPP API接口指针
        RK_S32 chn;          ///< 通道号

        // 全局流程控制标志
        RK_U32 frm_eos;      ///< 帧结束标志
        RK_U32 pkt_eos;      ///< 包结束标志
        RK_U32 frm_pkt_cnt;  ///< 当前帧的包计数
        RK_S32 frame_num;    ///< 需要编码的总帧数
        RK_S32 frame_count;  ///< 已编码的帧数
        RK_U64 stream_size;  ///< 已编码的流大小
        volatile RK_U32 loop_end;  ///< 循环结束标志

        // 编码器配置
        MppEncCfg cfg;       ///< 编码器配置
        MppEncPrepCfg   prep_cfg;  // 预处理（Prepare / Preprocess）配置
        MppEncRcCfg     rc_cfg;    // 码率控制（Rate Control）配置
        MppEncCodecCfg  codec_cfg;      // 编码器特定配置
        MppEncSliceSplit split_cfg;     // 切片分割配置
        MppEncOSDPltCfg osd_plt_cfg;    // OSD调色板配置
        MppEncOSDPlt    osd_plt;        // OSD调色板
        MppEncOSDData   osd_data;       // OSD数据
        MppEncROICfg    roi_cfg;        // ROI配置

        // 输入/输出缓冲区
        MppBufferGroup buf_grp;  ///< 缓冲区组
        MppBuffer frm_buf;       ///< 帧缓冲区
        MppBuffer pkt_buf;       ///< 包缓冲区
        MppBuffer md_info;      ///< 元数据缓冲区
        MppEncSeiMode sei_mode;  ///< SEI模式
        MppEncHeaderMode header_mode; ///< 头信息模式

        // 资源分配参数
        RK_U32 width;           ///< 图像宽度
        RK_U32 height;          ///< 图像高度
        RK_U32 hor_stride;      ///< 水平步长
        RK_U32 ver_stride;      ///< 垂直步长
        MppFrameFormat fmt;     ///< 帧格式
        MppCodingType type;     ///< 编码类型
        RK_S32 loop_times;      ///< 循环编码次数

        // 资源大小
        size_t header_size;     ///< 头信息大小
        size_t frame_size;      ///< 帧大小
        size_t mdinfo_size;     ///< 元数据大小
        size_t packet_size;     ///< 包大小

        // 码率控制参数
        RK_S32 fps_in_flex;     ///< 输入帧率灵活模式
        RK_S32 fps_in_den;      ///< 输入帧率分母
        RK_S32 fps_in_num;      ///< 输入帧率分子
        RK_S32 fps_out_flex;    ///< 输出帧率灵活模式
        RK_S32 fps_out_den;     ///< 输出帧率分母
        RK_S32 fps_out_num;     ///< 输出帧率分子
        RK_S32 bps;             ///< 目标码率
        RK_S32 bps_max;         ///< 最大码率
        RK_S32 bps_min;         ///< 最小码率
        RK_S32 rc_mode;         ///< 码率控制模式
        RK_S32 gop_mode;        ///< GOP模式
        RK_S32 gop_len;         ///< GOP长度
        RK_S32 vi_len;          ///< IDR间隔长度
        RK_S32 scene_mode;      ///< 场景模式
        RK_S32 cu_qp_delta_depth;       ///< CU QP变化深度
        RK_S32 anti_flicker_str;        ///< 抗闪烁强度
        RK_S32 atr_str_i;       ///< ATR强度I帧
        RK_S32 atr_str_p;       ///< ATR强度P帧
        RK_S32 atl_str;        ///< ATL强度
        RK_S32 sao_str_i;       ///< SAO强度I帧
        RK_S32 sao_str_p;       ///< SAO强度P帧
        RK_S64 first_frm;       ///< 第一帧时间戳
        RK_S64 first_pkt;       ///< 第一个包时间戳

        // 回调函数
        int (*write_frame)(uint8_t*data,int size);  ///< 写入编码后帧数据的回调函数
        int (*init_mpp)(void *mpp_enc_data);        ///< 初始化MPP的回调函数
        _Bool (*process_image)(uint8_t *p, int size,void *mpp_enc_data);  ///< 处理图像的回调函数
        _Bool (*get_header)(void *mpp_enc_data,SpsHeader *sps_header);  ///< 获取头信息的回调函数
        void (*close)(void* ctx);                   ///< 关闭MPP的回调函数

} MppContext;   // MPP编码器上下文结构体

MppContext* alloc_mpp_context();        // 分配MPP编码器上下文结构体

#ifdef __cplusplus      // extern "C"
        }
#endif                  // __cplusplus

#endif /* !_MPP_H */    // _MPP_H


/**
 * @brief 初始化MPP编码器
 * 该函数负责创建MPP编码器实例、配置编码参数、分配必要的缓冲区，并使编码器进入可工作状态。
 * @param mpp_enc_data MPP上下文指针，调用者需预先设置好 width/height/fps/rc/bps/type/fmt 等字段
 * @return int 成功返回0，失败返回非0错误码
 */