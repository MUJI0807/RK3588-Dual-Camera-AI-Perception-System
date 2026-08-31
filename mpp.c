/**
 * @file mpp.c
 * @brief MPP（Media Process Platform）编码器实现
 * 
 * 本文件实现了基于Rockchip MPP的视频编码功能，支持H.264编码。
 * 主要功能包括：
 * 1. MPP编码器的初始化和配置
 * 2. 视频帧的编码处理
 * 3. 编码器头信息的获取
 * 4. 资源的分配和释放
 * 
 * 备注：
 * - 本文件将“编码器实例 + 缓冲区 + 配置 + 回调接口”封装在 MppContext 结构体中；
 * - 外部调用者只需按模板设置参数并调用 ctx->init_mpp / ctx->process_image 等接口即可。
 */

//============================================使用模板============================================
    // //分配
    // mpp_ctx = alloc_mpp_context(); 调用你封装的 alloc_mpp_context()，在堆上分配并返回一个 MppContext 上下文对象指针。
    // // 配置MPP编码器参数
    // mpp_ctx->width = width;   设置输入视频帧的“图像宽度”（单位：像素）
    // mpp_ctx->height = height; 设置输入视频帧的“图像高度”

    // mpp_ctx->fps_in_flex = 0;  使用固定帧率模式：0 表示固定帧率；1 表示可变帧率（允许输入帧率动态变化）。
    // mpp_ctx->fps_in_num = fps;  MPP使用 fps_num / fps_den 来表示帧率；例如 30/1=30fps
    // mpp_ctx->fps_in_den = 1;

    // mpp_ctx->fps_in_flex = 0;  // 使用固定帧率模式。
    // mpp_ctx->fps_out_num = fps;  MPP使用 fps_num / fps_den 来表示帧率。
    // mpp_ctx->fps_out_den = 1;    设置输出帧率的“分母”
    
    // mpp_ctx->bps = bitrate;      设置目标码率
    // mpp_ctx->gop_len = fps * 2;  GOP长度为帧率的2倍：GOP 越小延迟更低但码率更高；GOP 越大：压缩更高效但随机访问较差。
    // mpp_ctx->write_frame = write_frame;   设置写入编码后帧数据的回调函数指针。
    // mpp_ctx->type = MPP_VIDEO_CodingAVC;  设置H.264编码
    // mpp_ctx->fmt = MPP_FMT_YUV420SP;      设置YUV420P格式
    // mpp_ctx->rc_mode = MPP_ENC_RC_MODE_CBR; 设置码率控制模式为“恒定码率”（CBR）

    // // 初始化MPP
    // mpp_ctx->init_mpp(mpp_ctx);
    
//============================================使用模板============================================

#include "mpp.h"  // 引入MPP接口与数据结构定义（MppContext、SpsHeader、MPP API 等）

// 声明内部函数
static void mpp_close(MppContext* ctx);  // 声明资源释放函数
// - ctx 类型：MppContext*（指向编码器上下文结构体的指针）
// - 作用：释放编码器实例、释放内部缓冲区，并回收 ctx 本体

static int init_mpp(MppContext *mpp_enc_data);  // 声明编码器初始化函数
// - mpp_enc_data 类型：MppContext*（输入/输出参数）
// - 输入：调用者在 init 之前需要填写 width/height/fps/rc/bps/type/fmt 等字段
// - 输出：函数会填充 ctx/mpi/cfg/buffer/stride/frame_size 等字段，使编码器进入可工作状态

static _Bool get_header(MppContext *mpp_enc_data, SpsHeader *sps_header);  // 声明获取SPS/PPS函数
// - mpp_enc_data 类型：MppContext*（输入参数）
//   必须已 init_mpp() 成功，否则 mpp_enc_data->ctx / mpi 等可能为空
// - sps_header 类型：SpsHeader*（输出参数）
//   用于承载编码头（H.264/H.265 的 SPS/PPS 或 VPS/SPS/PPS）
//   通常结构体包含：uint8_t* data; size_t size;
//   本函数内部会 malloc(sps_header->data)，调用者需要在使用完后 free(sps_header->data)
// - 返回值语义：1=成功，0=失败（[MIN-FIX] 修复原来失败也返回1的问题）

static _Bool process_image(uint8_t *p, int size, MppContext *mpp_enc_data);  // 声明单帧编码处理函数
// - p 类型：uint8_t*（输入参数）：指向原始图像帧数据的起始地址），需与 mpp_enc_data->fmt 匹配
// - size 类型：int（输入参数）：该帧数据字节数，通常应与 mpp_enc_data->frame_size 一致，否则 memcpy 可能越界或数据不完整
// - mpp_enc_data 类型：MppContext*（输入参数）：已初始化编码器上下文，内部包含 frame buffer / packet buffer / callback 等
// - 返回值语义（结合你当前实现）：1=继续编码，0=达到停止条件（如 EOS 或最大帧数）


// [MIN-FIX] 小工具：保证码率参数合法，避免 MPP 拒绝配置/出现奇怪 min/max
static inline RK_S32 clamp_bps(RK_S32 v)
{
    if (v < MPP_BPS_MIN_LIMIT) return MPP_BPS_MIN_LIMIT;
    if (v > MPP_BPS_MAX_LIMIT) return MPP_BPS_MAX_LIMIT;
    return v;
}


/**MppContext
 * @brief 分配并初始化MPP上下文
 * 创建MPP上下文结构体，并初始化所有回调函数指针
 * @return MppContext* 成功返回MPP上下文指针，失败返回NULL
 * - ctx 由 malloc 分配，属于堆内存对象；
 * - 调用者需要在结束时调用 ctx->close(ctx) 释放资源；
 */
MppContext * alloc_mpp_context()
{
    MppContext *ctx = (MppContext *)malloc(sizeof(MppContext));  // 申请一个MPP上下文结构体内存
    // 补充：ctx 可能为 NULL（内存不足），可在此处判空并返回 NULL
    if (!ctx) {
        return NULL;
    }

    // [MIN-FIX] 必须清零：否则 bps_min/max/rc 参数等会是随机垃圾值，引发 out-of-range / 段错
    memset(ctx, 0, sizeof(*ctx));

    ctx->init_mpp = init_mpp;                   // 绑定初始化函数指针
    ctx->close = mpp_close;                     // 绑定关闭释放函数
    ctx->get_header = get_header;               // 绑定获取编码头函数
    ctx->process_image = process_image;         // 绑定编码帧处理函数
    return ctx;                                 // 返回创建好的上下文对象
}

/**mpp_close
 * @brief 关闭并清理MPP编码器资源
 * 该函数负责清理所有MPP相关的资源，包括：
 * 1. 重置MPP上下文
 * 2. 销毁MPP上下文
 * 3. 释放帧缓冲区
 * 4. 释放MPP上下文结构体
 * 
 * @param ctx MPP上下文指针
 * 补充说明：
 * - ctx->ctx：MPP 内部编码器实例句柄（由 mpp_create/mpp_init 创建）
 * - ctx->mpi：MPP API 函数表接口（包含 control/encode_put_frame 等函数指针）
 * - ctx->frm_buf：输入帧缓冲区（存放原始图像数据）
 */
static void mpp_close(MppContext* ctx)
{
    if (!ctx) return;

    MPP_RET ret = MPP_OK;  // 保存MPP接口返回状态

    // 重置MPP上下文
    if (ctx->mpi && ctx->ctx) {
        ret = ctx->mpi->reset(ctx->ctx); 
        if (ret)
        {
                printf("mpi->reset failed\n");  // 重置失败时打印错误
        }
    }

    // 销毁MPP上下文
    if (ctx->ctx)
    {
            mpp_destroy(ctx->ctx);  // 销毁MPP上下文实例（释放 MPP 内部分配的资源）
            ctx->ctx = NULL;  // 防止后续误用
    }

    // 释放帧缓冲区
    if (ctx->frm_buf)
    {
            mpp_buffer_put(ctx->frm_buf);  // 释放/归还帧缓冲区（归还给 buffer group）
            ctx->frm_buf = NULL;  // 清空指针避免重复释放
    }

    // [MIN-FIX] 释放输出包缓冲区（你原来没释放）
    if (ctx->pkt_buf)
    {
            mpp_buffer_put(ctx->pkt_buf);
            ctx->pkt_buf = NULL;
    }

    // [MIN-FIX] 释放 cfg（你原来没释放，长期运行会泄漏）
    if (ctx->cfg)
    {
            mpp_enc_cfg_deinit(ctx->cfg);
            ctx->cfg = NULL;
    }

    // [MIN-FIX] 释放 buffer group（可选但建议）
    if (ctx->buf_grp)
    {
            mpp_buffer_group_put(ctx->buf_grp);
            ctx->buf_grp = NULL;
    }

    // 释放MPP上下文结构体
    free(ctx);  // 释放alloc_mpp_context申请的结构体内存
}


/**mpp_get_soc_type
 * @brief 获取当前SoC类型
 * 返回当前使用的Rockchip SoC类型
 * @return RockchipSocType 返回SoC类型枚举值
 */
static int mpp_get_soc_type()
{
        return ROCKCHIP_SOC_RK3588;
}

/**init_mpp
 * @brief 初始化MPP编码器
 * 该函数完成MPP编码器的完整初始化流程，包括：
 * 1. 设置基本编码参数（分辨率、格式等）
 * 2. 初始化缓冲区
 * 3. 创建MPP上下文
 * 4. 配置编码器参数
 * 5. 设置码率控制
 * 6. 配置H.264编码参数
 * @param mpp_enc_data MPP上下文指针
 * @return int 成功返回0，失败返回错误码
 */
static int init_mpp(MppContext *mpp_enc_data)  // 初始化并配置 MPP 编码器
{
    MPP_RET ret = MPP_OK;  // 保存 MPP 接口返回值（0 通常表示成功）
    MppPollType timeout = MPP_POLL_BLOCK;  // 设置输出取包策略为阻塞等待（encode_get_packet 会阻塞直到有包）

    printf("start to init mpp...\n ");  // 打印初始化开始日志

    // [MIN-FIX] 兜底初始化：避免外部未设置导致非法值
    if (mpp_enc_data->width == 0 || mpp_enc_data->height == 0) {
        printf("invalid width/height: %u x %u\n", mpp_enc_data->width, mpp_enc_data->height);
        return MPP_NOK;
    }
    if (mpp_enc_data->fps_in_num <= 0)  mpp_enc_data->fps_in_num  = 30;
    if (mpp_enc_data->fps_in_den <= 0)  mpp_enc_data->fps_in_den  = 1;
    if (mpp_enc_data->fps_out_num <= 0) mpp_enc_data->fps_out_num = mpp_enc_data->fps_in_num;
    if (mpp_enc_data->fps_out_den <= 0) mpp_enc_data->fps_out_den = mpp_enc_data->fps_in_den;

    // [MIN-FIX] 码率兜底 + clamp，避免 out-of-range
    if (mpp_enc_data->bps <= 0) mpp_enc_data->bps = 1152000; // 默认 1.152Mbps（与你日志一致）
    mpp_enc_data->bps = clamp_bps(mpp_enc_data->bps);
    if (mpp_enc_data->bps_max <= 0) mpp_enc_data->bps_max = mpp_enc_data->bps * 11 / 10;
    if (mpp_enc_data->bps_min <= 0) mpp_enc_data->bps_min = mpp_enc_data->bps *  9 / 10;
    mpp_enc_data->bps_max = clamp_bps(mpp_enc_data->bps_max);
    mpp_enc_data->bps_min = clamp_bps(mpp_enc_data->bps_min);

    // 设置基本编码参数
    mpp_enc_data->hor_stride   = MPP_ALIGN(mpp_enc_data->width, 16);  // 宽度 stride 按 16 对齐（硬件常见要求）
    mpp_enc_data->ver_stride   = MPP_ALIGN(mpp_enc_data->height, 16); // 高度 stride 按 16 对齐

    // 根据编码的输入格式计算视频帧的大小
    switch (mpp_enc_data->fmt & MPP_FRAME_FMT_MASK) {  // 取 fmt 的格式部分（屏蔽掉非格式标志位）
    case MPP_FMT_YUV420SP:
    case MPP_FMT_YUV420P: {
        mpp_enc_data->frame_size = MPP_ALIGN(mpp_enc_data->hor_stride, 64) *
                                  MPP_ALIGN(mpp_enc_data->ver_stride, 64) *
                                  3 / 2;
    } break;

    case MPP_FMT_YUV422_YUYV :
    case MPP_FMT_YUV422_YVYU :
    case MPP_FMT_YUV422_UYVY :
    case MPP_FMT_YUV422_VYUY :
    case MPP_FMT_YUV422P :
    case MPP_FMT_YUV422SP : {
        mpp_enc_data->frame_size = MPP_ALIGN(mpp_enc_data->hor_stride, 64) *
                                  MPP_ALIGN(mpp_enc_data->ver_stride, 64) *
                                  2;
    } break;

    case MPP_FMT_YUV400:
    case MPP_FMT_RGB444 :
    case MPP_FMT_BGR444 :
    case MPP_FMT_RGB555 :
    case MPP_FMT_BGR555 :
    case MPP_FMT_RGB565 :
    case MPP_FMT_BGR565 :
    case MPP_FMT_RGB888 :
    case MPP_FMT_BGR888 :
    case MPP_FMT_RGB101010 :
    case MPP_FMT_BGR101010 :
    case MPP_FMT_ARGB8888 :
    case MPP_FMT_ABGR8888 :
    case MPP_FMT_BGRA8888 :
    case MPP_FMT_RGBA8888 : {
        mpp_enc_data->frame_size = MPP_ALIGN(mpp_enc_data->hor_stride, 64) *
                                  MPP_ALIGN(mpp_enc_data->ver_stride, 64);
    } break;

    default: {
        mpp_enc_data->frame_size = MPP_ALIGN(mpp_enc_data->hor_stride, 64) *
                                  MPP_ALIGN(mpp_enc_data->ver_stride, 64) *
                                  4;
    } break;
    }

    // =================================初始化=================================
    // 初始化缓冲区组
    ret = mpp_buffer_group_get_internal(&mpp_enc_data->buf_grp, MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE);
    if (ret) {
        printf("failed to get mpp buffer group ret %d\n", ret);
        goto MPP_INIT_OUT;
    }
    else
    {
        printf("get mpp buffer group\n");
    }

    // 分配输入帧缓冲区
    ret = mpp_buffer_get(mpp_enc_data->buf_grp, &mpp_enc_data->frm_buf, mpp_enc_data->frame_size);
    if (ret) {
        printf("failed to get buffer for input frame ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 分配输出包缓冲区
    ret = mpp_buffer_get(mpp_enc_data->buf_grp, &mpp_enc_data->pkt_buf, mpp_enc_data->frame_size);
    if (ret) {
        printf("failed to get buffer for output packet ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // =================================编码=================================
    // 创建MPP上下文
    ret = mpp_create(&mpp_enc_data->ctx, &mpp_enc_data->mpi);
    if (ret) {
        printf("mpp_create failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 设置输出超时
    ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout);
    if (ret) {
        printf("mpi control set output timeout failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 初始化编码器
    ret = mpp_init(mpp_enc_data->ctx, MPP_CTX_ENC, mpp_enc_data->type);
    if (ret) {
        printf("mpp_init failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 初始化编码器配置
    ret = mpp_enc_cfg_init(&mpp_enc_data->cfg);
    if (ret) {
        printf("mpp_enc_cfg_init failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // [MIN-FIX] 获取默认配置：原来你这里误写成再次 get_internal(buf_grp)（会覆盖/泄漏且不对）
    ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_ENC_GET_CFG, mpp_enc_data->cfg);
    if (ret) {
        printf("get enc cfg (MPP_ENC_GET_CFG) failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 设置编码器基本参数，这些参数基本都是外部配置的mpp参数以及计算出来的值
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:width", mpp_enc_data->width);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:height", mpp_enc_data->height);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:hor_stride", mpp_enc_data->hor_stride);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:ver_stride", mpp_enc_data->ver_stride);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:format", mpp_enc_data->fmt);

    // 设置码率控制参数
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:mode", mpp_enc_data->rc_mode);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:bps_target", mpp_enc_data->bps);
    mpp_enc_cfg_set_u32(mpp_enc_data->cfg, "rc:max_reenc_times", 0);
    mpp_enc_cfg_set_u32(mpp_enc_data->cfg, "rc:super_mode", 0);

    // [MIN-FIX] 明确设置 min/max（且已 clamp）
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:bps_max", mpp_enc_data->bps_max);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:bps_min", mpp_enc_data->bps_min);

    // 设置帧率参数
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_in_flex", mpp_enc_data->fps_in_flex);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_in_num", mpp_enc_data->fps_in_num);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_in_denom", mpp_enc_data->fps_in_den);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_out_flex", mpp_enc_data->fps_out_flex);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_out_num", mpp_enc_data->fps_out_num);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_out_denom", mpp_enc_data->fps_out_den);

    // 设置GOP参数
    if (mpp_enc_data->gop_len <= 0) mpp_enc_data->gop_len = mpp_enc_data->fps_out_num * 2;
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:gop", mpp_enc_data->gop_len);

    // 设置编码器类型和H.264参数
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "codec:type", mpp_enc_data->type);
    RK_U32 constraint_set;

    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:profile", 100);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:level", 31);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:cabac_en", 1);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:cabac_idc", 0);
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:trans8x8", 1);

    // 应用配置：MPP_ENC_SET_CFG
    ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_ENC_SET_CFG, mpp_enc_data->cfg);
    if (ret) {
        printf("mpi control enc set cfg failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 设置SEI模式
    mpp_enc_data->sei_mode = MPP_ENC_SEI_MODE_ONE_FRAME;
    ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_ENC_SET_SEI_CFG, &mpp_enc_data->sei_mode);
    if (ret) {
        printf("mpi control enc set sei cfg failed ret %d\n", ret);
        goto MPP_INIT_OUT;
    }

    // 打印编码器配置信息
    printf("\n========== MPP编码器配置参数 ==========\n");
    printf("基础参数:\n");
    printf("  分辨率: %dx%d\n", mpp_enc_data->width, mpp_enc_data->height);
    printf("  像素格式: %s\n", mpp_enc_data->fmt == MPP_FMT_YUV420SP ? "YUV420SP" : 
                              mpp_enc_data->fmt == MPP_FMT_YUV420P ? "YUV420P" :
                              mpp_enc_data->fmt == MPP_FMT_BGR888 ? "BGR888" : "Unknown");
    printf("  编码类型: %s\n", mpp_enc_data->type == MPP_VIDEO_CodingAVC ? "H.264" : 
                              mpp_enc_data->type == MPP_VIDEO_CodingHEVC ? "H.265" : "Unknown");

    printf("\n帧率配置:\n");
    printf("  输入帧率: %d/%d (%s模式)\n", 
           mpp_enc_data->fps_in_num, 
           mpp_enc_data->fps_in_den,
           mpp_enc_data->fps_in_flex ? "灵活" : "固定");
    printf("  输出帧率: %d/%d (%s模式)\n", 
           mpp_enc_data->fps_out_num, 
           mpp_enc_data->fps_out_den,
           mpp_enc_data->fps_out_flex ? "灵活" : "固定");

    printf("\n码率控制:\n");
    printf("  控制模式: %s\n", mpp_enc_data->rc_mode == MPP_ENC_RC_MODE_CBR ? "CBR(固定码率)" : 
                              mpp_enc_data->rc_mode == MPP_ENC_RC_MODE_VBR ? "VBR(可变码率)" : "Unknown");
    printf("  目标码率: %.2f Mbps\n", mpp_enc_data->bps / (1024.0 * 1024.0));
    printf("  最大码率: %.2f Mbps\n", mpp_enc_data->bps_max / (1024.0 * 1024.0));
    printf("  最小码率: %.2f Mbps\n", mpp_enc_data->bps_min / (1024.0 * 1024.0));

    printf("\n编码参数:\n");
    printf("  GOP长度: %d\n", mpp_enc_data->gop_len);
    if (mpp_enc_data->type == MPP_VIDEO_CodingAVC) {
        printf("  H.264 Profile: %s\n", 
               mpp_enc_data->cfg ? "High" : "Unknown");
        printf("  H.264 Level: 31 (720p@30fps)\n");
    }

    printf("\n缓冲区配置:\n");
    printf("  帧缓冲区大小: %zu bytes\n", mpp_enc_data->frame_size);
    printf("  头信息大小: %zu bytes\n", mpp_enc_data->header_size);

    printf("=======================================\n\n");

    return 0;

MPP_INIT_OUT:
    if (mpp_enc_data->ctx) {
        mpp_destroy(mpp_enc_data->ctx);
        mpp_enc_data->ctx = NULL;
    }

    if (mpp_enc_data->frm_buf) {
        mpp_buffer_put(mpp_enc_data->frm_buf);
        mpp_enc_data->frm_buf = NULL;
    }

    if (mpp_enc_data->pkt_buf) {
        mpp_buffer_put(mpp_enc_data->pkt_buf);
        mpp_enc_data->pkt_buf = NULL;
    }

    if (mpp_enc_data->cfg) {
        mpp_enc_cfg_deinit(mpp_enc_data->cfg);
        mpp_enc_data->cfg = NULL;
    }

    if (mpp_enc_data->buf_grp) {
        mpp_buffer_group_put(mpp_enc_data->buf_grp);
        mpp_enc_data->buf_grp = NULL;
    }

    printf("init mpp failed!\n");
    return ret;
}

/**
 * @brief 获取编码器头信息get_header
 * 该函数用于获取H.264/HEVC编码器的头信息（SPS/PPS等）
 * @param mpp_enc_data MPP上下文指针
 * @param sps_header 用于存储头信息的结构体指针
 * @return _Bool 成功返回true(1)，失败返回false(0)
 */
static _Bool get_header(MppContext *mpp_enc_data, SpsHeader *sps_header)
{
    MPP_RET ret = MPP_OK;
    MppPacket packet = NULL;

    printf("开始获取编码器header信息...\n");

    // [MIN-FIX] 必要判空：init 失败或 pkt_buf 为空时直接失败返回
    if (!mpp_enc_data || !mpp_enc_data->ctx || !mpp_enc_data->mpi || !mpp_enc_data->pkt_buf) {
        printf("get_header failed: encoder not initialized or pkt_buf is NULL\n");
        return 0;
    }

    if (mpp_enc_data->type == MPP_VIDEO_CodingAVC || mpp_enc_data->type == MPP_VIDEO_CodingHEVC) {
        // 初始化数据包
        ret = mpp_packet_init_with_buffer(&packet, mpp_enc_data->pkt_buf);
        if (ret || !packet) {
            printf("mpp_packet_init_with_buffer failed ret=%d\n", ret);
            return 0;
        }
        mpp_packet_set_length(packet, 0);

        // 获取编码器头信息
        ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_ENC_GET_HDR_SYNC, packet);
        if (ret) {
            printf("mpi control enc get extra info failed ret=%d\n", ret);
            mpp_packet_deinit(&packet);
            return 0;  // [MIN-FIX] 失败返回 0
        }

        // 保存头信息
        if (packet) {
            void *ptr = mpp_packet_get_pos(packet);
            size_t len = mpp_packet_get_length(packet);

            if (sps_header) {
                sps_header->data = (uint8_t*)malloc(len);
                if (!sps_header->data) {
                    printf("failed to allocate memory for sps header\n");
                    mpp_packet_deinit(&packet);
                    return 0;
                }
                sps_header->size = (uint32_t)len;
                memcpy(sps_header->data, ptr, len);
            }
        }

        mpp_packet_deinit(&packet);
        printf("开始获取编码器header信息成功\n");
    }

    return 1;  // [MIN-FIX] 成功返回 1
}

/**process_image
 * @brief 处理图像编码
 * 该函数完成单帧图像的编码处理流程，包括：
 * 1. 将输入图像数据复制到编码缓冲区
 * 2. 设置编码帧参数
 * 3. 执行编码
 * 4. 获取编码后的数据包
 * 5. 处理编码后的数据（保存或发送）
 * 
 * @param p 输入图像数据指针
 * @param size 输入图像数据大小
 * @param mpp_enc_data MPP上下文指针
 * @return _Bool 成功返回true，失败返回false
 * 
 * 补充说明：
 * - p：必须指向一帧完整的原始图像数据（与 fmt 对应）
 * - size：通常应等于 mpp_enc_data->frame_size；如果更大，memcpy 会越界；如果更小，编码数据不完整
 * - write_frame：回调函数指针，若不为空则把编码后的码流输出给上层（如网络发送/写文件）
 * - 返回值语义（按当前代码习惯）：
 *   1：继续编码（尚未达到停止条件）
 *   0：停止编码（达到 frame_num 或 EOS）
 */
static _Bool process_image(uint8_t *p, int size, MppContext *mpp_enc_data)
{   
    MPP_RET ret = MPP_OK;
    MppFrame frame = NULL;
    MppPacket packet = NULL;
    MppMeta meta = NULL;

    // [MIN-FIX] 判空，避免 init 失败后仍送帧导致段错
    if (!mpp_enc_data || !mpp_enc_data->ctx || !mpp_enc_data->mpi || !mpp_enc_data->frm_buf || !mpp_enc_data->pkt_buf) {
        printf("process_image failed: encoder not initialized\n");
        return 0;
    }

    void *buf = mpp_buffer_get_ptr(mpp_enc_data->frm_buf);
    RK_U32 eoi = 1;
    static int save_count = 0;

    // [MIN-FIX] 防御：避免 memcpy 越界
    if (size <= 0 || (size_t)size > mpp_enc_data->frame_size) {
        printf("process_image invalid size=%d frame_size=%zu\n", size, mpp_enc_data->frame_size);
        return 1;
    }

    memcpy(buf, p, size);

    ret = mpp_frame_init(&frame);
    if (ret) {
        printf("mpp_frame_init failed\n");
        return 1;
    }

    mpp_frame_set_width(frame, mpp_enc_data->width);
    mpp_frame_set_height(frame, mpp_enc_data->height);
    mpp_frame_set_hor_stride(frame, mpp_enc_data->hor_stride);
    mpp_frame_set_ver_stride(frame, mpp_enc_data->ver_stride);
    mpp_frame_set_fmt(frame, mpp_enc_data->fmt);
    mpp_frame_set_buffer(frame, mpp_enc_data->frm_buf);
    mpp_frame_set_eos(frame, mpp_enc_data->frm_eos);

    meta = mpp_frame_get_meta(frame);

    ret = mpp_packet_init_with_buffer(&packet, mpp_enc_data->pkt_buf);
    if (ret || !packet) {
        printf("mpp_packet_init_with_buffer failed ret=%d\n", ret);
        mpp_frame_deinit(&frame);
        return 1;
    }
    mpp_packet_set_length(packet, 0);
    mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, packet);

    ret = mpp_enc_data->mpi->encode_put_frame(mpp_enc_data->ctx, frame);
    if (ret) {
        printf("mpp encode put frame failed\n");
        mpp_frame_deinit(&frame);
        mpp_packet_deinit(&packet);
        return 1;
    }

    mpp_frame_deinit(&frame);

    do {
        ret = mpp_enc_data->mpi->encode_get_packet(mpp_enc_data->ctx, &packet);
        if (ret) {
            printf("mpp encode get packet failed\n");
            return 1;
        }

        if (packet) {
            void *ptr = mpp_packet_get_pos(packet);
            size_t len = mpp_packet_get_length(packet);
            char log_buf[256];
            RK_S32 log_size = sizeof(log_buf) - 1;
            RK_S32 log_len = 0;

            mpp_enc_data->pkt_eos = mpp_packet_get_eos(packet);

            if (save_count < 5) {
                char filename[64];
                snprintf(filename, sizeof(filename), "encoded_frame_%d.h264", save_count);
                FILE *fp = fopen(filename, "wb");
                if (fp) {
                    fwrite(ptr, 1, len, fp);
                    fclose(fp);
                    printf("已保存编码后的帧到文件: %s, 大小: %zu bytes\n", filename, len);
                    save_count++;
                }
            }

            if (mpp_enc_data->write_frame)
                if (!(mpp_enc_data->write_frame)(ptr, (int)len))
                    printf("------------sendok!\n");

            log_len += snprintf(log_buf + log_len, log_size - log_len,
                              "encoded frame %-4d", mpp_enc_data->frame_count);

            if (mpp_packet_is_partition(packet)) {
                eoi = mpp_packet_is_eoi(packet);
                log_len += snprintf(log_buf + log_len, log_size - log_len,
                                  " pkt %d", mpp_enc_data->frm_pkt_cnt);
                mpp_enc_data->frm_pkt_cnt = (eoi) ? (0) : (mpp_enc_data->frm_pkt_cnt + 1);
            }

            log_len += snprintf(log_buf + log_len, log_size - log_len,
                              " size %-7zu", len);

            if (mpp_packet_has_meta(packet)) {
                meta = mpp_packet_get_meta(packet);
                RK_S32 temporal_id = 0;
                RK_S32 lt_idx = -1;
                RK_S32 avg_qp = -1;
                RK_S32 bps_rt = -1;

                if (MPP_OK == mpp_meta_get_s32(meta, KEY_TEMPORAL_ID, &temporal_id))
                    log_len += snprintf(log_buf + log_len, log_size - log_len,
                                      " tid %d", temporal_id);

                if (MPP_OK == mpp_meta_get_s32(meta, KEY_LONG_REF_IDX, &lt_idx))
                    log_len += snprintf(log_buf + log_len, log_size - log_len,
                                      " lt %d", lt_idx);

                if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_AVERAGE_QP, &avg_qp))
                    log_len += snprintf(log_buf + log_len, log_size - log_len,
                                      " qp %2d", avg_qp);

                if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_BPS_RT, &bps_rt))
                    log_len += snprintf(log_buf + log_len, log_size - log_len,
                                      " bps_rt %d", bps_rt);
            }

            printf("%s\n", log_buf);

            mpp_packet_deinit(&packet);

            mpp_enc_data->stream_size += len;
            mpp_enc_data->frame_count += eoi;

            if (mpp_enc_data->pkt_eos) {
                printf("found last packet\n");
            }
        }
    } while (!eoi);

    if (mpp_enc_data->frame_num > 0 && mpp_enc_data->frame_count >= mpp_enc_data->frame_num) {
        printf("encode max %d frames", mpp_enc_data->frame_count);
        return 0;
    }

    if (mpp_enc_data->frm_eos && mpp_enc_data->pkt_eos)
        return 0;

    return 1;
}
