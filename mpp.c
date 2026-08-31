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

// ================================ 日志降频配置 ================================
// MPP 编码属于每帧都会执行的高频路径。
// 原始版本每帧打印 sendok / encoded frame / 保存 h264 文件，会造成明显 CPU 与 IO 开销。
// 这里保留错误日志和初始化日志，只对编码状态做低频打印。
#define ENABLE_MPP_FRAME_LOG 1
#define MPP_FRAME_LOG_INTERVAL 60
#define ENABLE_MPP_SAVE_DEBUG 0


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
// - 返回值语义：目前函数总返回 1，表示“成功/已获取”，并未严格区分失败返回 0

static _Bool process_image(uint8_t *p, int size, MppContext *mpp_enc_data);  // 声明单帧编码处理函数
// - p 类型：uint8_t*（输入参数）：指向原始图像帧数据的起始地址），需与 mpp_enc_data->fmt 匹配
// - size 类型：int（输入参数）：该帧数据字节数，通常应与 mpp_enc_data->frame_size 一致，否则 memcpy 可能越界或数据不完整
// - mpp_enc_data 类型：MppContext*（输入参数）：已初始化编码器上下文，内部包含 frame buffer / packet buffer / callback 等
// - 返回值语义（结合你当前实现）：1=继续编码，0=达到停止条件（如 EOS 或最大帧数）



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
    MPP_RET ret = MPP_OK;  // 保存MPP接口返回状态
    // 重置MPP上下文
    ret = ctx->mpi->reset(ctx->ctx); 
    // 解释：调用 MPP 的 reset 接口，重置编码器内部状态。
    // ctx->mpi：MPP 提供的 API 函数表
    // ctx->ctx：MPP 编码器上下文句柄（由 mpp_create/mpp_init 创建）
    // reset 的作用通常包括：清空内部队列、释放内部缓存、恢复初始状态，避免残留帧/包影响后续。
    // 返回值 ret：MPP_OK(通常为0) 表示成功；非0表示失败（可能是 ctx 为空、状态异常等）。
    if (ret)
    {
            printf("mpi->reset failed\n");  // 重置失败时打印错误
    }
    // 解释：下面这一段开始释放“编码器实例本体”，确保 MPP 内部资源彻底回收。
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
    // 设置基本编码参数
    mpp_enc_data->hor_stride   = MPP_ALIGN(mpp_enc_data->width, 16);  // 宽度 stride 按 16 对齐（硬件常见要求）
    mpp_enc_data->ver_stride   = MPP_ALIGN(mpp_enc_data->height, 16); // 高度 stride 按 16 对齐

    // 根据编码的输入格式计算视频帧的大小
    switch (mpp_enc_data->fmt & MPP_FRAME_FMT_MASK) {  // 取 fmt 的格式部分（屏蔽掉非格式标志位）
        /*1) mpp_enc_data->fmt
      - 类型：MppFrameFormat（MPP 的像素格式枚举）
        2) MPP_FRAME_FMT_MASK
        作用：只保留 fmt 中“表示像素格式本身”的那几位
        */
    case MPP_FMT_YUV420SP:  // YUV420 半平面（NV12/NV21 类）
    case MPP_FMT_YUV420P: { // YUV420 平面（I420 类）
        mpp_enc_data->frame_size = MPP_ALIGN(mpp_enc_data->hor_stride, 64) *  // stride 再按 64 对齐（利于 DMA/块处理）
                                  MPP_ALIGN(mpp_enc_data->ver_stride, 64) *  // 同上
                                  3 / 2;                                     // YUV420 每像素 1.5 字节
    } break;  // 结束该 case

    case MPP_FMT_YUV422_YUYV :  // YUV422 打包格式：YUYV
    case MPP_FMT_YUV422_YVYU :  // YUV422 打包格式：YVYU
    case MPP_FMT_YUV422_UYVY :  // YUV422 打包格式：UYVY
    case MPP_FMT_YUV422_VYUY :  // YUV422 打包格式：VYUY
    case MPP_FMT_YUV422P :      // YUV422 平面
    case MPP_FMT_YUV422SP : {   // YUV422 半平面
        mpp_enc_data->frame_size = MPP_ALIGN(mpp_enc_data->hor_stride, 64) *  // 64 对齐后的行跨度
                                  MPP_ALIGN(mpp_enc_data->ver_stride, 64) *  // 64 对齐后的列跨度
                                  2;                                         // YUV422 每像素 2 字节
    } break;  // 结束该 case

    case MPP_FMT_YUV400:        // 仅亮度分量（灰度）
    case MPP_FMT_RGB444 :       // RGB 444
    case MPP_FMT_BGR444 :       // BGR 444
    case MPP_FMT_RGB555 :       // RGB 555
    case MPP_FMT_BGR555 :       // BGR 555
    case MPP_FMT_RGB565 :       // RGB 565
    case MPP_FMT_BGR565 :       // BGR 565
    case MPP_FMT_RGB888 :       // RGB 888
    case MPP_FMT_BGR888 :       // BGR 888
    case MPP_FMT_RGB101010 :    // RGB 10bit
    case MPP_FMT_BGR101010 :    // BGR 10bit
    case MPP_FMT_ARGB8888 :     // ARGB 8888
    case MPP_FMT_ABGR8888 :     // ABGR 8888
    case MPP_FMT_BGRA8888 :     // BGRA 8888
    case MPP_FMT_RGBA8888 : {   // RGBA 8888
        mpp_enc_data->frame_size = MPP_ALIGN(mpp_enc_data->hor_stride, 64) *  // 行跨度对齐
                                  MPP_ALIGN(mpp_enc_data->ver_stride, 64);   // 列跨度对齐（这里按“1字节/像素”兜底估算，实际大小通常还需乘以 bytes-per-pixel）
    } break;  // 结束该 case

    default: {  // 未覆盖格式：使用更保守的估算方式
        mpp_enc_data->frame_size = MPP_ALIGN(mpp_enc_data->hor_stride, 64) *  // 行跨度对齐
                                  MPP_ALIGN(mpp_enc_data->ver_stride, 64) *  // 列跨度对齐
                                  4;                                         // 按 4 字节/像素兜底（如 RGBA）
    } break;  // 结束 default
    }  // switch 结束


    // =================================初始化=================================  // 初始化阶段分界：申请内存/创建编码器等
    // 初始化缓冲区组
    ret = mpp_buffer_group_get_internal(&mpp_enc_data->buf_grp, MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE); 
    /*
    * 作用：创建 / 获取一个“MPP 内部缓冲区组（buffer group）”，并把句柄保存到 mpp_enc_data->buf_grp 中。
    * 参数解释：
    * 1) &mpp_enc_data->buf_grp
    *    - 类型：MppBufferGroup* 的地址（输出参数）
    *    - 含义：函数成功后会把创建好的 buffer group 句柄写到该变量里
    *    - 作用：后续 mpp_buffer_get() 从这个组里申请 frm_buf/pkt_buf；mpp_buffer_put() 归还给这个组
    * 2) MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE
    *    - 这是一个“类型 + 标志位”的组合（按位或）
    *    (a) MPP_BUFFER_TYPE_DRM
    *        - 表示缓冲区底层使用 DRM（Direct Rendering Manager）相关的内存分配方式
    *        - 通常对应“硬件/图形/视频编解码友好”的内存（如 DMA-BUF），利于 MPP 硬件模块直接访问
    *        - 目标：让硬件编码器能高效、零拷贝/少拷贝地读写这些 buffer
    *    (b) MPP_BUFFER_FLAGS_CACHABLE
    *        - 表示申请到的 buffer 在 CPU 侧是“可缓存”的（cacheable）
    *        - 目标：提升 CPU memcpy / 读写帧数据的速度（尤其是你后面 memcpy(buf, p, size)）
    * 返回值 ret：
    * - ret == MPP_OK（通常为 0）：成功获得 buffer group
    * - ret != MPP_OK：失败
    *
    * 为什么要先创建 buffer group？
    * - buffer group 相当于“统一的内存池/管理器”，你后面申请输入帧 buffer（frm_buf）和输出码流 buffer（pkt_buf）
    *   都从同一个 group 里来，便于统一管理、释放和复用。
    */
    // 
    if (ret) {  // 若创建失败
        printf("failed to get mpp buffer group ret %d\n", ret);  // 打印错误码
        goto MPP_INIT_OUT;  // 跳转统一清理出口
    }
    else
    {
        printf("get mpp buffer group\n");  // 创建成功提示
    }

    // 分配输入帧缓冲区，这里由于并没有使用FPC，所以frame_size并不需要加上header_size  

    // 初始化并配置 MPP mpp_enc_data 结构体
    // - frm_buf：存放原始输入帧（YUV/RGB）数据的 buffer
    // - frame_size 应与后续 process_image 的 memcpy size 匹配，否则可能越界
    ret = mpp_buffer_get(mpp_enc_data->buf_grp, &mpp_enc_data->frm_buf, mpp_enc_data->frame_size); 
    // 从buf组申请输入帧buffer（存原始图像）
    if (ret) {  // 申请失败
        printf("failed to get buffer for input frame ret %d\n", ret);  // 打印错误码
        goto MPP_INIT_OUT;  // 跳转清理
    }

    // 分配输出包缓冲区

    // - pkt_buf：存放编码后码流数据的 buffer（MPP 会向其中写入 H264 NALU）
    // - 这里预留大小为 frame_size 是一种“上限兜底策略”，通常码流比原始帧小；但极端场景可能需要更大 buffer
    ret = mpp_buffer_get(mpp_enc_data->buf_grp, &mpp_enc_data->pkt_buf, mpp_enc_data->frame_size); // 申请输出packet buffer（存码流，按上限预留）
    if (ret) {  // 申请失败
        printf("failed to get buffer for output packet ret %d\n", ret);  // 打印错误码
        goto MPP_INIT_OUT;  // 跳转清理
    }


    // =================================编码=================================  
    // 创建MPP上下文

    // - mpp_create 会创建一个 MppCtx (mpp_enc_data->ctx) 和一个 MppApi (mpp_enc_data->mpi)
    // - mpi 是函数表，后续通过 mpi->control / mpi->encode_put_frame 等与 MPP 交互
    ret = mpp_create(&mpp_enc_data->ctx, &mpp_enc_data->mpi); // 创建MPP上下文ctx以及接口表mpi（后续通过mpi调用control/encode等）
    if (ret) {  // 创建失败
        printf("mpp_create failed ret %d\n", ret);  // 打印错误码
        goto MPP_INIT_OUT;  // 跳转清理
    }

    // 设置输出超时

    // - timeout=MPP_POLL_BLOCK 表示 encode_get_packet 会阻塞直到拿到 packet
    ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout); 
    // control：设置输出取包等待方式/超时（timeout为BLOCK表示阻塞）
    if (ret) {  // 设置失败
        printf("mpi control set output timeout failed ret %d\n", ret);  // 打印错误码
        goto MPP_INIT_OUT;  // 跳转清理
    }

    // 初始化编码器

    // - MPP_CTX_ENC 表示 Encoder 上下文
    // - mpp_enc_data->type 为编码类型：MPP_VIDEO_CodingAVC(H264) 或 HEVC(H265)
    ret = mpp_init(mpp_enc_data->ctx, MPP_CTX_ENC, mpp_enc_data->type); 
    // 初始化ctx为编码器(MPP_CTX_ENC)，并指定编码类型type(H264/H265等)
    if (ret) {  // 初始化失败
        printf("mpp_init failed ret %d\n", ret);  // 打印错误码
        goto MPP_INIT_OUT;  // 跳转清理
    }

    // 初始化编码器配置
    // - cfg 是编码器配置对象，用“字符串键”设置参数
    // - 修改 cfg 后需要 MPP_ENC_SET_CFG 写回生效
    ret = mpp_enc_cfg_init(&mpp_enc_data->cfg); // 初始化编码配置对象cfg（后续用mpp_enc_cfg_set_*写参数）
    if (ret) {  // 初始化失败
        printf("mpp_enc_cfg_init failed ret %d\n", ret);  // 打印错误码
        goto MPP_INIT_OUT;  // 跳转清理
    }

    // 获取默认配置
    ret = mpp_buffer_group_get_internal(&mpp_enc_data->buf_grp,
                                    MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE);
            /*
            * 作用：
            *   创建一个 MPP 内部缓冲区组（buffer group），并将句柄保存到 mpp_enc_data->buf_grp。
            *   该 buffer group 相当于一个统一的“内存池管理器”，
            *   后续所有编码输入/输出 buffer 都从这个组中申请和归还。
            * 1) &mpp_enc_data->buf_grp
            *    - 类型：MppBufferGroup* 的地址（输出参数）
            *    - 含义：函数成功后会把 buffer group 句柄写入该变量
            *    - 作用：
            *        • 后续 mpp_buffer_get() 从该 group 申请内存
            *        • mpp_buffer_put() 归还 buffer 给该 group
            *        • 统一管理编码所需的内存生命周期
            * 2) MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE
            *    - 类型：缓冲区属性标志组合（按位或）
            *    - 含义：指定 buffer 的底层分配方式和访问属性
            *    (a) MPP_BUFFER_TYPE_DRM
            *        - 使用 DRM（Direct Rendering Manager）内存类型
            *        - 对应 DMA/硬件加速友好的物理内存
            *        - 适合视频编码硬件直接访问
            *        - 目标：减少 CPU ↔ 硬件 之间的数据拷贝
            *    (b) MPP_BUFFER_FLAGS_CACHABLE
            *        - 表示该 buffer 在 CPU 侧是“可缓存”的
            *        - 提升 memcpy / CPU 读写性能
            *        - 适合 process_image() 中大量 CPU 拷贝场景
            * 返回值 ret：
            *   ret == MPP_OK（通常为 0）→ buffer group 创建成功
            *   ret != MPP_OK(非0)→ 创建失败
            */
    if (ret) {  // 获取失败
        printf("get enc cfg failed ret %d\n", ret);  // 打印错误码
        goto MPP_INIT_OUT;  // 跳转清理
    }

    // 设置编码器基本参数，这些参数基本都是外部配置的mpp参数以及计算出来的值
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:width", mpp_enc_data->width); // 输入预处理参数：输入宽度
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:height", mpp_enc_data->height); // 输入预处理参数：输入高度
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:hor_stride", mpp_enc_data->hor_stride); // 输入预处理参数：水平stride（对齐后的行跨度）
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:ver_stride", mpp_enc_data->ver_stride); // 输入预处理参数：垂直stride（对齐后的列跨度）
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "prep:format", mpp_enc_data->fmt); // 输入预处理参数：像素格式（YUV420SP等）
    // 设置码率控制参数
    // - rc:mode 选择码控方式：CBR 固定码率 / VBR 可变码率 / FIXQP 固定QP
    // - bps_target/bps_min/bps_max 用于限制码率区间
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:mode", mpp_enc_data->rc_mode); // 码控模式：CBR/VBR/FIXQP等
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:bps_target", mpp_enc_data->bps); // 目标码率：bps_target（单位bit/s）
    mpp_enc_cfg_set_u32(mpp_enc_data->cfg, "rc:max_reenc_times", 0); // 最大重编码次数：0表示禁用重编码（降低延迟/复杂度）
    mpp_enc_cfg_set_u32(mpp_enc_data->cfg, "rc:super_mode", 0); // super_mode：高级码控/超分模式开关（0关闭）
    // 设置CBR模式下的码率范围：mpp_enc_data->bps
    mpp_enc_data->bps_max = mpp_enc_data->bps_max ? mpp_enc_data->bps_max : mpp_enc_data->bps * 17 / 16; // 若未手动指定max，则默认≈target×1.06
    mpp_enc_data->bps_min = mpp_enc_data->bps_min ? mpp_enc_data->bps_min : mpp_enc_data->bps * 15 / 16; // 若未手动指定min，则默认≈target×0.94
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:bps_max", mpp_enc_data->bps_max); // 写入最大码率约束
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:bps_min", mpp_enc_data->bps_min); // 写入最小码率约束
    // 设置帧率参数： mpp_enc_data->fps
    // - fps_in/out 使用“分子/分母”表达帧率，例如 30/1 = 30fps
    // - flex=0 表示固定帧率；flex=1 表示允许动态帧率（可变）
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_in_flex", mpp_enc_data->fps_in_flex); // 输入帧率是否灵活（1可变/0固定）
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_in_num", mpp_enc_data->fps_in_num); // 输入帧率分子
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_in_denom", mpp_enc_data->fps_in_den); // 输入帧率分母（注意你变量名fps_in_den对应denom）
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_out_flex", mpp_enc_data->fps_out_flex); // 输出帧率是否灵活
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_out_num", mpp_enc_data->fps_out_num); // 输出帧率分子
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:fps_out_denom", mpp_enc_data->fps_out_den); // 输出帧率分母
    // 设置GOP参数：mpp_enc_data->gop_len
    // - GOP=I 帧间隔，例如 gop=60 表示每 60 帧插入一个 IDR/I 帧
    // - GOP 变小：低延迟但码率更高；GOP 变大：压缩更好但随机访问更差
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "rc:gop", mpp_enc_data->gop_len); // GOP长度：I帧间隔（常用于控制延迟/压缩效率）
    // 设置编码器类型和H.264参数
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "codec:type", mpp_enc_data->type); // codec类型：H.264/H.265等（与mpp_init的type一致）
    RK_U32 constraint_set; // 约束集变量声明（本段未使用，可能预留给profile/constraint相关设置）
    /*
        * H.264 profile_idc parameter
        * 66  - Baseline profile
        * 77  - Main profile
        * 100 - High profile
        */
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:profile", 100); // High profile  // H.264 Profile=100(High)，
    /*
        * H.264 level_idc parameter
        * 10 / 11 / 12 / 13    - qcif@15fps / cif@7.5fps / cif@15fps / cif@30fps
        * 20 / 21 / 22         - cif@30fps / half-D1@@25fps / D1@12.5fps
        * 30 / 31 / 32         - D1@25fps / 720p@30fps / 720p@60fps
        * 40 / 41 / 42         - 1080p@30fps / 1080p@30fps / 1080p@60fps
        * 50 / 51 / 52         - 4K@30fps
        */
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:level", 31); // H.264 level=31（对应 720p@30fps 档位）
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:cabac_en", 1); // CABAC开关：1启用（提升压缩率，计算更重）
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:cabac_idc", 0); // CABAC初始化表idc：通常0为默认
    mpp_enc_cfg_set_s32(mpp_enc_data->cfg, "h264:trans8x8", 1); // 8x8变换开关：1启用（High profile常用）

    
    // 应用配置：MPP_ENC_SET_CFG
    ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_ENC_SET_CFG, mpp_enc_data->cfg); // control：将修改后的cfg写回编码器使其生效
    if (ret) { // 写回失败
        printf("mpi control enc set cfg failed ret %d\n", ret); // 打印错误码
        goto MPP_INIT_OUT; // 跳转清理
    }

    // 设置SEI模式：
    // - SEI（Supplemental Enhancement Information）可携带额外信息（时间戳、用户数据等）
    // - MPP_ENC_SEI_MODE_ONE_FRAME 表示每帧插入一次 SEI（会增加码流开销）
    mpp_enc_data->sei_mode = MPP_ENC_SEI_MODE_ONE_FRAME; // 设置SEI输出模式：每帧插入一次SEI（按MPP定义）
    ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_ENC_SET_SEI_CFG, &mpp_enc_data->sei_mode); // control：写入SEI配置MPP_ENC_SET_SEI_CFG, &mpp_enc_data->sei_mode
    if (ret) { // 设置失败
        printf("mpi control enc set sei cfg failed ret %d\n", ret); // 打印错误码
        goto MPP_INIT_OUT; // 跳转清理
    }


    // 打印编码器配置信息,这一段主要用于调试验证参数是否设置正确
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
    // “错误清理出口”：当 init_mpp() 在中途任意一步失败时，通过 goto MPP_INIT_OUT 跳转到这里，释放已经申请的资源，防止内存泄漏或句柄残留。
    if (mpp_enc_data->ctx) {
        mpp_destroy(mpp_enc_data->ctx);         // 释放 MPP 内部编码器实例
        mpp_enc_data->ctx = NULL;
    }

    if (mpp_enc_data->frm_buf) {
        mpp_buffer_put(mpp_enc_data->frm_buf);  // 把 buffer 归还给 buffer group
        mpp_enc_data->frm_buf = NULL;
    }

    if (mpp_enc_data->pkt_buf) {
        mpp_buffer_put(mpp_enc_data->pkt_buf);  // 释放输出 buffer
        mpp_enc_data->pkt_buf = NULL;
    }

    if (mpp_enc_data->cfg) {
        mpp_enc_cfg_deinit(mpp_enc_data->cfg);  // 销毁配置对象，释放内部内存
        mpp_enc_data->cfg = NULL;
    }

    printf("init mpp failed!\n");
    return ret;
}

/**
 * @brief 获取编码器头信息get_header
 * 该函数用于获取H.264/HEVC编码器的头信息（SPS/PPS等）
 * @param mpp_enc_data MPP上下文指针
 * @param sps_header 用于存储头信息的结构体指针
 * @return _Bool 成功返回true，失败返回false
 */
static _Bool get_header(MppContext *mpp_enc_data, SpsHeader *sps_header)
{
    MPP_RET ret = MPP_OK;
    MppPacket packet = NULL;

    printf("开始获取编码器header信息...\n");
    if (mpp_enc_data->type == MPP_VIDEO_CodingAVC || mpp_enc_data->type == MPP_VIDEO_CodingHEVC) {
        // 初始化数据包
        mpp_packet_init_with_buffer(&packet, mpp_enc_data->pkt_buf);
        mpp_packet_set_length(packet, 0);

        // 获取编码器头信息
        ret = mpp_enc_data->mpi->control(mpp_enc_data->ctx, MPP_ENC_GET_HDR_SYNC, packet);
        if (ret) {
            printf("mpi control enc get extra info failed\n");
            return 1;
        }

        // 保存头信息
        if (packet) {
            void *ptr = mpp_packet_get_pos(packet);      // header 数据起始地址
            size_t len = mpp_packet_get_length(packet);  // header 数据长度（字节）
            
            if (sps_header) {
                sps_header->data = (uint8_t*)malloc(len); // 为 header 数据分配独立内存（避免 packet 生命周期结束后数据丢失）
                if (!sps_header->data) {                  //获取信息失败打印提示
                    printf("failed to allocate memory for sps header\n");
                    mpp_packet_deinit(&packet);
                    return 1;
                }
                sps_header->size = len;                   // 记录 header 长度
                memcpy(sps_header->data, ptr, len);       // 拷贝 header 数据到用户结构体
            }
        }

        mpp_packet_deinit(&packet);
        printf("开始获取编码器header信息成功\n");
    }

    return 1;
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
    RK_U32 eoi = 1;
    void *buf = NULL;

    if (!p || !mpp_enc_data || !mpp_enc_data->frm_buf) {
        printf("mpp process_image invalid input\n");
        return 0;
    }

    buf = mpp_buffer_get_ptr(mpp_enc_data->frm_buf);
    if (!buf) {
        printf("mpp_buffer_get_ptr failed\n");
        return 0;
    }

    // 防止外部传入异常 size 导致 memcpy 越界。
    // 正常情况下 size 应小于等于 frame_size；如果偏大，则只拷贝 frame_size。
    if (size > (int)mpp_enc_data->frame_size) {
        printf("mpp input frame size too large, input=%d, frame_size=%zu\n",
               size, mpp_enc_data->frame_size);
        size = (int)mpp_enc_data->frame_size;
    }

    memcpy(buf, p, size);

    ret = mpp_frame_init(&frame);
    if (ret) {
        printf("mpp_frame_init failed\n");
        return 0;
    }

    mpp_frame_set_width(frame, mpp_enc_data->width);
    mpp_frame_set_height(frame, mpp_enc_data->height);
    mpp_frame_set_hor_stride(frame, mpp_enc_data->hor_stride);
    mpp_frame_set_ver_stride(frame, mpp_enc_data->ver_stride);
    mpp_frame_set_fmt(frame, mpp_enc_data->fmt);
    mpp_frame_set_buffer(frame, mpp_enc_data->frm_buf);
    mpp_frame_set_eos(frame, mpp_enc_data->frm_eos);

    meta = mpp_frame_get_meta(frame);
    mpp_packet_init_with_buffer(&packet, mpp_enc_data->pkt_buf);
    mpp_packet_set_length(packet, 0);
    mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, packet);

    ret = mpp_enc_data->mpi->encode_put_frame(mpp_enc_data->ctx, frame);
    if (ret) {
        printf("mpp encode put frame failed\n");
        if (packet)
            mpp_packet_deinit(&packet);
        mpp_frame_deinit(&frame);
        return 0;
    }

    mpp_frame_deinit(&frame);

    do {
        ret = mpp_enc_data->mpi->encode_get_packet(mpp_enc_data->ctx, &packet);
        if (ret) {
            printf("mpp encode get packet failed\n");
            return 0;
        }

        if (packet) {
            void *ptr = mpp_packet_get_pos(packet);
            size_t len = mpp_packet_get_length(packet);
            char log_buf[256];
            RK_S32 log_size = sizeof(log_buf) - 1;
            RK_S32 log_len = 0;

            mpp_enc_data->pkt_eos = mpp_packet_get_eos(packet);

#if ENABLE_MPP_SAVE_DEBUG
            static int save_count = 0;
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
#endif

            if (mpp_enc_data->write_frame) {
                int write_ret = (mpp_enc_data->write_frame)(ptr, len);
#if ENABLE_MPP_FRAME_LOG
                if (!write_ret && (mpp_enc_data->frame_count % MPP_FRAME_LOG_INTERVAL == 0)) {
                    printf("[mpp] send ok, frame=%d, packet_size=%zu\n",
                           mpp_enc_data->frame_count, len);
                }
#endif
            }

            log_len += snprintf(log_buf + log_len, log_size - log_len,
                                "encoded frame %-4d", mpp_enc_data->frame_count);

            if (mpp_packet_is_partition(packet)) {
                eoi = mpp_packet_is_eoi(packet);
                log_len += snprintf(log_buf + log_len, log_size - log_len,
                                    " pkt %d", mpp_enc_data->frm_pkt_cnt);
                mpp_enc_data->frm_pkt_cnt = (eoi) ? 0 : (mpp_enc_data->frm_pkt_cnt + 1);
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

#if ENABLE_MPP_FRAME_LOG
            if (mpp_enc_data->frame_count % MPP_FRAME_LOG_INTERVAL == 0) {
                printf("%s\n", log_buf);
            }
#endif

            mpp_packet_deinit(&packet);

            mpp_enc_data->stream_size += len;
            mpp_enc_data->frame_count += eoi;

            if (mpp_enc_data->pkt_eos) {
                printf("found last packet\n");
            }
        }
    } while (!eoi);

    if (mpp_enc_data->frame_num > 0 && mpp_enc_data->frame_count >= mpp_enc_data->frame_num) {
        printf("encode max %d frames\n", mpp_enc_data->frame_count);
        return 0;
    }

    if (mpp_enc_data->frm_eos && mpp_enc_data->pkt_eos)
        return 0;

    return 1;
}
