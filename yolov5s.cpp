#include "yolov5s.h"              // 引入 Yolov5s 类的声明（头文件）
#include "post_process.h"         // 引入后处理接口（NMS/解码/结果结构体等）
#include <stdexcept>

// 静态函数，用于打印 rknn_tensor_attr 结构体的信息
// static：仅在当前 .cpp 文件可见，避免与其他文件同名函数冲突
static void print_tensor_attr(rknn_tensor_attr *attr)
{
    // 构建形状字符串，例如 "640,480,3" 表示一个 640x480 的 RGB 图像
    // n_dims：维度个数；dims[]：每一维的大小
    string shape_str = attr->n_dims < 1 ? "" : to_string(attr->dims[0]);
    for(int i = 1; i < attr->n_dims; i++)
    {
        // 将 dims[i] 逐个拼接到 shape_str 里，形成可读的形状信息
        string current_str = to_string(attr->dims[i]);
        shape_str += "," + current_str;
    }

    // // 打印张量的索引、名称、维度数、维度、大小和格式
    // printf("index = %d, name = %s， n_dims = %d, dims = [%s], \nsize = %d, fmt = %s\n", 
    //         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->size, get_format_string(attr->fmt));
    // printf("\n");
}

// ================================ 构造函数：加载模型 + 初始化 RKNN + 查询 tensor 信息 ================================
Yolov5s::Yolov5s(const char* model_path, int npu_index)
    : context(0), model_size(0), model_data(nullptr)
{
    int ret;                                              // RKNN API 返回值，0 表示成功（通常约定）
    model_data = load_model(model_path, this->model_size); // 从文件读取 .rknn 模型到内存，并得到大小
    if (model_data == nullptr || model_size == 0) {
        throw std::runtime_error(std::string("failed to load RKNN model: ") + model_path);
    }

    /* 模型初始化加载到RKNN中 */
    // rknn_init：把模型加载到 NPU 推理上下文中
    // RKNN_FLAG_PRIOR_HIGH：优先级较高（具体含义与平台/SDK有关）
    ret = rknn_init(&this->context, model_data, this->model_size , RKNN_FLAG_PRIOR_HIGH, NULL);
    if (ret != 0)
    {  
        free(model_data);
        model_data = nullptr;
        throw std::runtime_error("rknn_init failed, error code: " + std::to_string(ret));
    } 
    else 
    {
        printf("yolo %d初始化成功！\n",npu_index);          // 初始化成功，打印当前实例（线程）编号
    }

    /* 对不同线程分配NPU，加速计算 */
    // 目的：多线程时把不同 Yolov5s 实例绑定到不同 NPU 核心，减少争用，提高并行推理效率
    // RK3588 常见有 3 个 NPU core（0/1/2），这里用 npu_index % 4 做映射：
    // - 0 -> core0
    // - 1 -> core1
    // - 2/3 -> core2（注意：这样 core2 会更“忙”一些）
    if(npu_index %4 == 0)     {   ret = rknn_set_core_mask(this->context,RKNN_NPU_CORE_0); }
    else if(npu_index %4 == 1){   ret = rknn_set_core_mask(this->context,RKNN_NPU_CORE_1); }
    else                     {   ret = rknn_set_core_mask(this->context,RKNN_NPU_CORE_2); }

    if (ret != 0){ printf("npu set failed! error code: %d\n", ret);} // 绑定失败打印错误

    /* 够查询获取到模型输入输出信息、逐层运行时间、模型推理的总时间、
        SDK版本、内存占用信息、用户自定义字符串等信息 */
    // RKNN_QUERY_IN_OUT_NUM：查询模型输入/输出 tensor 数量
    ret = rknn_query(context, RKNN_QUERY_IN_OUT_NUM,&this->num_tensors ,sizeof(this->num_tensors) );
    if (ret != 0){ printf("rknn_query failed! error code: %d\n", ret); } 

    // printf("输入 tensor个数为：%d \n",num_tensors.n_output); // 这里应该是 n_input
    // printf("输出 tensor个数为：%d \n",num_tensors.n_input);  // 这里应该是 n_output

    /* 根据tensor信息调整输入和输出的tensor个数 */
    // vector resize：为每个 input/output 准备一个 rknn_tensor_attr
    input_attrs.resize(num_tensors.n_input);
    output_attrs.resize(num_tensors.n_output);
    
    /* 获取模型需要的输入和输出的tensor信息 */
    // 遍历每个输入 tensor，查询其属性（shape/format/type 等）
    for(int i = 0;i < num_tensors.n_input; i++)
    {
        input_attrs[i].index = i; // 指定要查询第 i 个输入 tensor
        ret = rknn_query(context, RKNN_QUERY_INPUT_ATTR,&(this->input_attrs[i]) ,sizeof( this->input_attrs[i]) );
        if (ret != 0)   { printf("rknn_query input_attrs failed! error code: %d\n", ret); } 

        printf("输入  的tensor%d  属性为：\n",i);
        print_tensor_attr(&(this->input_attrs[i])); // 打印属性（当前 printf 被注释，默认不输出）
    }

    // 遍历每个输出 tensor，查询其属性（zp/scale 等量化参数也在这里）
    for(int i = 0;i < num_tensors.n_output; i++)
    {
        output_attrs[i].index = i; // 指定要查询第 i 个输出 tensor
        ret = rknn_query(context, RKNN_QUERY_OUTPUT_ATTR,&(this->output_attrs[i]) ,sizeof( this->output_attrs[i]) );
        if (ret != 0)   { printf("rknn_query output_attrs failed! error code: %d\n", ret); } 

        // printf("输出  的tensor%d  属性为：\n",i);
        print_tensor_attr(&(this->output_attrs[i]));
    }

    /* 获取模型要求输入图像的参数信息 */
    // 根据输入张量的格式确定模型的维度信息
    // RKNN_TENSOR_NCHW：dims 通常是 [N, C, H, W]
    // RKNN_TENSOR_NHWC：dims 通常是 [N, H, W, C]
    if(input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        model_channel = input_attrs[0].dims[1]; // C
        model_height  = input_attrs[0].dims[2]; // H
        model_width   = input_attrs[0].dims[3]; // W
    }
    else if(input_attrs[0].fmt == RKNN_TENSOR_NHWC)
    {
        model_height  = input_attrs[0].dims[1]; // H
        model_width   = input_attrs[0].dims[2]; // W
        model_channel = input_attrs[0].dims[3]; // C
    }
    // 小提示：如果 fmt 不是 NCHW/NHWC（例如 UNKNOWN），这里可能需要一个 else 做容错
}

// ================================ 析构函数：释放 RKNN 资源 + 释放模型内存 ================================
// 析构函数中
Yolov5s::~Yolov5s()
{
    if (context) {
        rknn_destroy(context);    // 释放RKNN上下文：释放 NPU 侧资源/内部句柄等
    }
    free(this->model_data);       // 释放 load_model malloc 的模型内存
    // 小提示：如果 model_data 可能为 nullptr，free(nullptr) 也是安全的
}

// ================================ 读取模型文件到内存 ================================
unsigned char * Yolov5s::load_model(const char* model_path, unsigned int &model_size)
{
    model_size = 0;
    FILE *fp;                     // 文件指针
    unsigned char* model_data = nullptr;    // 模型数据缓冲区（malloc 分配）

    fp = fopen(model_path, "rb"); // 以二进制只读方式打开模型文件
    if(fp == NULL)
    {
        printf("open model failed: %s\n", model_path);
        return nullptr;
    }

    int ret = fseek(fp, 0, SEEK_END); // 移动到文件末尾，用于获取文件长度
    if(ret)
    {
        printf("fseek err : %d\n",ret);
        fclose(fp);
        return nullptr;
    }

    const long file_size = ftell(fp); // ftell 获取当前文件指针偏移量，即文件大小（字节）
    if (file_size <= 0) {
        printf("invalid model size: %ld\n", file_size);
        fclose(fp);
        return nullptr;
    }
    model_size = static_cast<unsigned int>(file_size);
    
    model_data =(unsigned char *) malloc(model_size); // 分配足够内存存放模型文件
    if (model_data == nullptr) {
        printf("allocate model buffer failed: %u bytes\n", model_size);
        fclose(fp);
        model_size = 0;
        return nullptr;
    }

    ret = fseek(fp, 0, SEEK_SET);     // 回到文件开头，准备读取
    if(ret)
    {
        printf("fseek err : %d\n",ret);
        free(model_data);
        fclose(fp);
        model_size = 0;
        return nullptr;
    }

    const size_t bytes_read = fread(model_data, 1, model_size, fp);
    fclose(fp);
    if(bytes_read != model_size)
    {
        printf("read model failed: %zu/%u bytes\n", bytes_read, model_size);
        free(model_data);
        model_size = 0;
        return nullptr;
    }

    return model_data;                 // 返回模型数据指针（由调用者 free）
}

// ================================ 推理：预处理(RGA) + RKNN 推理 + 后处理 + 绘制 ================================
int Yolov5s::inference_image(const Mat& orig_img, detect_result_group_t &result_group)
{
    int ret = 0; // 返回码：一般 0 表示成功

    float nms_threshold      = NMS_THRESHOLD; // NMS 阈值：用于去掉重叠框（来自 post_process.h 的宏/常量）
    float box_conf_threshold = BOX_THRESHOLD; // 置信度阈值：低于该阈值的框会被过滤

    Mat bkg;                                 // bkg：背景图（可能是 padding 后的图）

    // 记录输入图像的尺寸信息（orig_img 是原始帧）
    this->img_height  = orig_img.rows;       // 获取原始图像的高度
    this->img_width   = orig_img.cols;       // 获取原始图像的宽度
    this->img_channel = orig_img.channels(); // 获取原始图像的通道数（一般是 3）

    // 检查图像尺寸是否为16的倍数，如果不是则进行填充
    // 原因：很多硬件加速（RGA/DRM 等）对 stride/对齐有要求，16 对齐很常见
    if(img_width % 16 != 0 || img_height % 16 != 0)
    {
        int bkg_width  = (img_width  + 15) / 16 * 16; // 向上取整到 16 的倍数
        int bkg_height = (img_height + 15) / 16 * 16; // 向上取整到 16 的倍数

        bkg = Mat(bkg_height, bkg_width, CV_8UC3, cv::Scalar(0, 0, 0)); // 创建黑色背景图像（BGR=0）
        orig_img.copyTo(bkg(cv::Rect(0, 0, orig_img.cols, orig_img.rows))); // 将原图复制到左上角区域
        cv::imwrite("img_bkg.jpg", bkg); // 保存背景图像（调试用：会影响性能，发布建议关掉）
        this->img_width  = bkg_width;    // 更新图像宽度（后续以 padding 后尺寸为准）
        this->img_height = bkg_height;   // 更新图像高度
    }
    else
    {
        // 尺寸已经是 16 的倍数，不需要 padding
        bkg = orig_img.clone();          // 创建 bkg，并复制原始图像内容 (深拷贝)
        // 或者，如果你只是想避免 bkg 为空，也可以使用浅拷贝，但深拷贝更安全
        // bkg = orig_img;               // 浅拷贝，bkg 和 orig_img 共享数据，不推荐，可能引起意外修改
    }

    //error
    // this->img_height    = (orig_img.rows + 15) /16 * 16;
    // this->img_width     = (orig_img.cols + 15) /16 * 16;
    // this->img_channel   = orig_img.channels();

    // 模型输入尺寸（来自构造函数解析 input_attrs）
    int resize_height  = this->model_height;
    int resize_width   = this->model_width;
    int resize_channel = this->model_channel;

    // 打印图像的原始尺寸和调整后的尺寸（调试用）
    // printf("Image Height: %d\n", img_height);
    // printf("Image Width: %d\n", img_width);
    // printf("Image Channels: %d\n", img_channel);
    // printf("Resize Height: %d\n", resize_height);
    // printf("Resize Width: %d\n", resize_width);
    // printf("Resize Channels: %d\n", resize_channel);

    // 记录开始时间（用于统计耗时）
    auto start    = std::chrono::high_resolution_clock::now();
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // //opencv（备用方案）：用 OpenCV 做 BGR->RGB + resize
    // // 注释掉可能是为了改用 RGA 提升速度
    // Mat img_cvt;
    // Mat img_resize;
    // start       = std::chrono::high_resolution_clock::now();
    // cv::cvtColor(orig_img, img_cvt, cv::COLOR_BGR2RGB);
    // cv::resize(img_cvt, img_resize, Size(resize_height, resize_width), 0, 0, cv::INTER_LINEAR);
    // end         = std::chrono::high_resolution_clock::now();
    // duration    =std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // printf("opencv process time : %ld ms.\n", duration.count());
    // cv::imwrite("img_cv_linear.jpg", img_resize);    

    // ================================ 使用 RGA 进行图像预处理（颜色转换 + resize） ================================
    Mat img_rga;   // 用于封装 resize 后的结果（dst_buf）
    Mat img_cvt;   // 用于封装颜色转换后的结果（src_cvt_buf）
    start = std::chrono::high_resolution_clock::now();

    // 原始/目标缓冲区指针（用户态虚拟地址）
    char *src_buf, *dst_buf, *src_cvt_buf;

    // RGA buffer 句柄：用于把虚拟地址导入为 RGA 可识别的 handle
    rga_buffer_handle_t src_handle, dst_handle, src_cvt_handle;

    // 分配内存
    // 注意：这里按 img_height*img_width*img_channel 分配，确保与 bkg 的实际尺寸一致
    // 新手提示：malloc 失败要检查返回值，否则后续 memcpy 会崩溃
    src_buf     = (char *)malloc(img_height * img_width * img_channel);            // 存放 BGR 原图数据
    src_cvt_buf = (char *)malloc(img_height * img_width * img_channel);            // 存放 RGB 转换后的数据
    dst_buf     = (char *)malloc(resize_height * resize_width * resize_channel);   // 存放 resize 后的 RGB 数据（送入 NPU）

    if (bkg.empty()) 
    {
        printf("错误：bkg Mat 对象为空，可能未正确初始化！\n");
        // 可以选择直接返回错误，或者采取其他错误处理措施
        return -1; // 返回错误代码，表示初始化失败
    }

    // 复制数据并初始化内存
    // bkg.data 指向 Mat 的原始像素数据（连续内存时可直接 memcpy）
    memcpy(src_buf, bkg.data, img_height * img_width * img_channel);
    memset(src_cvt_buf, 0x00, img_height * img_width * img_channel);               // 先清零，避免脏数据
    memset(dst_buf,     0x00, resize_height * resize_width * resize_channel);      // 先清零

    // 导入缓冲区：将用户虚拟地址注册到 RGA，得到 handle
    src_handle     = importbuffer_virtualaddr(src_buf,     img_height * img_width * img_channel);
    src_cvt_handle = importbuffer_virtualaddr(src_cvt_buf, img_height * img_width * img_channel);
    dst_handle     = importbuffer_virtualaddr(dst_buf,     resize_height * resize_width * resize_channel);

    if(src_handle == 0 || src_cvt_handle == 0 || dst_handle == 0)
    {
        printf("import va failed.\n");
        // 新手提示：这里最好 return 或 goto release_buffer，否则后续 wrapbuffer_handle 会出错
    }
    
    // 定义 rga_buffer_t：描述图像的宽高和像素格式
    // 注意：这里 src 是 BGR，src_cvt 是 RGB，dst 也是 RGB（符合很多 YOLO 训练输入）
    rga_buffer_t src     = wrapbuffer_handle(src_handle,     img_width,  img_height,  RK_FORMAT_BGR_888);
    rga_buffer_t src_cvt = wrapbuffer_handle(src_cvt_handle, img_width,  img_height,  RK_FORMAT_RGB_888);
    rga_buffer_t dst     = wrapbuffer_handle(dst_handle,     resize_width, resize_height, RK_FORMAT_RGB_888);

    // 检查图像格式/参数是否合理（src -> dst 是否可操作）
    ret = imcheck(src, dst, {}, {});
    if(ret != IM_STATUS_NOERROR)
    {
        printf("%d, imcheck error! %s\n", __LINE__,  imStrError((IM_STATUS)ret));
        ret = -1;
        // goto release_buffer; // 你原本想用 goto 做统一释放，这里注释掉了
    }

    // 颜色转换：BGR -> RGB
    ret = imcvtcolor(src, src_cvt, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
    if(ret == IM_STATUS_SUCCESS)
    {
        // printf("convert color OK!\n");
    }   
    else
    {
        printf("%d, cvtColor error! %s\n", __LINE__,  imStrError((IM_STATUS)ret));
        ret = -1;
        // goto release_buffer;
    }

    // 调整图像大小：src_cvt -> dst
    ret = imresize(src_cvt, dst);
    if(ret == IM_STATUS_SUCCESS)
    {
        // printf("resize color OK!\n");
    }
    else
    {
        printf("%d, resize error! %s\n", __LINE__,  imStrError((IM_STATUS)ret));
        ret = -1;
        // goto release_buffer;
    }
    
    // 记录结束时间并计算处理时间
    end      = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // printf("rga process time : %ld ms.\n", duration.count());

    // 将处理后的 buffer 包装成 Mat，便于保存/调试
    // 注意：Mat 这里并不拷贝数据，只是“引用” src_cvt_buf/dst_buf 指针
    img_cvt = Mat(img_height, img_width, CV_8UC3, src_cvt_buf);
    cv::imwrite("img_rga_cvt.jpg", img_cvt); // 保存颜色转换结果（调试：耗时+占磁盘）
    img_rga = Mat(resize_height, resize_width, CV_8UC3, dst_buf);
    cv::imwrite("img_rga_rsz.jpg", img_rga); // 保存 resize 结果（调试用）

    // ================================ RKNN 推理：inputs_set -> run -> outputs_get ================================
    start = std::chrono::high_resolution_clock::now();

    // 组织输入结构体数组
    int inputs_num = num_tensors.n_input;    // 模型输入个数
    rknn_input inputs[inputs_num];           // C99 风格 VLA（变长数组），部分编译器可能不支持
                                             // 新手提示：更稳妥写法是用 std::vector<rknn_input> inputs(inputs_num);
    memset(inputs, 0, sizeof(inputs));       // 清零，避免未初始化字段

    // 这里只设置第 0 个输入（大多数 YOLO 模型只有 1 个输入）
    inputs[0].index        = 0;                              // 输入索引
    inputs[0].type         = RKNN_TENSOR_UINT8;              // 输入数据类型：uint8（量化模型常见）
    inputs[0].size         = model_height * model_width * model_channel; // 输入 buffer 大小（字节）
    inputs[0].pass_through = false;                          // false 表示让 RKNN 做必要的转换（如量化/归一等，取决于模型/SDK）
    inputs[0].fmt          = RKNN_TENSOR_NHWC;                // 输入布局：NHWC（HWC）
    inputs[0].buf          = dst_buf;                         // 输入数据指针：RGA resize 后的 RGB 图像

    // 设置模型输入到 RKNN 上下文
    rknn_inputs_set(context, inputs_num, inputs);

    // 组织输出结构体数组
    int outputs_num = num_tensors.n_output; // 模型输出个数（YOLOv5 常见 3 个输出头）
    rknn_output outputs[outputs_num];       // 变长数组（同上，建议 vector 更稳）
    memset(outputs, 0, sizeof(outputs));

    // want_float = 0 表示输出保持量化形式（int8/uint8 等），更快；后处理需要 zp/scale 反量化
    for (int i = 0; i < outputs_num; i++)
    {
        outputs[i].want_float = 0;
    }

    // 执行推理
    ret = rknn_run(context, NULL);
    if (ret == 0)
    {
        //printf("model inferencing OK!\n");
    }

    // 获取模型输出（buffer 指针通常会由 RKNN 分配或指向内部内存）
    rknn_outputs_get(context, outputs_num, outputs, NULL);

    end      = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //printf("model inferencing time : %ld ms.\n", duration.count());

    // ================================ 后处理：把输出 tensor 解码成检测框，并映射回原图尺度 ================================
    // postprocess  640 / 960
    // scale_w / scale_h：将原图坐标映射到模型输入坐标时的缩放比（这里是 model / img）
    // 注意：你这里用了 padding 后的 img_width/img_height（可能比 orig_img 大）
    // 这会影响框的还原，实际是否正确要看 post_process 内部是否考虑 padding/letterbox
    float scale_w = (float)model_width  / img_width;
    float scale_h = (float)model_height / img_height;

    vector<int32_t> qnt_zps;   // 存放每个输出的 zero point（量化零点）
    vector<float>   qnt_scales;// 存放每个输出的 scale（量化比例）

    for (int i = 0; i < outputs_num; i++)
    {
        //printf("第%d个output的zp和scale分别是：",i);
        qnt_zps.emplace_back(output_attrs[i].zp);       // 从 rknn_query 得到的输出属性里取 zp
        qnt_scales.emplace_back(output_attrs[i].scale); // 取 scale
        // printf("%d,%f\n",output_attrs[i].zp,output_attrs[i].scale);
    }

    // 进行后处理操作：把 3 个输出 head 解码 + 置信度过滤 + NMS
    // outputs[i].buf：量化输出 buffer（因为 want_float=0）
    // result_group：输出检测结果（框坐标、类别、置信度）
    int debug = post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, 
                model_height, model_width, box_conf_threshold, nms_threshold,
                scale_w, scale_h, qnt_zps, qnt_scales, result_group);

    // 绘制结果到原图上
    // 注意：这里传的是 orig_img（原始尺寸，未 padding），而 scale_w/scale_h 用的是 padding 后的 img_width/img_height
    // 若框有偏移/比例不对，重点检查：padding 是否影响坐标映射，以及 post_process 是否处理了 pad 的偏移量
    draw_result(orig_img, result_group);

    ret = 0; // 这里固定返回 0（即使前面某些步骤失败也可能被覆盖），新手建议：按步骤返回真实错误码

// release_buffer:
    //free
    // 释放资源：释放 RGA handle，再 free malloc 的内存
    if(src_handle)
    {
        releasebuffer_handle(src_handle); // 释放导入的 src handle
    }
    if(src_cvt_handle)
    {
        releasebuffer_handle(src_cvt_handle); // 释放导入的 src_cvt handle
    }
    if(dst_handle)
    {
        releasebuffer_handle(dst_handle); // 释放导入的 dst handle
    }

    free(src_buf);      // 释放 BGR 原图 buffer
    free(dst_buf);      // 释放 resize 后 RGB buffer（也是 NPU 输入）
    free(src_cvt_buf);  // 释放 RGB 转换 buffer

    // 新手提示：这里缺少 rknn_outputs_release(context, outputs_num, outputs);
    // 在很多 RKNN 版本里 outputs_get 之后需要 release，否则会造成内存泄漏/性能下降
    // 具体要看你的 RKNN SDK 文档与 outputs[i].is_prealloc 等字段

    return ret; // 返回推理状态
}
 
// ================================ 绘制检测结果：rectangle + putText ================================
int Yolov5s::draw_result(const cv::Mat &orig_img, detect_result_group_t& result_group)
{
    // 遍历所有检测框
    for(int i = 0; i < result_group.box_count; i++)
    {
        // 读取第 i 个目标的边界框坐标（左上角 xmin,ymin；右下角 xmax,ymax）
        int xmin = result_group.result[i].box.xmin;
        int ymin = result_group.result[i].box.ymin;
        int xmax = result_group.result[i].box.xmax;
        int ymax = result_group.result[i].box.ymax;

        // 在图像上画矩形框（BGR颜色：蓝色(255,0,0)，线宽 3）
        // 注意：orig_img 这里是 const Mat&，但 OpenCV 的绘图函数需要可写 Mat
        // 你这段代码在严格编译下可能会报错/或通过 const_cast 绕过
        // 建议：把函数参数改为 cv::Mat& orig_img（非 const）
        cv::rectangle(orig_img, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                      cv::Scalar(255, 0, 0, 255), 3);

        // 构造标签字符串："类别:置信度%"
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) // 设置固定小数点表示法，保留两位小数
           << result_group.result[i].label << ":" // label：类别名/类别id（取决于 post_process 定义）
           << result_group.result[i].box_conf*100 << " %"; // box_conf：置信度（0~1），转成百分比
        std::string img_label = ss.str();
        
        // 在框的左上角上方写文字
        cv::putText(
            orig_img,                       // 要添加文字的图像
            img_label,                      // 要添加的文字内容
            cv::Point(xmin, ymin-15),       // 文字左下角坐标（x,y），这里放在框上方 15 像素
            FONT_HERSHEY_SIMPLEX,           // 字体类型
            0.8,                            // 字体缩放比例
            cv::Scalar(0, 0, 255),          // 文字颜色（BGR：红色）
            1,                              // 线条粗细
            cv::LINE_8,                     // 线条类型
            false                           // 是否以左下角为原点（一般 false 即可）
        );
    }
    
    return 0; // 绘制成功
}
