// undistort.cl
// OpenCL kernel for lens undistortion on Mali GPU (RK3588)
//
// 算法：对每个输出像素 (x, y)，计算归一化相机坐标，应用逆畸变模型，
//       映射回输入图像坐标，使用双线性插值采样。
//
// 畸变模型：径向畸变 (k1, k2, k3) + 切向畸变 (p1, p2)
// 针孔相机模型：fx, fy, cx, cy

// 双线性插值采样函数
// 输入：input_img - 输入图像 (BGR 三通道)
//       x, y      - 浮点坐标（可以是亚像素位置）
//       width, height - 图像尺寸
// 输出：BGR 三通道值
inline float4 sample_bilinear(__read_only image2d_t input_img,
                              float x, float y,
                              int width, int height,
                              sampler_t sampler)
{
    // 边界检查：如果坐标超出图像范围，返回黑色
    if (x < 0.0f || x >= (float)(width - 1) ||
        y < 0.0f || y >= (float)(height - 1)) {
        return (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // 取整得到四个相邻像素坐标
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // 计算小数部分（权重）
    float dx = x - (float)x0;
    float dy = y - (float)y0;

    // 读取四个相邻像素
    float4 p00 = read_imagef(input_img, sampler, (int2)(x0, y0));
    float4 p10 = read_imagef(input_img, sampler, (int2)(x1, y0));
    float4 p01 = read_imagef(input_img, sampler, (int2)(x0, y1));
    float4 p11 = read_imagef(input_img, sampler, (int2)(x1, y1));

    // 双线性插值
    float4 top = mix(p00, p10, dx);
    float4 bottom = mix(p01, p11, dx);
    return mix(top, bottom, dy);
}

// 主内核：去畸变
// 每个工作项处理一个输出像素
__kernel void undistort_kernel(
    __read_only image2d_t input_img,      // 输入图像
    __write_only image2d_t output_img,    // 输出图像
    const int width,                       // 图像宽度
    const int height,                      // 图像高度
    const float fx,                        // 相机内参：x 方向焦距
    const float fy,                        // 相机内参：y 方向焦距
    const float cx,                        // 相机内参：光心 x 坐标
    const float cy,                        // 相机内参：光心 y 坐标
    const float k1,                        // 径向畸变系数
    const float k2,
    const float k3,
    const float p1,                        // 切向畸变系数
    const float p2)
{
    // 获取当前工作项的全局 ID（即输出像素坐标）
    int x_out = get_global_id(0);
    int y_out = get_global_id(1);

    // 边界检查
    if (x_out >= width || y_out >= height) {
        return;
    }

    // 步骤 1：将输出像素坐标转换为归一化相机坐标
    // x_norm = (x - cx) / fx
    // y_norm = (y - cy) / fy
    float x_norm = ((float)x_out - cx) / fx;
    float y_norm = ((float)y_out - cy) / fy;

    // 步骤 2：计算径向畸变因子
    // r^2 = x_norm^2 + y_norm^2
    float r2 = x_norm * x_norm + y_norm * y_norm;
    float r4 = r2 * r2;
    float r6 = r2 * r4;

    // 径向畸变因子：1 + k1*r^2 + k2*r^4 + k3*r^6
    float radial_dist = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;

    // 步骤 3：应用切向畸变
    // x_distorted = x_norm * radial_dist + 2*p1*x_norm*y_norm + p2*(r^2 + 2*x_norm^2)
    // y_distorted = y_norm * radial_dist + p1*(r^2 + 2*y_norm^2) + 2*p2*x_norm*y_norm
    float x_distorted = x_norm * radial_dist +
                        2.0f * p1 * x_norm * y_norm +
                        p2 * (r2 + 2.0f * x_norm * x_norm);
    float y_distorted = y_norm * radial_dist +
                        p1 * (r2 + 2.0f * y_norm * y_norm) +
                        2.0f * p2 * x_norm * y_norm;

    // 步骤 4：将畸变后的归一化坐标映射回输入图像像素坐标
    // x_in = fx * x_distorted + cx
    // y_in = fy * y_distorted + cy
    float x_in = fx * x_distorted + cx;
    float y_in = fy * y_distorted + cy;

    // 步骤 5：双线性插值采样
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                        CLK_ADDRESS_CLAMP_TO_EDGE |
                        CLK_FILTER_NEAREST;

    float4 pixel_value = sample_bilinear(input_img, x_in, y_in,
                                         width, height, sampler);

    // 步骤 6：写入输出图像
    write_imagef(output_img, (int2)(x_out, y_out), pixel_value);
}
