#include <iostream>
#include <chrono>
#include <string>

// CUDA 头文件
#include <cuda.h>
#include <cuda_runtime.h>
// NPP 头文件
#include <npp.h>
// FFmpeg 头文件
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
}
// OpenCV 头文件
#include <opencv2/opencv.hpp>


/**
 * @brief 严格校验 NV12 格式的 CUDA 帧参数（仅需执行一次）
 */
int check_nv12_frame_once(AVFrame* hw_frame, int& w, int& h, int& y_stride, size_t& nv12_size, size_t& bgr_size) 
{
    if (hw_frame->format != AV_PIX_FMT_CUDA) 
    {
        std::cerr << "错误：hw_frame 不是 CUDA 格式！" << std::endl;
        return -1;
    }

    // 获取固定的宽高/行步长（同一视频所有帧一致）
    w = hw_frame->width;
    h = hw_frame->height;
    y_stride = hw_frame->linesize[0];
    // NV12 总大小 = Y(w*h) + UV(w*h/2)
    nv12_size = (size_t)w * h * 3 / 2;
    // BGR 总大小（固定）
    bgr_size = (size_t)w * h * 3;

    // 仅一次校验行步长
    if (y_stride < w) {
        std::cerr << "错误：Y 平面行步长异常！stride=" << y_stride << " width=" << w << std::endl;
        return -1;
    }

    std::cout << "NV12 帧参数校验完成（全局生效）：" << std::endl;
    std::cout << "  宽=" << w << " 高=" << h << " Y行步长=" << y_stride << std::endl;
    std::cout << "  NV12总大小=" << nv12_size << " BGR总大小=" << bgr_size << std::endl;
    return 0;
}

/**
 * @brief GPU 端 NV12 → BGR 转换（复用显存/参数）
 */
int nv12_to_bgr_cuda(uint8_t* dev_nv12, int w, int h, int y_stride, uint8_t* dev_bgr) 
{
    // 1. 计算 NV12 各平面指针（考虑行步长对齐）
    Npp8u* pY = dev_nv12;
    Npp8u* pUV = dev_nv12 + (size_t)y_stride * h;
    
    // 2. 封装为 NPP 要求的指针数组
    const Npp8u* const ppSrc[] = {pY, pUV};
    
    // 3. 复用固定的行步长参数
    int nSrcStep = y_stride;       
    int nDstStep = w * 3;          
    NppiSize oSizeROI = {w, h};    

    // 4. 调用 NPP 转换函数
    NppStatus npp_ret = nppiNV12ToBGR_8u_P2C3R(
        ppSrc,       // 输入：Y+UV 指针数组
        nSrcStep,    // 输入行步长（复用全局参数）
        dev_bgr,     // 输出：复用的BGR显存指针
        nDstStep,    // 输出行步长（复用）
        oSizeROI     // 有效图像尺寸（复用）
    );

    return 0;
}
