#include "yolov5_thread_pool.h"
#include "global_var.h"
#include <chrono>

#ifdef USE_NVCODEC
#include "nvdec_decode.h"
#endif

static int g_frame_start_id = 0; // 读取视频帧的索引
static int g_frame_end_id = 0;   // 模型处理完的索引
static Yolov5ThreadPool *g_pool = nullptr;
bool end = false;


#ifndef USE_NVCODEC
void read_stream(const std::string& video_file)
{
    cv::VideoCapture cap(video_file);
    if (!cap.isOpened())
        return;

    cv::Mat img;
    while (true)
    {
        cap >> img;
        if (img.empty())
        {
            end = true;
            break;
        }
        g_pool->submitImgTask(img.clone(), g_frame_start_id++);
    }
    cap.release();
}
#else
void read_stream_nvcodec(const std::string& video_file)
{
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    int video_stream_idx = 0;
    int ret = 0;

    // 打开视频
    ret = avformat_open_input(&fmt_ctx, video_file.c_str(), nullptr, nullptr);
    if (ret < 0)
    {
        printf("avformat_open_input failed\n");
        return;
    }
    else{
        printf("avformat_open_input success\n");
    }

    // 查找流信息
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0)
    {
        printf("avformat_find_stream_info failed\n");
        return;
    }
    else{    
        printf("avformat_find_stream_info success\n");
    }

    // 找视频流
    video_stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);

    // 创建 CUDA 设备上下文
    ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (ret < 0)
    {
        printf("av_hwdevice_ctx_create failed\n");
        return;
    }
    else{    
        printf("av_hwdevice_ctx_create success\n");
    }

    // 强制使用 NVDEC 解码器
    const AVCodec* codec = avcodec_find_decoder_by_name("h264_cuvid");
    if (!codec) {
        codec = avcodec_find_decoder_by_name("hevc_cuvid"); // 兼容 HEVC
    }
    if (!codec) {
        std::cerr << "错误：未找到 NVDEC 解码器（h264_cuvid/hevc_cuvid）" << std::endl;
        return;
    }
    else{
        printf("avcodec_find_decoder_by_name success\n");
    }

    // 初始化解码器上下文
    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        std::cerr << "avcodec_alloc_context3 failed" << std::endl;
        return;
    }
    else{
        printf("avcodec_alloc_context3 success\n");
    }

    ret = avcodec_parameters_to_context(codec_ctx, fmt_ctx->streams[video_stream_idx]->codecpar);
    if (ret < 0)
    {
        printf("avcodec_parameters_to_context failed\n");
        return;
    }
    else{     
        printf("avcodec_parameters_to_context success\n");
    }

    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    codec_ctx->thread_count = 1; // GPU 解码无需多线程

    ret = avcodec_open2(codec_ctx, codec, nullptr);
    if( ret < 0)
    {
        printf("avcodec_open2 failed\n");
        return;
    }
    else{     
        printf("avcodec_open2 success\n");
    }

    // ========== 2. 全局变量：复用帧参数和显存 ==========
    AVPacket pkt;
    av_init_packet(&pkt);
    pkt.data = nullptr;
    pkt.size = 0;

    AVFrame* hw_frame = av_frame_alloc();
    if (!hw_frame) {
        printf("av_frame_alloc failed\n");
        return;
    }
    else{
        printf("av_frame_alloc success\n");
    }

    uint8_t* dev_bgr = nullptr; // 全局复用的BGR显存
    bool is_param_init = false; // 参数是否已初始化
    int frame_count = 0;        // 帧计数器

    while (av_read_frame(fmt_ctx, &pkt) >= 0) {
        if (pkt.stream_index != video_stream_idx) {
            av_packet_unref(&pkt);
            continue;
        }

        // 发送数据包
        ret = avcodec_send_packet(codec_ctx, &pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            av_packet_unref(&pkt);
            continue;
        }
        if(ret == AVERROR_EOF)
            goto cleanup;

        // 接收解码帧
        while (ret >= 0) {
            ret = avcodec_receive_frame(codec_ctx, hw_frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) 
                break;

            frame_count++;

            // ========== 4. 仅第一帧执行：参数校验 + 显存分配 ==========
            if (!is_param_init) {
                // 一次性校验帧参数
                ret = check_nv12_frame_once(hw_frame, global_w, global_h, global_y_stride, global_nv12_size, global_bgr_size);
                if (ret < 0) {
                    av_frame_unref(hw_frame);
                    av_packet_unref(&pkt);
                    //goto cleanup;
                }

                // 一次性分配BGR显存（复用至所有帧）
                ret = cudaMalloc(&dev_bgr, global_bgr_size);

                is_param_init = true; // 标记参数已初始化
            }

            // ========== 5. 快速校验：防止极端情况（帧尺寸突变） ==========
            if (hw_frame->width != global_w || hw_frame->height != global_h || hw_frame->linesize[0] != global_y_stride) {
                std::cerr << "错误：帧参数突变！当前帧宽=" << hw_frame->width << " 全局宽=" << global_w << std::endl;
                av_frame_unref(hw_frame);
                continue;
            }

            // ========== 6. GPU 转换（复用显存/参数） ==========
            ret = nv12_to_bgr_cuda((uint8_t*)hw_frame->data[0], global_w, global_h, global_y_stride, dev_bgr);
            if (ret < 0) {
                av_frame_unref(hw_frame);
                continue;
            }

            g_pool->submitImgTask(dev_bgr, g_frame_start_id++);
            av_frame_unref(hw_frame);
        }

        av_packet_unref(&pkt);
    }

cleanup:
    av_frame_free(&hw_frame);
    avcodec_free_context(&codec_ctx);
    av_buffer_unref(&hw_device_ctx);
    avformat_close_input(&fmt_ctx);
    exit(0);
}
#endif

void get_results()
{
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    //cv::VideoWriter writer = cv::VideoWriter("result.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(1280, 720));
    while (true)
    {
        cv::Mat img;
        std::vector<Detection> detections;
        auto ret = g_pool->getTargetResult(detections, g_frame_end_id++);
        //auto ret = g_pool->getTargetImgResult(img, g_frame_end_id++);
        if (end)
        {
            g_pool->stopAll();
            break;
        }

        //writer << img;
        //std::cout << "frame:" << g_frame_end_id<<" "<<detections.size() << std::endl;

        frame_count++;
        auto end_all = std::chrono::high_resolution_clock::now();
        auto elapsed_all_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count() / 1000.f;
        if (elapsed_all_2 >= 1000)
        {
            printf("FPS:%f \n", frame_count / (elapsed_all_2 / 1000.0f));
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }
    g_pool->stopAll();
}


int main(int argc, char **argv)
{
    if( argc < 3)
    {
        printf("Usage: %s <model_path> <video_path> <thread_num>\n", argv[0]);
        return -1;
    }

    g_pool = new Yolov5ThreadPool();
    g_pool->setUp(argv[1], atoi(argv[3]));

#ifndef USE_NVCODEC
    std::cout << "Using opencv" << std::endl;
    std::thread read_stream_thread(read_stream, argv[2]);
#else
    std::cout << "Using nvcodec" << std::endl;
    std::thread read_stream_thread(read_stream_nvcodec, argv[2]);
#endif

    std::thread result_thread(get_results);

    read_stream_thread.join();
    result_thread.join();

    return 0;
}