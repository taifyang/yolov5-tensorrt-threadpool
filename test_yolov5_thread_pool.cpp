#include "yolov5_thread_pool.h"
#include <chrono>


static int g_frame_start_id = 0; // 读取视频帧的索引
static int g_frame_end_id = 0;   // 模型处理完的索引
static Yolov5ThreadPool *g_pool = nullptr;
bool end = false;


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
        g_pool->submitTask(img.clone(), g_frame_start_id++);
    }
    cap.release();
}


void get_results()
{
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    //cv::VideoWriter writer = cv::VideoWriter("result.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(1280, 720));
    while (true)
    {
        cv::Mat img;
        auto ret = g_pool->getTargetImgResult(img, g_frame_end_id++);
        if (end)
        {
            g_pool->stopAll();
            break;
        }
        //cv::imwrite("output/" + std::to_string(g_frame_end_id) + ".jpg", img);
        //writer << img;

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
    g_pool = new Yolov5ThreadPool();
    g_pool->setUp(argv[1], atoi(argv[2]));

    std::thread read_stream_thread(read_stream, "bj_full.mp4");
    std::thread result_thread(get_results);

    read_stream_thread.join();
    result_thread.join();

    return 0;
}