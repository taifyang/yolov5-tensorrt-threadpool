#include "yolov5.h"

int main()
{
    Yolov5* yolov5 = new Yolov5();
    yolov5->load_model("yolov5n_int8.engine");

    cv::Mat image = cv::imread("bus.jpg");
    std::vector<Detection> detections;
    yolov5->infer(image, detections);
    std::cout << "detections size: " << detections.size() << std::endl;
    yolov5->draw_detections(0, image, detections);
    cv::imwrite("result.jpg", image);
    
    cv::VideoCapture cap("bj_full.mp4");
    if (!cap.isOpened())
        return -1;

    int frame_count = 0;
    auto start_all = std::chrono::high_resolution_clock::now();
    while (true)
    {
        cap >> image;
        if (image.empty())
            break;
        yolov5->infer(image, detections);
        frame_count++;
        auto end_all = std::chrono::high_resolution_clock::now();
        auto elapsed_all = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count() / 1000.f;
        if (elapsed_all >= 1000)
        {
            printf("FPS:%f \n", frame_count / (elapsed_all / 1000.0f));
            frame_count = 0;
            start_all = std::chrono::high_resolution_clock::now();
        }
    }
    cap.release();

    return 0;
}
