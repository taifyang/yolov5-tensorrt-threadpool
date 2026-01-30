#include "yolov5.h"

int main()
{
    Yolov5* yolov5 = new Yolov5();
    yolov5->load_model("yolov5n_int8.engine");

    cv::Mat image = cv::imread("bus.jpg");
    std::vector<Detection> detections;
    yolov5->infer(image, detections);
    std::cout << "detections size: " << detections.size() << std::endl;
    
    clock_t start = clock();
    for (int i = 0; i < 1; i++)
    {
        cv::Mat image = cv::imread("bus.jpg");
        yolov5->infer(image, detections);
    }
    clock_t end = clock();
    std::cout << "time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    yolov5->draw_detections(0, image, detections);
    cv::imwrite("result.jpg", image);

    return 0;
}

