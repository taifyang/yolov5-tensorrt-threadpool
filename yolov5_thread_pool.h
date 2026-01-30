#include "yolov5.h"

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>


class Yolov5ThreadPool
{
private:
#ifndef USE_NVCODEC
    std::queue<std::pair<int, cv::Mat>> tasks;             // <id, img>用来存放任务
#else   
    std::queue<std::pair<int, uint8_t*>> tasks;            // <id, img>用来存放任务
#endif

    std::vector<std::shared_ptr<Yolov5>> yolov5_instances; // 模型实例
    std::map<int, std::vector<Detection>> results;         // <id, objects>用来存放结果（检测框）

#ifndef USE_NVCODEC
    std::map<int, cv::Mat> img_results;                    // <id, img>用来存放结果（图片）
#else
    std::map<int, uint8_t*> img_results;                   // <id, img>用来存放结果（图片）
#endif

    std::vector<std::thread> threads;                      // 线程池
    std::mutex mtx1;
    std::mutex mtx2;
    std::condition_variable cv_task, cv_result;
    bool stop;

    void worker(int id);

public:
    Yolov5ThreadPool();
    ~Yolov5ThreadPool();

    int setUp(const std::string &model_path, int num_threads = 12);     // 初始化

#ifndef USE_NVCODEC
    int submitImgTask(const cv::Mat &img, int id);                      // 提交任务
#else
    int submitImgTask(uint8_t* img, int id);                            // 提交任务
#endif

    int getTargetResult(std::vector<Detection> &objects, int id);       // 获取结果

#ifndef USE_NVCODEC
    int getTargetImgResult(cv::Mat &img, int id);                       // 获取结果（图片）
#else
    int getTargetImgResult(uint8_t* img, int id);                       // 获取结果（图片）
#endif

    void stopAll();                                                     // 停止所有线程
};

