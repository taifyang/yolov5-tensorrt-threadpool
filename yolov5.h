#ifndef YOLOV5_H
#define YOLOV5_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>

#include "preprocess.h"
#include "postprocess.h"

//#define USE_CUDA

#ifdef USE_CUDA
    #include "preprocess.cuh"
    #include "decode.cuh"
#endif


class Yolov5
{
public:
    Yolov5();
    ~Yolov5();

    int load_model(const std::string& model_path);                        // 加载模型
    int infer(const cv::Mat &image, std::vector<Detection> &detections); // 推理运行模型

private:
    int pre_process(const cv::Mat &image);   // 图像预处理
    int post_process(const cv::Mat &image, std::vector<Detection>& detections); // 后处理

    const cv::Size input_size = cv::Size(640, 640);
    const int input_numel = 1 * 3 * input_size.width * input_size.height;
    const float confidence_threshold = 0.5;
    const float score_threshold = 0.25;
    const float nms_threshold = 0.45;
    const int class_num = 80;
    const int output_numprob = 5 + class_num;
    const int output_numbox = 3 * (input_size.width / 8 * input_size.height / 8 + input_size.width / 16 * input_size.height / 16 + input_size.width / 32 * input_size.height / 32);
    const int output_numel = 1 * output_numprob * output_numbox;

    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* execution_context = nullptr;
    cudaStream_t stream = nullptr;
    float* input_h = nullptr;
    float* output_h = nullptr;
    float* input_d = nullptr;          	
	float* output_d = nullptr;
    float* bindings[2]; 

#ifdef USE_CUDA
	uint8_t* input_host;
	float* d2s_host;
   	float* d2s_device; 
    float* s2d_host;
    float* s2d_device;
    float* output_box_host;
    float* output_box_device;
    const int max_box = 1024;
   	const int nubox_element = 7; 
    const int max_input_size = sizeof(float) * 3 * 1024 * 1024;
#endif
};

#endif // YOLOV5_H
