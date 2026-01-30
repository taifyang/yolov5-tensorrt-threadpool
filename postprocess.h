#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <opencv2/opencv.hpp>


void nms(std::vector<cv::Rect>& boxes, std::vector<float>& scores, float score_threshold, float nms_threshold, std::vector<int>& indices);

void scale_boxes(cv::Rect& box, cv::Size input_size, cv::Size output_size);


#endif // POSTPROCESS_H