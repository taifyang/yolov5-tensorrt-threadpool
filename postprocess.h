#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <opencv2/opencv.hpp>


struct Detection
{
	int id;             //class id
	float score;   		//score
	cv::Rect bbox;      //bounding box
};

void nms(std::vector<cv::Rect>& boxes, std::vector<float>& scores, float score_threshold, float nms_threshold, std::vector<int>& indices);

void scale_boxes(cv::Rect& box, cv::Size input_size, cv::Size output_size);

void draw_detections(cv::Mat& image, std::vector<Detection>& detections);

#endif // POSTPROCESS_H