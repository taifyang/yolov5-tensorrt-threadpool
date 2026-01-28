#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>


void LetterBox(const cv::Mat& image, cv::Mat& outImage, const cv::Size& newShape = cv::Size(640, 640), const cv::Scalar& color = cv::Scalar(114, 114, 114));

#endif // PREPROCESS_H