#include "preprocess.h"


void LetterBox(const cv::Mat& image, cv::Mat& outImage, const cv::Size& newShape, const cv::Scalar& color)
{
	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]) / 2;
	auto dh = (float)(newShape.height - new_un_pad[1]) / 2;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	else
		outImage = image.clone();

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));

	cv::Vec4d params;
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;

	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}
