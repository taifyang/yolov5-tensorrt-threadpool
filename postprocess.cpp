#include "postprocess.h"


void nms(std::vector<cv::Rect>& boxes, std::vector<float>& scores, float score_threshold, float nms_threshold, std::vector<int>& indices)
{
	assert(boxes.size() == scores.size());

	struct BoxScore
	{
		cv::Rect box;
		float score;
		int id;
	};
	std::vector<BoxScore> boxes_scores;
	for (size_t i = 0; i < boxes.size(); i++)
	{
		BoxScore box_conf;
		box_conf.box = boxes[i];
		box_conf.score = scores[i];
		box_conf.id = i;
		if (scores[i] > score_threshold)	boxes_scores.push_back(box_conf);
	}

	std::sort(boxes_scores.begin(), boxes_scores.end(), [](BoxScore a, BoxScore b) { return a.score > b.score; });

	std::vector<float> area(boxes_scores.size());
	for (size_t i = 0; i < boxes_scores.size(); ++i)
	{
		area[i] = boxes_scores[i].box.width * boxes_scores[i].box.height;
	}

	std::vector<bool> isSuppressed(boxes_scores.size(), false);
	for (size_t i = 0; i < boxes_scores.size(); ++i)
	{
		if (isSuppressed[i])  continue;
		for (size_t j = i + 1; j < boxes_scores.size(); ++j)
		{
			if (isSuppressed[j])  continue;

			float x1 = (std::max)(boxes_scores[i].box.x, boxes_scores[j].box.x);
			float y1 = (std::max)(boxes_scores[i].box.y, boxes_scores[j].box.y);
			float x2 = (std::min)(boxes_scores[i].box.x + boxes_scores[i].box.width, boxes_scores[j].box.x + boxes_scores[j].box.width);
			float y2 = (std::min)(boxes_scores[i].box.y + boxes_scores[i].box.height, boxes_scores[j].box.y + boxes_scores[j].box.height);
			float w = (std::max)(0.0f, x2 - x1);
			float h = (std::max)(0.0f, y2 - y1);
			float inter = w * h;
			float ovr = inter / (area[i] + area[j] - inter);

			if (ovr >= nms_threshold)  isSuppressed[j] = true;
		}
	}

	for (int i = 0; i < boxes_scores.size(); ++i)
	{
		if (!isSuppressed[i])	indices.push_back(boxes_scores[i].id);
	}
}


void scale_boxes(cv::Rect& box, cv::Size input_size, cv::Size output_size)
{
	float gain = std::min(input_size.width * 1.0 / output_size.width, input_size.height * 1.0 / output_size.height);
	int pad_w = (input_size.width - output_size.width * gain) / 2;
	int pad_h = (input_size.height - output_size.height * gain) / 2;
	box.x -= pad_w;
	box.y -= pad_h;
	box.x /= gain;
	box.y /= gain;
	box.width /= gain;
	box.height /= gain;
}

void draw_detections(cv::Mat& image, std::vector<Detection>& detections)
{
    for (int i = 0; i < detections.size(); i++)
    {
        Detection detection = detections[i];
        int idx = detection.id;
        float score = detection.score;
        cv::Rect bbox = detection.bbox;
        std::string label = "class" + std::to_string(idx) + ":" + cv::format("%.2f", score);
        cv::rectangle(image, bbox, cv::Scalar(0, 255, 0), 2);
        cv::putText(image, label, cv::Point(bbox.x, bbox.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    }
}
