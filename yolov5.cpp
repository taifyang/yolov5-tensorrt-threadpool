#include "yolov5.h"
#include "logger.h"
#include "global_var.h"

int global_w = 0, global_h = 0, global_y_stride = 0;
size_t global_nv12_size = 0, global_bgr_size = 0;


Yolov5::Yolov5()
{
	runtime = nvinfer1::createInferRuntime(logger);

	cudaMallocHost(&input_h, sizeof(float) * input_numel);
    cudaMallocHost(&output_h, sizeof(float) * output_numel);

	cudaMalloc(&input_d, sizeof(float) * input_numel);
	cudaMalloc(&output_d, sizeof(float) * output_numel);

	bindings[0] = input_d;
	bindings[1] = output_d;

#ifdef CUDA_PREPROCESS
	cudaMalloc(&input_device, max_input_size);
	cudaMallocHost(&d2s_host, sizeof(float) * 6);
	cudaMalloc(&d2s_device, sizeof(float) * 6);
	cudaMallocHost(&s2d_host, sizeof(float) * 6);
	cudaMalloc(&s2d_device, sizeof(float) * 6);
	cudaMallocHost(&output_box_host, sizeof(float) * (nubox_element * max_box + 1));
	cudaMalloc(&output_box_device, sizeof(float) * (nubox_element * max_box + 1));
#endif
}

Yolov5::~Yolov5()
{
	cudaStreamDestroy(stream);
	cudaFree(input_device);
	cudaFree(input_d);
    cudaFree(output_d);
	cudaFreeHost(input_h);
    cudaFreeHost(output_h);
}

int Yolov5::load_model(const std::string& model_path)
{
    std::ifstream in(model_path, std::ios::binary);
    if (!in.is_open()) 
	{
		std::cerr << "Failed to open engine file: " << model_path << std::endl;
		return -1;
	}

    in.seekg(0, std::ios::end);
    size_t size = in.tellg();
    std::vector<unsigned char> engine_data(size);
    in.seekg(0, std::ios::beg);
    in.read((char*)engine_data.data(), size);
    in.close();

    engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
	if(engine == nullptr) 
	{
		std::cerr << "Deserialize engine failed" << std::endl;
		return -1;
	}

    execution_context = engine->createExecutionContext();
	if(execution_context == nullptr)
	{
		std::cerr << "Create execution context failed" << std::endl;
		return -1;
	}

    cudaStreamCreate(&stream);
    return 0;
}


int Yolov5::pre_process(const cv::Mat &image)
{
#ifdef CUDA_PREPROCESS
	cudaMemcpy(input_device, image.data, sizeof(uint8_t) * 3 * image.cols * image.rows, cudaMemcpyHostToDevice);
	preprocess_cuda(input_device, image.cols, image.rows, input_d, input_size.width, input_size.height, d2s_host, s2d_host);
	cudaMemcpy(d2s_device, d2s_host, sizeof(float) * 6, cudaMemcpyHostToDevice);
	cudaMemcpy(s2d_device, s2d_host, sizeof(float) * 6, cudaMemcpyHostToDevice);

#else
	cv::Mat letterbox;
	LetterBox(image, letterbox, input_size);
	letterbox.convertTo(letterbox, CV_32FC3, 1.0f / 255.0f);

	int image_area = letterbox.cols * letterbox.rows;
	float* pimage = (float*)letterbox.data;
	float* phost_b = input_h + image_area * 0;
	float* phost_g = input_h + image_area * 1;
	float* phost_r = input_h + image_area * 2;
	for (int i = 0; i < image_area; ++i, pimage += 3)
	{
		*phost_r++ = pimage[0];
		*phost_g++ = pimage[1];
		*phost_b++ = pimage[2];
	}

	cudaMemcpyAsync(input_d, input_h, sizeof(float) * input_numel, cudaMemcpyHostToDevice, stream);
#endif

    return 0;
}

int Yolov5::infer(const cv::Mat& image, std::vector<Detection>& detections)
{
    pre_process(image);

	bool success = execution_context->executeV2((void**)bindings);
	if(!success)
	{
	    std::cerr << "Failed to run inference" << std::endl;
	    return -1;
	}

#ifndef CUDA_POSTPROCESS
	cudaMemcpyAsync(output_h, output_d, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
#endif

    post_process(image.size(), detections);
    return 0;
}

#ifdef USE_NVCODEC
int Yolov5::infer_nvcodec(uint8_t* image, std::vector<Detection> &detections)
{
	preprocess_cuda(image, global_w, global_h, input_d, input_size.width, input_size.height, d2s_host, s2d_host);
	// cudaMemcpy(d2s_device, d2s_host, sizeof(float) * 6, cudaMemcpyHostToDevice);
	// cudaMemcpy(s2d_device, s2d_host, sizeof(float) * 6, cudaMemcpyHostToDevice);

	bool success = execution_context->executeV2((void**)bindings);
	if(!success)
	{
		std::cerr << "Failed to run inference" << std::endl;
		return -1;
	}
	cudaMemcpyAsync(output_h, output_d, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	post_process(cv::Size(global_w, global_h), detections);
	return 0;
}
#endif

int Yolov5::post_process(const cv::Size& image_size,  std::vector<Detection>& detections)
{
    std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

#ifdef CUDA_POSTPROCESS
	cudaMemset(output_box_device, 0, sizeof(float) * (nubox_element * max_box + 1));	
	decode_kernel_invoker(output_d, output_numbox, class_num, score_threshold, d2s_device, output_box_device, max_box, nubox_element, stream);
	nms_kernel_invoker(output_box_device, nms_threshold, max_box, nubox_element, stream);
	cudaMemcpyAsync(output_box_host, output_box_device, sizeof(float) * (nubox_element * max_box + 1), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	for (size_t i = 0; i < max_box; i++)
	{
		if (output_box_host[7 * i + 7])
		{
			float x1 = output_box_host[7 * i + 1];
			float y1 = output_box_host[7 * i + 2];
			float x2 = output_box_host[7 * i + 3];
			float y2 = output_box_host[7 * i + 4];
			boxes.push_back(cv::Rect(x1, y1, x2-x1, y2-y1));
			scores.push_back(output_box_host[7 * i + 5]);
			class_ids.push_back(output_box_host[7 * i + 6]);
		}
	}

	detections.clear();
	detections.resize(boxes.size());
	for (int i = 0; i < boxes.size(); i++)
	{
		detections[i].bbox = boxes[i];
		detections[i].score = scores[i];
		detections[i].id = class_ids[i];
	}

#else
	for (int i = 0; i < output_numbox; ++i)
	{
		float* ptr = output_h + i * output_numprob;
		float obj_score = ptr[4];
		if (obj_score < confidence_threshold)
			continue;

		float* classes_scores = 5 + ptr;
		int class_id = std::max_element(classes_scores, classes_scores + class_num) - classes_scores;
		float score = classes_scores[class_id] * obj_score;
		if (score < score_threshold)
			continue;

		float x = ptr[0];
		float y = ptr[1];
		float w = ptr[2];
		float h = ptr[3];
		int left = int(x - 0.5 * w);
		int top = int(y - 0.5 * h);
		int width = int(w);
		int height = int(h);

		cv::Rect box = cv::Rect(left, top, width, height);
		scale_boxes(box, input_size, image_size);
		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, indices);
	//nms(boxes, scores, score_threshold, nms_threshold, indices);

	detections.clear();
	detections.resize(indices.size());
	for (int i = 0; i < indices.size(); ++i)
	{
	    int idx = indices[i];
		detections[i].bbox = boxes[idx];
		detections[i].score = scores[idx];
		detections[i].id = class_ids[idx];
	}
#endif

    return 0;
}

void Yolov5::draw_detections(int id, cv::Mat& image, std::vector<Detection>& detections)
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

	cv::imwrite("output/" + std::to_string(id) + ".jpg", image);
}

void Yolov5::draw_detections(int id, uint8_t* image, std::vector<Detection>& detections)
{
    cv::Mat image_mat(global_h, global_w, CV_8UC3);
	cudaMemcpy(image_mat.data, image, sizeof(uint8_t) * 3 * global_w * global_h, cudaMemcpyDeviceToHost);

    for (int i = 0; i < detections.size(); i++)
    {
        Detection detection = detections[i];
        int idx = detection.id;
        float score = detection.score;
        cv::Rect bbox = detection.bbox;
        std::string label = "class" + std::to_string(idx) + ":" + cv::format("%.2f", score);
        cv::rectangle(image_mat, bbox, cv::Scalar(0, 255, 0), 2);
        cv::putText(image_mat, label, cv::Point(bbox.x, bbox.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    }

	cv::imwrite("output/" + std::to_string(id) + ".jpg", image_mat);
}
