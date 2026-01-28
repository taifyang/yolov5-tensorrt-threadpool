#include "yolov5.h"
#include "logger.h"


Yolov5::Yolov5()
{
	runtime = nvinfer1::createInferRuntime(logger);

	cudaMallocHost(&input_h, sizeof(float) * input_numel);
    cudaMallocHost(&output_h, sizeof(float) * output_numel);

	cudaMalloc(&input_d, sizeof(float) * input_numel);
	cudaMalloc(&output_d, sizeof(float) * output_numel);

	bindings[0] = input_d;
	bindings[1] = output_d;

#ifdef USE_CUDA
	cudaMallocHost(&input_host, max_input_size);
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
	cudaFree(input_d);
    cudaFree(output_d);
	cudaFreeHost(input_h);
    cudaFreeHost(output_h);
}


int Yolov5::load_model(const std::string& model_path)
{
    std::ifstream in(model_path, std::ios::binary);
    if (!in.is_open()) 
		return -1;

    in.seekg(0, std::ios::end);
    size_t size = in.tellg();
    std::vector<unsigned char> engine_data(size);
    in.seekg(0, std::ios::beg);
    in.read((char*)engine_data.data(), size);
    in.close();

    engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
	if(engine == nullptr) 
		return -1;

    execution_context = engine->createExecutionContext();
	if(execution_context == nullptr)
		return -1;

    cudaStreamCreate(&stream);
    return 0;
}


int Yolov5::pre_process(const cv::Mat &image)
{
#ifdef USE_CUDA
	cudaMemcpyAsync(input_host, image.data, sizeof(uint8_t) * 3 * image.cols * image.rows, cudaMemcpyHostToDevice, stream);
	preprocess_kernel_img(input_host, image.cols, image.rows, input_d, input_size.width, input_size.height, d2s_host, s2d_host, stream);
	cudaMemcpyAsync(d2s_device, d2s_host, sizeof(float) * 6, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(s2d_device, s2d_host, sizeof(float) * 6, cudaMemcpyHostToDevice, stream);
#else
	cv::Mat letterbox;
	LetterBox(image, letterbox, input_size);
	//cv::resize(image, letterbox, input_size);
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

#ifndef USE_CUDA
	cudaMemcpyAsync(output_h, output_d, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
#endif

    post_process(image, detections);
    return 0;
}


int Yolov5::post_process(const cv::Mat &image,  std::vector<Detection>& detections)
{
    std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> class_ids;

#ifdef USE_CUDA
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
	// float x_ratio = float(image.cols) / input_size.width;
	// float y_ratio = float(image.rows) / input_size.height;
	for (int i = 0; i < output_numbox; ++i)
	{
		float* ptr = output_h + i * output_numprob;
		float obj_score = ptr[4];
		if (obj_score > confidence_threshold)
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
		scale_boxes(box, input_size, image.size());
		boxes.push_back(box);
		scores.push_back(score);
		class_ids.push_back(class_id);
	}

	std::vector<int> indices;
	nms(boxes, scores, score_threshold, nms_threshold, indices);
	
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
