#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;

cv::Mat transformation(const cv::Mat& image, const cv::Size& targetSize, const cv::Scalar& mean, const cv::Scalar& std) {

	cv::Mat resizedImage;
	//图片尺寸缩放
	cv::resize(image, resizedImage, targetSize, 0, 0, cv::INTER_AREA);
	cv::Mat normalized;
	resizedImage.convertTo(normalized, CV_32F);
	cv::subtract(normalized / 255.0, mean, normalized);
	cv::divide(normalized, std, normalized);
	return normalized;
}
typedef struct {
	cv::Rect box;
	float confidence;
	int index;
} BBOX;

bool comp(const BBOX& a, const BBOX& b)
{
	return a.confidence > b.confidence;
}

// Function to calculate the intersection over union (IoU)
static float get_iou_value(cv::Rect rect1, cv::Rect rect2)
{
	int xx1, yy1, xx2, yy2;

	xx1 = cv::max(rect1.x, rect2.x);
	yy1 = cv::max(rect1.y, rect2.y);
	xx2 = cv::min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
	yy2 = cv::min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

	int insection_width, insection_height;
	insection_width = cv::max(0, xx2 - xx1 + 1);
	insection_height = cv::max(0, yy2 - yy1 + 1);

	float insection_area, union_area, iou;
	insection_area = float(insection_width) * insection_height;
	union_area = float(rect1.width * rect1.height + rect2.width * rect2.height - insection_area);
	iou = insection_area / union_area;
	return iou;
}

// Non-Maximum Suppression (NMS) function
void nms_boxes(vector<cv::Rect>& boxes, vector<float>& confidences, float confThreshold, float nmsThreshold, vector<int>& indices)
{
	BBOX bbox;
	vector<BBOX> bboxes;
	int i, j;

	// Construct BBOX structs for easier sorting and manipulation
	for (i = 0; i < boxes.size(); i++)
	{
		bbox.box = boxes[i];
		bbox.confidence = confidences[i];
		bbox.index = i;
		bboxes.push_back(bbox);
	}

	// Sort the bounding boxes by confidence in descending order
	sort(bboxes.begin(), bboxes.end(), comp);

	// Iterate over the sorted bounding boxes
	for (i = 0; i < bboxes.size(); i++)
	{
		if (bboxes[i].confidence < confThreshold)  // Ignore low-confidence boxes
			continue;

		indices.push_back(bboxes[i].index);  // Add the box index to the result

		// Remove overlapping boxes using IoU thresholding
		for (j = i + 1; j < bboxes.size(); j++)
		{
			float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
			if (iou > nmsThreshold)
			{
				// Remove the box by erasing from the vector
				bboxes.erase(bboxes.begin() + j);
				j--;  // Adjust index after removal
			}
		}
	}
}
struct Detection
{
	cv::Rect box;
	float conf{};
	int classId{};
};

void letterbox(const cv::Mat& image, cv::Mat& outImage,
	const cv::Size& newShape = cv::Size(640, 640),
	const cv::Scalar& color = cv::Scalar(114, 114, 114),
	bool auto_ = true,
	bool scaleFill = false,
	bool scaleUp = true,
	int stride = 32)
{
	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width);
	if (!scaleUp)
		r = std::min(r, 1.0f);

	float ratio[2]{ r, r };
	int newUnpad[2]{ (int)std::round((float)shape.width * r),
					 (int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - newUnpad[0]);
	auto dh = (float)(newShape.height - newUnpad[1]);

	if (auto_)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		newUnpad[0] = newShape.width;
		newUnpad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
	{
		cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
	}

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
	cv::Mat resizedImage, floatImage;
	cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
	letterbox(resizedImage, resizedImage, cv::Size(640, 640),
		cv::Scalar(114, 114, 114), false,
		false, true, 32);

	inputTensorShape[2] = resizedImage.rows;
	inputTensorShape[3] = resizedImage.cols;

	resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
	blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
	cv::Size floatImageSize{ floatImage.cols, floatImage.rows };

	// hwc -> chw
	std::vector<cv::Mat> chw(floatImage.channels());
	for (int i = 0; i < floatImage.channels(); ++i)
	{
		chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
	}
	cv::split(floatImage, chw);

}

int main(int argc, char** argv) {
	cv::Scalar mean(0.485, 0.456, 0.406); // 均值
	cv::Scalar std(0.229, 0.224, 0.225);  // 标准差
	cv::Mat frame = cv::imread("D:\\something\\use_onnx_yolov11\\img_1.png");
	std::string onnxpath = "D:\\something\\use_onnx_yolov11\\yolo11n.onnx";
	std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
	Ort::SessionOptions session_options;
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolo.onnx");
	// 设定单个操作(op)内部并行执行的最大线程数,可以提升速度
	session_options.SetIntraOpNumThreads(20);
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
	std::cout << "onnxruntime inference try to use GPU Device" << std::endl;
	// 是否使用GPU

	Ort::Session session_(env, modelPath.c_str(), session_options);

	int input_nodes_num = session_.GetInputCount();
	int output_nodes_num = session_.GetOutputCount();
	std::vector<std::string> input_node_names;
	std::vector<std::string> output_node_names;
	Ort::AllocatorWithDefaultOptions allocator;

	int input_h = 0;
	int input_w = 0;

	// 获得输入信息
	for (int i = 0; i < input_nodes_num; i++) {
		auto input_name = session_.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());
		auto inputShapeInfo = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		int ch = inputShapeInfo[1];
		input_h = inputShapeInfo[2];
		input_w = inputShapeInfo[3];
		std::cout << "input format: " << ch << "x" << input_h << "x" << input_w << std::endl;
	}

	// 获得输出信息 多输出
	for (int i = 0; i < output_nodes_num; i++) {
		auto output_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(output_name.get());
		auto outShapeInfo = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		int batch = outShapeInfo[0];
		int attri = outShapeInfo[1];
		int number = outShapeInfo[2];
		std::cout << "batch :  " << batch << "attri:" << attri << "number:" << number << std::endl;
	}

	// 图象预处理 - 格式化操作
	int64 start = cv::getTickCount();
	cv::Mat rgbImage;
	cv::cvtColor(frame, rgbImage, cv::COLOR_BGR2RGB);
	cv::Size targetSize(input_w, input_h);
	// 对原始图像resize和归一化
	//cv::Mat normalized = transformation(rgbImage, targetSize, mean, std);
	//cv::Mat blob = cv::dnn::blobFromImage(normalized);
	size_t tpixels = input_w * input_h * 3;
	std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
	//std::array<int64_t, 3> output_shape_info{ 1, 84, 8400 };

	float* blob = nullptr;
	std::vector<int64_t> inputTensorShape{ 1, 3, -1, -1 };
	preprocessing(frame, blob, inputTensorShape);
	std::vector<float> inputTensorValues(blob, blob + tpixels);
	//std::array<float, 2> outputTensorValues;

	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, inputTensorValues.data(), tpixels, input_shape_info.data(), input_shape_info.size());
	//Ort::Value output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, outputTensorValues.data(), 84*8400, output_shape_info.data(), output_shape_info.size());
	// 输入一个数据
	const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
	// 输出多个数据
	const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
	std::vector<Ort::Value> ort_outputs;
	try {
		ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, inputNames.size(), outNames.data(), outNames.size());
		//session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, inputNames.size(), outNames.data(), &output_tensor_, outNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}
	// 选择最后一个输出作为最终的mask
	auto output = ort_outputs[0].GetTensorMutableData<float>();
	//auto output = ort_outputs.front().GetTensorMutableData<typename std::remove_pointer<float>::type>();
	auto outShape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	//size_t count = ort_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
	//auto* rawOutput = ort_outputs[0].GetTensorData<float>();
	//std::vector<float> output(rawOutput, rawOutput + count);
	int batch = outShape[0];
	int attri = outShape[1];
	int number = outShape[2];

	int step = attri * number;
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	cv::Mat rawData;
	rawData = cv::Mat(attri, number, CV_32F, output);
	rawData = rawData.t();
	float* data = (float*)rawData.data;
	for (int i = 0; i < number; ++i)
	{
		float* classesScores = data + 4;
		cv::Mat scores(1, 80, CV_32FC1, classesScores);
		cv::Point class_id;
		double maxClassScore;
		cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
		if (maxClassScore > 0.25)
		{
			confidences.push_back(maxClassScore);
			class_ids.push_back(class_id.x);
			float x = data[0];
			float y = data[1];
			float w = data[2];
			float h = data[3];

			int left = int((x - 0.5 * w) * 1);
			int top = int((y - 0.5 * h) * 1);

			int width = int(w * 1);
			int height = int(h * 1);

			boxes.push_back(cv::Rect(left, top, width, height));
		}
		data += attri;
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, 0.3, 0.4, indices);
	//nms_boxes(boxes, confidences, 0.3, 0.4, indices);
	// std::cout << "amount of NMS indices: " << indices.size() << std::endl;

 	std::vector<Detection> detections;
	for (int idx : indices)
	{
		Detection det;
		det.box = cv::Rect(boxes[idx]);
		//utils::scaleCoords(resizedImageShape, det.box, originalImageShape);

		det.conf = confidences[idx];
		det.classId = class_ids[idx];
		detections.emplace_back(det);
	}

	vector<string> classNames = { "1","2","3","4","5","6","7","8" };

	for (const Detection& detection : detections)
	{
		cv::rectangle(frame, detection.box, cv::Scalar(229, 160, 21), 2);

		int x = detection.box.x;
		int y = detection.box.y;

		int conf = (int)std::round(detection.conf * 100);
		int classId = detection.classId;
		//std::string label = classNames[classId] + " 0." + std::to_string(conf);
		std::string label = "123";
		int baseline = 0;
		cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline);
		cv::rectangle(frame,
			cv::Point(x, y - 25), cv::Point(x + size.width, y),
			cv::Scalar(229, 160, 21), -1);

		cv::putText(frame, label,
			cv::Point(x, y - 3), cv::FONT_ITALIC,
			0.8, cv::Scalar(255, 255, 255), 2);
		
	}

	std::string output_path = "D:\\something\\use_onnx_yolov11\\result_image.png";
	cv::imwrite(output_path, frame);
	std::cout << "Result saved to: " << output_path << std::endl;

	// 释放资源
	//session_options.release();
	//session_.release();
	return 0;
}
