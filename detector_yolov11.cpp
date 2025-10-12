#include "detector.h"


using namespace std;

typedef struct {
    cv::Rect box;
    float confidence;
    int index;
} BBOX;

// Comparator to sort boxes by confidence in descending order
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



YOLODetector::YOLODetector(const std::string& modelPath,
    const bool& isGPU = true,
    const cv::Size& inputSize = cv::Size(640, 640))
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    this->isDynamicInputShape = false;
    // checking if width and height are dynamic
    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
    {
        std::cout << "Dynamic input shape" << std::endl;
        this->isDynamicInputShape = true;
    }

    for (auto shape : inputTensorShape)
        std::cout << "Input shape: " << shape << std::endl;

    Ort::AllocatedStringPtr input_node_name = session.GetInputNameAllocated(0, allocator);
    char* temp_buf = new char[50];
    strcpy_s(temp_buf, sizeof(temp_buf), input_node_name.get());
    inputNames.push_back(temp_buf);

    Ort::AllocatedStringPtr output_node_name = session.GetOutputNameAllocated(0, allocator);
    temp_buf = new char[10];
    strcpy_s(temp_buf, sizeof(temp_buf), output_node_name.get());
    outputNames.push_back(temp_buf);

    /*   inputNames.push_back(session.GetInputName(0, allocator));
       outputNames.push_back(session.GetOutputName(0, allocator));*/

    std::cout << "Input name: " << inputNames[0] << std::endl;
    std::cout << "Output name: " << outputNames[0] << std::endl;

    this->inputImageShape = cv::Size2f(inputSize);
}

void YOLODetector::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
    float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }

}

void YOLODetector::preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterbox(resizedImage, resizedImage, this->inputImageShape,
        cv::Scalar(114, 114, 114), this->isDynamicInputShape,
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

std::vector<Detection> YOLODetector::postprocessing(const cv::Size& resizedImageShape,
    const cv::Size& originalImageShape,
    std::vector<Ort::Value>& outputTensors,
    const float& confThreshold, const float& iouThreshold)
{
    std::vector<cv::Rect> boxes;

    Ort::TypeInfo typeInfo = outputTensors.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensors.front().GetTensorMutableData<typename std::remove_pointer<float>::type>();

    // for (const int64_t& shape : outputShape)
    //     std::cout << "Output Shape: " << shape << std::endl;

    // first 5 elements are box[4] and obj confidence
    int signalResultNum = outputNodeDims[1];//84
    int strideNum = outputNodeDims[2];//8400
    std::vector<int> class_ids;
    std::vector<float> confidences;

    cv::Mat rawData;

        // FP32

        rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);

        // FP16
        //rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
        //rawData.convertTo(rawData, CV_32F);

 
    rawData = rawData.t();

    float* data = (float*)rawData.data;

    for (int i = 0; i < strideNum; ++i)
    {
        float* classesScores = data + 4;
        cv::Mat scores(1, 80, CV_32FC1, classesScores);
        cv::Point class_id;
        double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        if (maxClassScore > confThreshold)
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
        data += signalResultNum;
    }




    std::vector<int> indices;
    /*  cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);*/
    nms_boxes(boxes, confidences, confThreshold, iouThreshold, indices);
    // std::cout << "amount of NMS indices: " << indices.size() << std::endl;

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        utils::scaleCoords(resizedImageShape, det.box, originalImageShape);

        det.conf = confidences[idx];
        det.classId = confidences[idx];
        detections.emplace_back(det);
    }

    return detections;
}

std::vector<Detection> YOLODetector::detect(cv::Mat& image, const float& confThreshold = 0.4,
    const float& iouThreshold = 0.45)
{
    float* blob = nullptr;
    std::vector<int64_t> inputTensorShape{ 1, 3, -1, -1 };
    this->preprocessing(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()
    ));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{ nullptr },
        inputNames.data(),
        inputTensors.data(),
        1,
        outputNames.data(),
        1);

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    std::vector<Detection> result = this->postprocessing(resizedShape,
        image.size(),
        outputTensors,
        confThreshold, iouThreshold);

    delete[] blob;

    return result;
}