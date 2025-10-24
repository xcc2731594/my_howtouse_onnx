# my_howtouse_onnx

使用yolov5 onnx 输出的是 25200x85
使用yolov8 v11 输出的是 84x8400 
所以这里面涉及到一些矩阵转换

yolov5：

    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    // for (const int64_t& shape : outputShape)
    //     std::cout << "Output Shape: " << shape << std::endl;

    // first 5 elements are box[4] and obj confidence
    int numClasses = (int)outputShape[2] - 5;
    int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

    // only for batch size = 1
    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
    {
        float clsConf = it[4];

        if (clsConf > confThreshold)
        {
            int centerX = (int)(it[0]);
            int centerY = (int)(it[1]);
            int width = (int)(it[2]);
            int height = (int)(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            this->getBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }


    yolov11：
    
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
