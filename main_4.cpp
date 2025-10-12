#include <iostream>
#include <opencv2/opencv.hpp>

#include "cmdline.h"
#include "utils.h"
#include "detector.h"


int main(int argc, char* argv[])
{
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;

    //cmdline::parser cmd;
    //cmd.add<std::string>("model_path", 'm', "Path to onnx model.", true, "yolov5.onnx");
    //cmd.add<std::string>("image", 'i', "Image source to be detected.", true, "bus.jpg");
    //cmd.add<std::string>("class_names", 'c', "Path to class names file.", true, "coco.names");
    //cmd.add("gpu", '\0', "Inference on cuda device.");

    //cmd.parse_check(argc, argv);

    //bool isGPU = cmd.exist("gpu");
    //const std::string classNamesPath = cmd.get<std::string>("class_names");
    //const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    //const std::string imagePath = cmd.get<std::string>("image");
    //const std::string modelPath = cmd.get<std::string>("model_path");
    bool isGPU=false;
    const std::string modelPath = "D:\\something\\use_onnx_yolov11\\yolov5m.onnx";  // Path to your ONNX model
    const std::string imagePath = "D:\\something\\use_onnx_yolov11\\img_1.png";  // Path to the image for detection
    const std::string classNamesPath = "D:\\somethingc++\\onnxruntime_onnx\\coco.names";  // Path to the class names file
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    YOLODetector detector{ nullptr };
    cv::Mat image;
    std::vector<Detection> result;

    try
    {
        detector = YOLODetector(modelPath, isGPU, cv::Size(640, 640));
        std::cout << "Model was initialized." << std::endl;

        image = cv::imread(imagePath);
        result = detector.detect(image, confThreshold, iouThreshold);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    utils::visualizeDetection(image, result, classNames);

    cv::imshow("result", image);
    // cv::imwrite("result.jpg", image);
    cv::waitKey(0);


    std::string output_path = "D:\\something\\use_onnx_yolov11\\result_image.png";
    imwrite(output_path, image);
    std::cout << "Result saved to: " << output_path << std::endl;
    return 0;
}