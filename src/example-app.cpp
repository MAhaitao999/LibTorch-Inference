#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;

int main(int argc, char *argv[]) {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    torch::jit::script::Module module = torch::jit::load("../models/resnet18.pt");
    std::string image_path = "../images/cat.jpg";

    auto image = cv::imread(image_path, cv::ImreadModes::IMREAD_COLOR);
    // cout << format(image, Formatter::FMT_PYTHON) << endl;
    cv::Mat image_transformed;
    cv::resize(image, image_transformed, cv::Size(224, 224));
    cv::cvtColor(image_transformed, image_transformed, cv::COLOR_BGR2RGB);

    cout << "(" << image_transformed.cols << ", " << image_transformed.rows << ")" << endl;

    // 图像转换为Tensor
    torch::Tensor tensor_image = torch::from_blob(image_transformed.data, {image_transformed.rows, image_transformed.cols, 3}, torch::kByte);
    tensor_image = tensor_image.permute({2, 0, 1});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);
    tensor_image = tensor_image.unsqueeze(0);

    // cout << tensor_image << endl;
    
    at::Tensor output = module.forward({tensor_image}).toTensor();
    auto start = system_clock::now();
    output = module.forward({tensor_image}).toTensor();
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout <<  "推理花费了" 
         << 1000 * double(duration.count()) * microseconds::period::num / microseconds::period::den 
         << "毫秒" << endl;

    auto max_result = output.max(1, true);
    auto max_index = std::get<1>(max_result).item<float>();

    cout << "max index predicted: " << max_index << endl;

    cout << "ok\n";  
}
