// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detectors.hpp"

#include <utils/config_factory.h>
#include <utils/image_utils.h>
#include <utils/ocv_common.hpp>
#include <utils/nms.hpp>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/tools/gen_op_registration.h>

#include <algorithm>
#include <map>

namespace {
constexpr size_t ndetections = 200;
constexpr size_t batchSize = 1;

std::map<TfLiteType, std::string> tfLiteTypeToStr {
  {kTfLiteNoType, "kTfLiteNoType"},
  {kTfLiteFloat32, "kTfLiteFloat32"},
  {kTfLiteInt32, "kTfLiteInt32"},
  {kTfLiteUInt8, "kTfLiteUInt8"},
  {kTfLiteInt64, "kTfLiteInt64"},
  {kTfLiteString, "kTfLiteString"},
  {kTfLiteBool, "kTfLiteBool"},
  {kTfLiteInt16, "kTfLiteInt16"},
  {kTfLiteComplex64, "kTfLiteComplex64"},
  {kTfLiteInt8, "kTfLiteInt8"},
  {kTfLiteFloat16, "kTfLiteFloat16"},
  {kTfLiteFloat64, "kTfLiteFloat64"},
  {kTfLiteComplex128, "kTfLiteComplex128"},
  {kTfLiteUInt64, "TfLiteUInt64"},
  {kTfLiteResource, "kTfLiteResource"},
  {kTfLiteVariant, "kTfLiteVariant"},
  {kTfLiteUInt32, "kTfLiteUInt32"},
  {kTfLiteUInt16, "kTfLiteUInt16"},
};

std::string getShapeString(const TfLiteIntArray& arr) {
    std::ostringstream oss;
    oss << "[";
    std::copy(arr.data, arr.data + arr.size - 1,
        std::ostream_iterator<int>(oss, ","));
    oss << arr.data[arr.size - 1] << "]";
    return oss.str();
}
}  // namespace

TFLiteModel::TFLiteModel(const std::string &modelFile) {
    readModel(modelFile);
}

void TFLiteModel::readModel(const std::string &modelFile) {
    slog::info << "Reading model: " << modelFile << slog::endl;
    model = tflite::FlatBufferModel::BuildFromFile(modelFile.c_str());
    if (!model) {
        throw std::runtime_error("Failed to read model " + modelFile);
    }

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        throw std::runtime_error("Interpreter failed to build");
    }
}

void BlazeFace::generateAnchors() {
    int layerId = 0;
    while (layerId < ssdModelOptions.numLayers) {
        int last_same_stride_layer = layerId;
        int repeats = 0;
        while (last_same_stride_layer < ssdModelOptions.numLayers &&
               ssdModelOptions.strides[last_same_stride_layer] == ssdModelOptions.strides[layerId]) {
            last_same_stride_layer += 1;
            repeats += 2;
        }
        size_t stride = ssdModelOptions.strides[layerId];
        int feature_map_height = inputHeight / stride;
        int feature_map_width = inputWidth / stride;
        for(int y = 0; y < feature_map_height; ++y) {
            float y_center = (y + ssdModelOptions.anchorOffsetY) / feature_map_height;
            for(int x = 0; x < feature_map_width; ++x) {
                float x_center = (x + ssdModelOptions.anchorOffsetX) / feature_map_width;
                for(int i = 0; i < repeats; ++i)
                    anchors.emplace_back(x_center, y_center);
            }
        }

        layerId = last_same_stride_layer;
    }
}

void TFLiteModel::setNumThreads(int nthreads) {
    interpreter->SetNumThreads(nthreads);
}

void TFLiteModel::allocateTensors() {
    interpreter->AllocateTensors();
}

void TFLiteModel::infer() {
    interpreter->Invoke();
}

std::unique_ptr<Result> TFLiteModel::run(const cv::Mat &img) {
    preprocess(img);
    infer();
    return postprocess();
}

BlazeFace::BlazeFace(const std::string &modelFile, float threshold)
    : TFLiteModel(modelFile),
      confidenceThreshold(threshold) {
    checkInputsOutputs();
    allocateTensors();
    generateAnchors();
}

void BlazeFace::checkInputsOutputs() {
    if (interpreter->inputs().size() != 1) {
        throw std::logic_error("Model expected to have only one input");
    }
    std::string inputName = interpreter->GetInputName(0);
    auto inputTensor = interpreter->input_tensor(0);
    auto dims = inputTensor->dims;
    inputWidth = dims->data[2];
    inputHeight = dims->data[1];

    if (dims->size != 4) {
        throw std::logic_error("Models input expected to have 4 demensions, not " + std::to_string(dims->size));
    }
    slog::info << "\tInputs:" << slog::endl;
    slog::info << "\t\t" << inputName << " : " << getShapeString(*dims) << " "
        << tfLiteTypeToStr.at(inputTensor->type) << slog::endl;

    if (interpreter->outputs().size() != 2) {
        throw std::logic_error("Model expected to have 2 outputs");
    }
    slog::info << "\tOutputs:" << slog::endl;
    for (size_t i = 0; i < interpreter->outputs().size(); ++i) {
        auto outputTensor = interpreter->output_tensor(i);
        auto dims = outputTensor->dims;
        slog::info << "\t\t" << interpreter->GetOutputName(i) << " : " << getShapeString(*dims) <<  " "
            << tfLiteTypeToStr.at(outputTensor->type) << slog::endl;
        if (dims->data[dims->size - 1] == 16) {
            boxesTensorId = i;
        } else if (dims->data[dims->size - 1] == 1) {
            scoresTensorId = i;
        } else {
            throw std::logic_error("Incorrect last dimension for " + std::string(interpreter->GetOutputName(i)) + " output!");
        }
    }
}

void BlazeFace::preprocess(const cv::Mat &img) {
    origImageHeight = img.size().height;
    origImageWidth = img.size().width;

    cv::Mat resizedImage = resizeImageExt(img, inputWidth, inputHeight,
        RESIZE_MODE::RESIZE_KEEP_ASPECT_LETTERBOX, cv::INTER_LINEAR);

    imgScale = std::min(static_cast<double>(inputWidth) / origImageWidth,
        static_cast<double>(inputHeight) / origImageHeight);
    xPadding = (inputWidth - std::floor(origImageWidth * imgScale)) / 2;
    yPadding = (inputHeight -  std::floor(origImageHeight * imgScale)) / 2;

    resizedImage.convertTo(resizedImage, CV_32F);
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);
    resizedImage -= means;
    resizedImage /= cv::Mat(scales);

    int channelsNum = resizedImage.channels();
    float* inputTensor = interpreter->typed_input_tensor<float>(0);
    float* imgData = resizedImage.ptr<float>();
    for (int32_t i = 0; i < 128 * 128; i++) {
        for (int32_t c = 0; c < channelsNum; c++) {
            inputTensor[i * channelsNum + c] = imgData[i * 3 + c];
        }
    }
}


void BlazeFace::decodeBoxes(float* boxes) {
    for(int i = 0; i < ssdModelOptions.numBoxes; ++i) {
        size_t scale = inputHeight;
        size_t num_points = ssdModelOptions.numPoints / 2;
        const int start_pos = i * ssdModelOptions.numPoints;
        for(size_t j = 0; j < num_points; ++j) {
            boxes[start_pos + 2 * j]  = boxes[start_pos + 2 * j]  / scale;
            boxes[start_pos + 2 * j + 1]  = boxes[start_pos + 2 * j + 1] / scale;
            if (j != 1) {
                boxes[start_pos + 2 * j] += anchors[i].x;
                boxes[start_pos + 2 * j + 1] += anchors[i].y;
            }
        }

        // convert x_center, y_center, w, h to xmin, ymin, xmax, ymax

        float half_width = boxes[start_pos + 2] / 2;
        float half_height = boxes[start_pos + 3] / 2;
        float center_x = boxes[start_pos];
        float center_y = boxes[start_pos + 1];

        boxes[start_pos] -= half_width;
        boxes[start_pos + 1] -= half_height;

        boxes[start_pos + 2] = center_x + half_width;
        boxes[start_pos + 3] = center_y + half_height;
    }
}

std::pair<std::vector<BBox>, std::vector<float>> BlazeFace::getDetections(const std::vector<float>& scores, float* boxes) {
    std::vector<BBox> detections;
    std::vector<float> filteredScores;
    for(int box_index = 0; box_index < ssdModelOptions.numBoxes; ++box_index) {
        float score = scores[box_index];

        if (score < confidenceThreshold) {
            continue;
        }

        BBox object;
        object.confidence = score;

        const int start_pos = box_index * ssdModelOptions.numPoints;
        const float x0 = (std::min(std::max(0.0f, boxes[start_pos]), 1.0f) * inputWidth - xPadding) / imgScale;
        const float y0 = (std::min(std::max(0.0f, boxes[start_pos + 1]), 1.0f) * inputHeight -yPadding) / imgScale;
        const float x1 = (std::min(std::max(0.0f, boxes[start_pos + 2]), 1.0f) * inputWidth - xPadding) / imgScale;
        const float y1 = (std::min(std::max(0.0f, boxes[start_pos + 3]), 1.0f) * inputHeight - yPadding) / imgScale;

        object.left = static_cast<int>(round(static_cast<double>(x0)));
        object.top  = static_cast<int>(round(static_cast<double>(y0)));
        object.right = static_cast<int>(round(static_cast<double>(x1)));
        object.bottom = static_cast<int>(round(static_cast<double>(y1)));

        filteredScores.push_back(score);
        detections.push_back(object);
    }

    return {detections, filteredScores};
}

std::unique_ptr<Result> BlazeFace::postprocess() {
    std::vector<BBox> faces;
    float* boxesPtr = interpreter->typed_output_tensor<float>(boxesTensorId);
    float* scoresPtr = interpreter->typed_output_tensor<float>(scoresTensorId);

    std::vector<float> scores(scoresPtr, scoresPtr + ssdModelOptions.numBoxes);

    auto sigmoid = [](float& score) {
        score = 1.f / (1.f + exp(-score));
    };

    std::for_each(scores.begin(), scores.end(), sigmoid);
    // auto max_score = *std::max_element(std::begin(scores), std::end(scores));
    decodeBoxes(boxesPtr);

    auto [detections, filteredScores] = getDetections(scores, boxesPtr);
    std::vector<int> keep = nms(detections, filteredScores, 0.5);
    DetectionResult* result = new DetectionResult();
    for(auto& index : keep) {
        result->boxes.push_back(detections[index]);
    }
    return std::unique_ptr<Result>(result);
}

cv::Mat FaceMesh::enlargeFaceRoi(const cv::Mat& img, cv::Rect roi) {
    int inflationX = std::lround(roi.width * roiEnlargeCoeff);
    int inflationY = std::lround(roi.height * roiEnlargeCoeff);
    roi -= cv::Point(inflationX / 2, inflationY / 2);
    roi += cv::Size(inflationX, inflationY);
    return img(roi);
}

FaceMesh::FaceMesh(const std::string &modelFile)
    : TFLiteModel(modelFile) {
    checkInputsOutputs();
    allocateTensors();
}

void FaceMesh::checkInputsOutputs() {
    if (interpreter->inputs().size() != 1) {
        throw std::logic_error("Model expected to have only one input");
    }
    std::string inputName = interpreter->GetInputName(0);
    auto inputTensor = interpreter->input_tensor(0);
    auto dims = inputTensor->dims;
    inputWidth = dims->data[2];
    inputHeight = dims->data[1];
    if (dims->size != 4) {
        throw std::logic_error("Models input expected to have 4 demensions, not " + std::to_string(dims->size));
    }
    slog::info << "\tInputs:" << slog::endl;
    slog::info << "\t\t" << inputName << " : " << getShapeString(*dims) << " "
        << tfLiteTypeToStr.at(inputTensor->type) << slog::endl;

    if (interpreter->outputs().size() != 2) {
        throw std::logic_error("Model expected to have 2 outputs");
    }
    slog::info << "\tOutputs:" << slog::endl;
    for (size_t i = 0; i < interpreter->outputs().size(); ++i) {
        auto outputTensor = interpreter->output_tensor(i);
        auto dims = outputTensor->dims;
        slog::info << "\t\t" << interpreter->GetOutputName(i) << " : " << getShapeString(*dims) <<  " "
            << tfLiteTypeToStr.at(outputTensor->type) << slog::endl;
    }
}

void FaceMesh::preprocess(const cv::Mat &img) {
    origImageHeight = img.size().height;
    origImageWidth = img.size().width;

    cv::Mat resizedImage = resizeImageExt(img, 192, 192,
        RESIZE_MODE::RESIZE_KEEP_ASPECT_LETTERBOX, cv::INTER_LINEAR);

    imgScale = std::min(static_cast<double>(192) / origImageWidth,
        static_cast<double>(192) / origImageHeight);
    xPadding = (192 - std::floor(origImageWidth * imgScale)) / 2;
    yPadding = (192 -  std::floor(origImageHeight * imgScale)) / 2;

    resizedImage.convertTo(resizedImage, CV_32F);
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);
    resizedImage -= means;
    resizedImage /= cv::Mat(scales);

    int channelsNum = resizedImage.channels();
    float* inputTensor = interpreter->typed_input_tensor<float>(0);
    float* imgData = resizedImage.ptr<float>();
    for (int32_t i = 0; i < 192 * 192; i++) {
        for (int32_t c = 0; c < channelsNum; c++) {
            inputTensor[i * channelsNum + c] = imgData[i * 3 + c];
        }
    }
}

std::unique_ptr<Result> FaceMesh::postprocess() {
    std::vector<cv::Point> landmarks;
    float* landmarksPtr = interpreter->typed_output_tensor<float>(0);

    LandmarksResult* result = new LandmarksResult();
    for (int i = 0; i < 1404; ++i) {
        float x = landmarksPtr[i * 3];
        float y = landmarksPtr[i * 3 + 1];

        result->landmarks.emplace_back(x, y);
    }

    return std::unique_ptr<Result>(result);
}
