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
    slog::info << "Model name: " << model->GetModel() << slog::endl;

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
        int feature_map_height = ssdModelOptions.inputHeight / stride;
        int feature_map_width = ssdModelOptions.inputWidth / stride;
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
    if (dims->size != 4) {
        throw std::logic_error("Models input expected to have 4 demensions, not " + std::to_string(dims->size));
    }
    slog::info << "Inputs:" << slog::endl;
    slog::info << "\t" << inputName << " : " << getShapeString(*dims) << " "
        << tfLiteTypeToStr.at(inputTensor->type) << slog::endl;

    if (interpreter->outputs().size() != 2) {
        throw std::logic_error("Model expected to have 2 outputs");
    }
    slog::info << "Outputs:" << slog::endl;
    for (int i = 0; i < interpreter->outputs().size(); ++i) {
        auto outputTensor = interpreter->output_tensor(i);
        auto dims = outputTensor->dims;
        slog::info << "\t" << interpreter->GetOutputName(i) << " : " << getShapeString(*dims) <<  " "
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

    cv::Mat resizedImage = resizeImageExt(img, ssdModelOptions.inputWidth, ssdModelOptions.inputHeight,
        RESIZE_MODE::RESIZE_KEEP_ASPECT_LETTERBOX, cv::INTER_LINEAR);

    imgScale = std::min(static_cast<double>(ssdModelOptions.inputWidth) / origImageWidth,
        static_cast<double>(ssdModelOptions.inputHeight) / origImageHeight);
    xPadding = (ssdModelOptions.inputWidth - std::floor(origImageWidth * imgScale)) / 2;
    yPadding = (ssdModelOptions.inputHeight -  std::floor(origImageHeight * imgScale)) / 2;

    resizedImage.convertTo(resizedImage, CV_32F);
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);
    resizedImage -= ssdModelOptions.means;
    resizedImage /= cv::Mat(ssdModelOptions.scales);

    int channelsNum = resizedImage.channels();
    float* inputTensor = interpreter->typed_input_tensor<float>(0);
    float* imgData = resizedImage.ptr<float>();
    for (int32_t i = 0; i < 128 * 128; i++) {
        for (int32_t c = 0; c < channelsNum; c++) {
            inputTensor[i * channelsNum + c] = imgData[i * 3 + c];
        }
    }
}

void BlazeFace::infer() {
    interpreter->Invoke();
}

void BlazeFace::decodeBoxes(float* boxes) {
    for(int i = 0; i < ssdModelOptions.numBoxes; ++i) {
        size_t scale = ssdModelOptions.inputHeight;
        size_t num_points = ssdModelOptions.numPoints / 2;
        const int start_pos = i * ssdModelOptions.numPoints;
        for(int j = 0; j < num_points; ++j) {
            boxes[start_pos + 2*j]  = boxes[start_pos + 2*j]  / scale;
            boxes[start_pos + 2*j + 1]  = boxes[start_pos + 2*j + 1] / scale;
            if (j != 1) {
                boxes[start_pos + 2*j] += anchors[i].x;
                boxes[start_pos + 2*j + 1] += anchors[i].y;
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

std::pair<std::vector<FaceBox>, std::vector<float>> BlazeFace::getDetections(const std::vector<float>& scores, float* boxes) {
    std::vector<FaceBox> detections;
    std::vector<float> filteredScores;
    for(int box_index = 0; box_index < ssdModelOptions.numBoxes; ++box_index) {
        float score = scores[box_index];

        if (score < confidenceThreshold) {
            continue;
        }

        FaceBox detected_object;
        detected_object.confidence = score;

        const int start_pos = box_index * ssdModelOptions.numPoints;
        const float x0 = (std::min(std::max(0.0f, boxes[start_pos]), 1.0f) * ssdModelOptions.inputWidth - xPadding) / imgScale;
        const float y0 = (std::min(std::max(0.0f, boxes[start_pos + 1]), 1.0f) * ssdModelOptions.inputHeight -yPadding) / imgScale;
        const float x1 = (std::min(std::max(0.0f, boxes[start_pos + 2]), 1.0f) * ssdModelOptions.inputWidth - xPadding) / imgScale;
        const float y1 = (std::min(std::max(0.0f, boxes[start_pos + 3]), 1.0f) * ssdModelOptions.inputHeight - yPadding) / imgScale;

        detected_object.left = static_cast<int>(round(static_cast<double>(x0)));
        detected_object.top  = static_cast<int>(round(static_cast<double>(y0)));
        detected_object.right = static_cast<int>(round(static_cast<double>(x1)));
        detected_object.bottom = static_cast<int>(round(static_cast<double>(y1)));

        filteredScores.push_back(score);
        detections.push_back(detected_object);
    }

    return {detections, filteredScores};
}

std::vector<FaceBox> BlazeFace::postprocess() {
    std::vector<FaceBox> faces;
    float* boxesPtr = interpreter->typed_output_tensor<float>(boxesTensorId);
    float* scoresPtr = interpreter->typed_output_tensor<float>(scoresTensorId);

    std::vector<float> scores(scoresPtr, scoresPtr + ssdModelOptions.numBoxes);

    auto sigmoid = [](float& score) {
        score = 1.f / (1.f + exp(-score));
    };
    std::for_each(scores.begin(), scores.end(), sigmoid);
    auto max_score = *std::max_element(std::begin(scores), std::end(scores));
    decodeBoxes(boxesPtr);

    auto [detections, filteredScores] = getDetections(scores, boxesPtr);
    std::vector<int> keep = nms(detections, filteredScores, 0.5);
    std::vector<FaceBox> results;
    for(auto& index : keep) {
        results.push_back(detections[index]);
    }
    return results;
}

std::vector<FaceBox> BlazeFace::run(const cv::Mat &img) {
    preprocess(img);
    infer();
    return postprocess();
}

CallStat::CallStat():
    _number_of_calls(0), _total_duration(0.0), _last_call_duration(0.0), _smoothed_duration(-1.0) {
}

double CallStat::getSmoothedDuration() {
    // Additional check is needed for the first frame while duration of the first
    // visualisation is not calculated yet.
    if (_smoothed_duration < 0) {
        auto t = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<ms>(t - _last_call_start).count();
    }
    return _smoothed_duration;
}

double CallStat::getTotalDuration() {
    return _total_duration;
}

double CallStat::getLastCallDuration() {
    return _last_call_duration;
}

void CallStat::calculateDuration() {
    auto t = std::chrono::steady_clock::now();
    _last_call_duration = std::chrono::duration_cast<ms>(t - _last_call_start).count();
    _number_of_calls++;
    _total_duration += _last_call_duration;
    if (_smoothed_duration < 0) {
        _smoothed_duration = _last_call_duration;
    }
    double alpha = 0.1;
    _smoothed_duration = _smoothed_duration * (1.0 - alpha) + _last_call_duration * alpha;
    _last_call_start = t;
}

void CallStat::setStartTime() {
    _last_call_start = std::chrono::steady_clock::now();
}

void Timer::start(const std::string& name) {
    if (_timers.find(name) == _timers.end()) {
        _timers[name] = CallStat();
    }
    _timers[name].setStartTime();
}

void Timer::finish(const std::string& name) {
    auto& timer = (*this)[name];
    timer.calculateDuration();
}

CallStat& Timer::operator[](const std::string& name) {
    if (_timers.find(name) == _timers.end()) {
        throw std::logic_error("No timer with name " + name + ".");
    }
    return _timers[name];
}
