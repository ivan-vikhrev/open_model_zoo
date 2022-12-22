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

#define _USE_MATH_DEFINES
#include <cmath>

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

std::unique_ptr<Result> TFLiteModel::run(const cv::Mat &img, const std::shared_ptr<MetaData>& metaData) {
    this->metaData = metaData;
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
    xPadding = (inputWidth - std::floor(origImageWidth * imgScale)) / (2 * inputWidth);
    yPadding = (inputHeight -  std::floor(origImageHeight * imgScale)) / (2 * inputHeight);

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
        float xScale = 1 - (xPadding * 2);
        float yScale = 1 - (yPadding * 2);
        const int start_pos = box_index * ssdModelOptions.numPoints;
        const float x0 = (boxes[start_pos] - xPadding) / xScale; // (std::min(std::max(0.0f, boxes[start_pos]), 1.0f) * inputWidth - xPadding);
        const float y0 = (boxes[start_pos + 1]- yPadding) / yScale; //(std::min(std::max(0.0f, boxes[start_pos + 1]), 1.0f) * inputHeight -yPadding);
        const float x1 = (boxes[start_pos + 2] - xPadding) / xScale; // (std::min(std::max(0.0f, boxes[start_pos + 2]), 1.0f) * inputWidth - xPadding);
        const float y1 = (boxes[start_pos + 3] - yPadding) / yScale; //(std::min(std::max(0.0f, boxes[start_pos + 3]), 1.0f) * inputHeight - yPadding);

        const float xLeftEye = (boxes[start_pos + 4] - xPadding) / xScale;; //(std::min(std::max(0.0f, boxes[start_pos + 4]), 1.0f) * inputWidth - xPadding) / imgScale;
        const float yLeftEye = (boxes[start_pos + 5] - yPadding) / yScale; //(std::min(std::max(0.0f, boxes[start_pos + 5]), 1.0f) * inputHeight - yPadding) / imgScale;

        const float xRightEye = (boxes[start_pos + 6] - xPadding) / xScale; // (std::min(std::max(0.0f, boxes[start_pos + 6]), 1.0f) * inputWidth - xPadding) / imgScale;
        const float yRightEye = (boxes[start_pos + 7] - yPadding) / yScale; //(std::min(std::max(0.0f, boxes[start_pos + 7]), 1.0f) * inputHeight - yPadding) / imgScale;

        const float xNose = (std::min(std::max(0.0f, boxes[start_pos + 8]), 1.0f) * inputWidth - xPadding) / imgScale;
        const float yNose = (std::min(std::max(0.0f, boxes[start_pos + 9]), 1.0f) * inputHeight - yPadding) / imgScale;

        const float xMouth = (std::min(std::max(0.0f, boxes[start_pos + 10]), 1.0f) * inputWidth - xPadding) / imgScale;
        const float yMouth = (std::min(std::max(0.0f, boxes[start_pos + 11]), 1.0f) * inputHeight - yPadding) / imgScale;

        const float xLeftTragion = (std::min(std::max(0.0f, boxes[start_pos + 12]), 1.0f) * inputWidth - xPadding) / imgScale;
        const float yLeftTragion  = (std::min(std::max(0.0f, boxes[start_pos + 13]), 1.0f) * inputHeight - yPadding) / imgScale;

        const float xRightTragion = (std::min(std::max(0.0f, boxes[start_pos + 14]), 1.0f) * inputWidth - xPadding) / imgScale;
        const float yRightTragion = (std::min(std::max(0.0f, boxes[start_pos + 15]), 1.0f) * inputHeight - yPadding) / imgScale;


        object.left = x0 * origImageWidth; //clamp(x0, 0.f, static_cast<float>(origImageWidth)) / imgScale;
        object.top  = y0 * origImageHeight; //clamp(y0, 0.f, static_cast<float>(origImageHeight)) / imgScale;
        object.right = x1 * origImageWidth; //clamp(x1, 0.f, static_cast<float>(origImageWidth)) / imgScale;
        object.bottom = y1 * origImageHeight; //clamp(y1, 0.f, static_cast<float>(origImageHeight)) / imgScale;

        object.leftEye = {clamp(xLeftEye * origImageWidth, 0.f, static_cast<float>(origImageWidth)),
             clamp(yLeftEye * origImageHeight, 0.f, static_cast<float>(origImageHeight))};
        object.rightEye = {clamp(xRightEye * origImageWidth, 0.f, static_cast<float>(origImageWidth)),
             clamp(yRightEye * origImageHeight, 0.f, static_cast<float>(origImageHeight))};
        object.nose = {clamp(xNose, 0.f, static_cast<float>(origImageWidth)),
             clamp(yNose, 0.f, static_cast<float>(origImageHeight))};
        object.mouth = {clamp(xMouth, 0.f, static_cast<float>(origImageWidth)),
             clamp(yMouth, 0.f, static_cast<float>(origImageHeight))};
        object.leftTragion = {clamp(xLeftTragion, 0.f, static_cast<float>(origImageWidth)),
             clamp(yLeftTragion, 0.f, static_cast<float>(origImageHeight))};
        object.rightTragion = {clamp(xRightTragion, 0.f, static_cast<float>(origImageWidth)),
             clamp(yRightTragion, 0.f, static_cast<float>(origImageHeight))};

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

    decodeBoxes(boxesPtr);

    auto [detections, filteredScores] = getDetections(scores, boxesPtr);
    std::vector<int> keep = nms(detections, filteredScores, 0.5);
    DetectionResult* result = new DetectionResult();
    for(auto& index : keep) {
        result->boxes.push_back(detections[index]);
    }
    return std::unique_ptr<Result>(result);
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

cv::Rect FaceMesh::enlargeFaceRoi(cv::Rect roi) {
    int inflationX = std::lround(roi.width * roiEnlargeCoeff) - roi.width;
    int inflationY = std::lround(roi.height * roiEnlargeCoeff) - roi.height;
    roi -= cv::Point(inflationX / 2, inflationY / 2);
    roi += cv::Size(inflationX, inflationY);
    return roi;
}

double FaceMesh::calculateRotationRad(cv::Point p0, cv::Point p1) {
    double rad = -std::atan2(p0.y - p1.y, p1.x - p0.x);
    double radNormed = rad - 2 * M_PI * std::floor((rad + M_PI) / (2 * M_PI));  // normalized to [0, 2*PI]
    return radNormed;
}

std::vector<cv::Point2f> FaceMesh::rotatePoints(std::vector<cv::Point2f> pts, double rad, cv::Point2f rotCenter) {
    double sin = std::sin(rad);
    double cos = std::cos(rad);
    for (auto& p : pts) {
        p -= rotCenter;
        float x = p.x * cos - p.y * sin;
        float y = p.x * sin + p.y * cos;
        p = {x, y};
        p += rotCenter;
    }

    return pts;
}

void FaceMesh::preprocess(const cv::Mat &img) {
    origImageHeight = img.size().height;
    origImageWidth = img.size().width;
    FaceMeshData mdata = metaData->asRef<FaceMeshData>();

    cv::Rect faceRect = mdata.faceRect;
    faceRect = FaceMesh::enlargeFaceRoi(faceRect) & cv::Rect({}, img.size());
    faceRoiWidth = faceRect.width;
    faceRoiHeight = faceRect.height;
    rotationCenter = (faceRect.tl() + faceRect.br()) * 0.5;

    std::vector<cv::Point2f> dstPoints = {
        {0, 0},
        {192, 0},
        {192, 192},
        {0, 192}
    };
    std::vector<cv::Point2f> srcPoints = {
        faceRect.tl(),
        faceRect.tl() + cv::Point{faceRect.width, 0},
        faceRect.br(),
        faceRect.br() - cv::Point{faceRect.width, 0},
    };

    rotationRad = calculateRotationRad(mdata.leftEye, mdata.rightEye);
    srcPoints = rotatePoints(srcPoints, rotationRad, rotationCenter);
    auto lambda = cv::getPerspectiveTransform(srcPoints, dstPoints);
    cv::Mat resizedImage;
    warpPerspective(img, resizedImage, lambda, {inputWidth, inputHeight});
    // cv::imshow("transformed", resizedImage);
    // cv::waitKey(0);
    resizedImage.convertTo(resizedImage, CV_32F);
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);

    resizedImage /= cv::Mat(scales);

    int channelsNum = resizedImage.channels();
    float* inputTensor = interpreter->typed_input_tensor<float>(0);
    float* imgData = resizedImage.ptr<float>();
    for (int32_t i = 0; i < inputWidth * inputHeight; i++) {
        for (int32_t c = 0; c < channelsNum; c++) {
            inputTensor[i * channelsNum + c] = imgData[i * 3 + c];
        }
    }
}

std::unique_ptr<Result> FaceMesh::postprocess() {
    float* landmarksPtr = interpreter->typed_output_tensor<float>(0);
    LandmarksResult* result = new LandmarksResult();
    auto fillLandmarks = [this, landmarksPtr](std::vector<cv::Point>& lms, const std::vector<int>& ids) {
        for (int i : ids) {
            // normalize
            float x = (landmarksPtr[i * 3]) / inputWidth;
            float y = (landmarksPtr[i * 3 + 1]) / inputHeight;
            // rotate
            cv::Point2f p = rotatePoints({{x, y}}, rotationRad, {0.5, 0.5}).front();
            // rotX *= origImageHeight;
            // rotY *= origImageHeight;
            lms.emplace_back(clamp(p.x * faceRoiWidth + rotationCenter.x - faceRoiWidth / 2, 0.f, static_cast<float>(origImageWidth)),
                clamp(p.y * faceRoiHeight + rotationCenter.y - faceRoiHeight / 2, 0.f, static_cast<float>(origImageHeight)));
        }
    };

    // std::sort(faceOvalIdx.begin(), faceOvalIdx.end());
    // std::sort(leftEyeIdx.begin(), leftEyeIdx.end());
    // std::sort(rightEyeIdx.begin(), rightEyeIdx.end());
    // std::sort(leftBrowIdx.begin(), leftBrowIdx.end());
    // std::sort(rightBrowIdx.begin(), rightBrowIdx.end());
    // std::sort(lipsIdx.begin(), lipsIdx.end());

    // std::sort(all.begin(), all.end());
    // std::set<int> s;
    // for (int i = 0; i < 468; ++i) {
    //     s.insert(i);
    // }
    // std::vector<int> res;
    // std::set_difference(s.begin(), s.end(), all.begin(), all.end(), std::back_inserter(res));

    fillLandmarks(result->landmarks.faceOval, faceOvalIdx);
    fillLandmarks(result->landmarks.leftEye, leftEyeIdx);
    fillLandmarks(result->landmarks.rightEye, rightEyeIdx);
    fillLandmarks(result->landmarks.leftBrow, leftBrowIdx);
    fillLandmarks(result->landmarks.rightBrow, rightBrowIdx);
    fillLandmarks(result->landmarks.lips, lipsIdx);
    // fillLandmarks(result->landmarks.left, res);

    return std::unique_ptr<Result>(result);
}
