// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

# pragma once

#include <utils/common.hpp>

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/tools/gen_op_registration.h>

#include <vector>
#include <string>

struct BBox {
    float left;
    float top;
    float right;
    float bottom;

    cv::Point leftEye;
    cv::Point rightEye;
    cv::Point nose;
    cv::Point mouth;
    cv::Point leftTragion;
    cv::Point rightTragion;

    float confidence;
};

struct FacialLandmarks {
    std::vector<cv::Point> lips;
    std::vector<cv::Point> faceOval;
    std::vector<cv::Point2f> leftEye;
    std::vector<cv::Point> rightEye;
    std::vector<cv::Point> leftBrow;
    std::vector<cv::Point> rightBrow;
};

struct Face {
    cv::Rect box;
    float confidence;
    FacialLandmarks landmarks;

    Face(cv::Rect box, float conf, FacialLandmarks lm)
        : box(box), confidence(conf), landmarks(lm) {}

    float width() const {
        return box.width;
    }

    float height() const {
        return box.height;
    }
};


struct Result {
    virtual ~Result() {}

    template <class T>
    T& asRef() {
        return dynamic_cast<T&>(*this);
    }

    template <class T>
    const T& asRef() const {
        return dynamic_cast<const T&>(*this);
    }
};

struct DetectionResult : Result{
    std::vector<BBox> boxes;
};

struct LandmarksResult : Result {
    FacialLandmarks landmarks;
};


class TFLiteModel {
protected:
    // tflite
    /// // Create model from file. Note that the model instance must outlive the
    /// // interpreter instance.
    int nthreads;

    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    std::vector<int> inputShape;

    int inputWidth;
    int inputHeight;
    int origImageWidth;
    int origImageHeight;

    double imgScale;
    int xPadding;
    int yPadding;

    void readModel(const std::string &modelFile);
    virtual void checkInputsOutputs() = 0;
    void allocateTensors();
    virtual void preprocess(const cv::Mat &img) = 0;
    void infer();
    virtual std::unique_ptr<Result> postprocess() = 0;

public:
    TFLiteModel(const std::string &modelFile);
    virtual ~TFLiteModel() = default;
    void setNumThreads(int nthreads);
    std::unique_ptr<Result> run(const cv::Mat &img);
};

class BlazeFace : public TFLiteModel {
public:
    BlazeFace(const std::string &modelFile,
                  float threshold);

protected:
    void checkInputsOutputs() override;
    void preprocess(const cv::Mat &img) override;
    std::unique_ptr<Result> postprocess();

private:
    int boxesTensorId;
    int scoresTensorId;

    struct {
        int numLayers = 4;
        int numBoxes = 896;
        size_t numPoints = 16;
        float anchorOffsetX = 0.5;
        float anchorOffsetY = 0.5;
        std::vector<int> strides = {8, 16, 16, 16};
        double interpolatedScaleAspectRatio = 1.0;
    } ssdModelOptions;

    double confidenceThreshold;

    cv::Scalar means = {127.5, 127.5, 127.5};
    cv::Scalar scales = {127.5, 127.5, 127.5};

    std::vector<cv::Point2f> anchors;
    void generateAnchors();
    void decodeBoxes(float* boxes);
    std::pair<std::vector<BBox>, std::vector<float>> getDetections(const std::vector<float>& scores, float* boxes);
};

class FaceMesh : public TFLiteModel {
public:
    FaceMesh(const std::string &modelFile);
    static cv::Rect enlargeFaceRoi(cv::Rect roi);
    static cv::Mat calculateRotation(std::vector<cv::Point2f> lm, cv::Point p1, cv::Point p2);

protected:
    void checkInputsOutputs() override;
    void preprocess(const cv::Mat &img) override;
    std::unique_ptr<Result> postprocess();

private:
    std::vector<int> lipsIdx = {

    };

    std::vector<int> faceOvalIdx = {
        10, 338,  297, 332, 284, 251,
        389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377,
        152,148,  176, 149, 150, 136,
        172, 58,  132, 93,  234, 127,
        162, 21,  54,  103, 67,  109
    };

    std::vector<int> leftEyeIdx = {
        33, 7, 163, 144, 145, 153, 154,
        155, 133, 246, 161, 160, 159,
        158, 157, 173,
    };

    std::vector<int> leftBrowIdx = {
        46, 53, 52, 65, 55, 70, 63, 105, 66, 107
    };

    std::vector<int> rightBrowIdx = {
        276, 283, 282, 295, 285, 300, 293, 334, 296, 336
    };

    std::vector<int> rightEyeIdx = {
        263, 249, 390, 373, 374, 380,
        381, 382, 362, 263, 466, 388, 387,
        386, 385, 384, 398, 362,
    };

    constexpr static double roiEnlargeCoeff = 1.5;
    const cv::Scalar means = {127.5, 127.5, 127.5};
    const cv::Scalar scales = {127.5, 127.5, 127.5};
};
