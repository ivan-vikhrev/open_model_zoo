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
    float confidence;
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
    std::vector<cv::Point> landmarks;
};

struct Face {
    cv::Rect box;
    float confidence;
    std::vector<cv::Point> landmarks;

    Face(cv::Rect box, float conf, std::vector<cv::Point> lm)
        : box(box), confidence(conf), landmarks(lm) {}

    float width() const {
        return box.width;
    }

    float height() const {
        return box.height;
    }
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
    static cv::Mat enlargeFaceRoi(const cv::Mat& img, cv::Rect roi);
protected:
    void checkInputsOutputs() override;
    void preprocess(const cv::Mat &img) override;
    std::unique_ptr<Result> postprocess();

private:
    constexpr static double roiEnlargeCoeff = 1.5;
    const cv::Scalar means = {127.5, 127.5, 127.5};
    const cv::Scalar scales = {127.5, 127.5, 127.5};
};
