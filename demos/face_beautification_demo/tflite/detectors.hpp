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

    cv::Point2f leftEye;
    cv::Point2f rightEye;
    cv::Point nose;
    cv::Point mouth;
    cv::Point leftTragion;
    cv::Point rightTragion;

    float confidence;
};

struct FacialLandmarks {
    std::vector<cv::Point> faceOval;
    std::vector<cv::Point> leftEye;
    std::vector<cv::Point> leftBrow;
    std::vector<cv::Point> rightEye;
    std::vector<cv::Point> rightBrow;
    std::vector<cv::Point> nose;
    std::vector<cv::Point> lips;
    std::vector<cv::Point> left;
};

struct Face {
    cv::Rect box;
    float confidence;
    FacialLandmarks landmarks;

    Face(cv::Rect box, float conf, FacialLandmarks lm)
        : box(box), confidence(conf), landmarks(lm) {}

    std::vector<std::vector<cv::Point>> getFeatures() const {
        return {landmarks.leftBrow, landmarks.leftEye, landmarks.rightBrow,
            landmarks.rightEye, landmarks.nose, landmarks.lips};
    }

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

struct MetaData {
    virtual ~MetaData() {}

    template <class T>
    T& asRef() {
        return dynamic_cast<T&>(*this);
    }

    template <class T>
    const T& asRef() const {
        return dynamic_cast<const T&>(*this);
    }
};

struct FaceMeshData : public MetaData {
    cv::Rect faceRect;
    cv::Point2f leftEye;
    cv::Point2f rightEye;
    FaceMeshData(cv::Rect faceRect, cv::Point leftEye, cv::Point rightEye) :
        faceRect(faceRect), leftEye(leftEye), rightEye(rightEye) {}
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
    std::shared_ptr<MetaData> metaData;

    std::vector<int> inputShape;

    int inputWidth;
    int inputHeight;
    int origImageWidth;
    int origImageHeight;

    double imgScale;
    double xPadding;
    double yPadding;

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
    std::unique_ptr<Result> run(const cv::Mat &img, const std::shared_ptr<MetaData>& metaData = nullptr);
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

protected:
    void checkInputsOutputs() override;
    void preprocess(const cv::Mat &img) override;
    std::unique_ptr<Result> postprocess();

private:
    int faceRoiWidth;
    int faceRoiHeight;
    double rotationRad;
    cv::Point2f rotationCenter;

    cv::Rect enlargeFaceRoi(cv::Rect roi);
    double calculateRotationRad(cv::Point p0, cv::Point p1);
    std::vector<cv::Point2f> rotatePoints(std::vector<cv::Point2f> pts, double rad, cv::Point2f rotCenter);


    std::vector<int> faceOvalIdx = {
        10, 338,  297, 332, 284, 251,
        389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377,
        152,148,  176, 149, 150, 136,
        172, 58,  132, 93,  234, 127,
        162, 21,  54,  103, 67,  109
    };

    std::vector<int> leftEyeIdx = {
        130, 7, 163, 144, 145, 153, 154,
        155, 133, 173, 56, 28, 27, 29, 30, 247
        //246, 161, 160, 159, 158, 157, 173,
    };

    std::vector<int> leftBrowIdx = {
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46
        // 46, 53, 52, 65, 55, 70, 63, 105, 66, 107
    };

    std::vector<int> rightEyeIdx = {
        362, 382, 381, 380, 374, 373, 373, 390, 249,
        359, 467, 260, 259, 257, 258, 286, 414, 463
    };

   std::vector<int> rightBrowIdx = {
        336, 296, 334, 293, 300, 276, 283, 282, 295, 285
    };

    const std::vector<int> noseIdx = {
        2, 99, 240, 235, 219, 218, 237, 44, 19,
        274, 457, 438, 392, 289, 305, 290, 462, 370
    };

    std::vector<int> lipsIdx = {
        61, 146, 91, 181, 84, 17, 314,
        405, 321, 375, 291, 61, 185, 40,
        39, 37, 0, 267, 269, 270, 409,
        291, 78, 95, 88, 178, 87, 14, 317,
        402, 318, 324, 308, 78, 191, 80,
        81, 82, 13, 312, 311, 310, 415, 308
    };

    std::vector<int> left = {
        //1, 2, 3, 4, 5, 6 // vertical line nose
    };

    std::set<int> all = {
        61, 146, 91, 181, 84, 17, 314,
        405, 321, 375, 291, 61, 185, 40,
        39, 37, 0, 267, 269,
        270, 409, 291, 78, 95, 88,
        178, 87, 14, 317, 402, 318,
        324, 308, 78, 191, 80, 81, 82, 13,
        312, 311, 310, 415, 308,
         276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
        263, 249, 390, 373, 374, 380,
        381, 382, 362, 263, 466, 388, 387,
        386, 385, 384, 398, 362,
        46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
        33, 7, 163, 144, 145, 153, 154,
        155, 133, 246, 161, 160, 159,
        158, 157, 173,
        10, 338,  297, 332, 284, 251,
        389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377,
        152,148,  176, 149, 150, 136,
        172, 58,  132, 93,  234, 127,
        162, 21,  54,  103, 67,  109
    };

    constexpr static double roiEnlargeCoeff = 1.5;
    const cv::Scalar means = {127.5, 127.5, 127.5};
    const cv::Scalar scales = {255.0, 255.0, 255.0};
};
