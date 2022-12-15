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

struct FaceBox {
    float left;
    float top;
    float right;
    float bottom;

    float confidence;

    float getWidth() const {
        return (right - left) + 1.0f;
    }
    float getHeight() const {
        return (bottom - top) + 1.0f;
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

    void readModel(const std::string &modelFile);
    virtual void checkInputsOutputs() = 0;
    void allocateTensors();
    virtual void preprocess(const cv::Mat &img) = 0;
    virtual void infer() = 0;
    // virtual std::vector<Result> postprocess() = 0;

public:
    TFLiteModel(const std::string &modelFile);
    virtual ~TFLiteModel() = default;
    void setNumThreads(int nthreads);
    // virtual std::vector<Result> run(const cv::Mat &img) = 0;
};

class BlazeFace : TFLiteModel {
public:
    BlazeFace(const std::string &modelFile,
                  float threshold);
    std::vector<FaceBox> run(const cv::Mat &img);

protected:
    void checkInputsOutputs() override;
    void preprocess(const cv::Mat &img) override;
    void infer() override;
    std::vector<FaceBox> postprocess();

private:
    struct {
        int numLayers = 4;
        int numBoxes = 896;
        size_t numPoints = 16;
        int inputHeight = 128;
        int inputWidth = 128;
        float anchorOffsetX = 0.5;
        float anchorOffsetY = 0.5;
        std::vector<int> strides = {8, 16, 16, 16};
        double interpolatedScaleAspectRatio = 1.0;
        cv::Scalar means = {127.5, 127.5, 127.5};
        cv::Scalar scales = {127.5, 127.5, 127.5};
    } ssdModelOptions;

    double confidenceThreshold;

    double imgScale;
    int origImageWidth;
    int origImageHeight;

    int boxesTensorId;
    int scoresTensorId;

    int xPadding;
    int yPadding;

    std::vector<cv::Point2f> anchors;
    void generateAnchors();
    void decodeBoxes(float* boxes);
    std::pair<std::vector<FaceBox>, std::vector<float>> getDetections(const std::vector<float>& scores, float* boxes);
};



// struct FacialLandmarksDetection : BaseDetection {
//     size_t enquedFaces;
//     std::vector<std::vector<float>> landmarks_results;
//     std::vector<cv::Rect> faces_bounding_boxes;

//     FacialLandmarksDetection(const std::string &pathToModel,
//                              bool doRawOutputMessages);

//     void read() override;
//     // void submitRequest();

//     // void enqueue(const cv::Mat &face);
//     // std::vector<float> operator[](int idx);
// };


// struct Load {
//     BaseDetection& detector;

//     explicit Load(BaseDetection& detector);

//     void into(ov::Core& core, const std::string & deviceName) const;
// };

class CallStat {
public:
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

    CallStat();

    double getSmoothedDuration();
    double getTotalDuration();
    double getLastCallDuration();
    void calculateDuration();
    void setStartTime();

private:
    size_t _number_of_calls;
    double _total_duration;
    double _last_call_duration;
    double _smoothed_duration;
    std::chrono::time_point<std::chrono::steady_clock> _last_call_start;
};

class Timer {
public:
    void start(const std::string& name);
    void finish(const std::string& name);
    CallStat& operator[](const std::string& name);

private:
    std::map<std::string, CallStat> _timers;
};
