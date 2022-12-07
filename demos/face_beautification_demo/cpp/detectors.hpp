// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

# pragma once

#include <utils/common.hpp>

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

struct BaseDetection {
    ov::InferRequest request;
    std::string pathToModel;
    ov::Shape inShape;
    const bool doRawOutputMessages;

    int nthreads;
    std::string nstreams;
    int nireq;

    BaseDetection(const std::string &pathToModel, bool doRawOutputMessages, int nthreads = 0, std::string nstreams = "", int nireq = 0);
    virtual ~BaseDetection() = default;
    virtual std::shared_ptr<ov::Model> read(const ov::Core& core) = 0;
    bool enabled() const;
};

struct FaceDetection : BaseDetection {
    struct Result {
        int label;
        float confidence;
        cv::Rect location;
    };

    std::string output;
    std::string labels_output;
    double detectionThreshold;
    int objectSize;
    float width;
    float height;
    size_t model_input_width;
    size_t model_input_height;
    float bb_enlarge_coefficient;
    float bb_dx_coefficient;
    float bb_dy_coefficient;

    FaceDetection(const std::string &pathToModel,
                  double detectionThreshold, bool doRawOutputMessages,
                  float bb_enlarge_coefficient, float bb_dx_coefficient,
                  float bb_dy_coefficient,
                  int nthreads = 0,
                  std::string nstreams = "",
                  int nireq = 0);

    std::shared_ptr<ov::Model> read(const ov::Core& core) override;
    void submitRequest(const cv::Mat &frame);
    std::vector<Result> fetchResults();
};



struct FacialLandmarksDetection : BaseDetection {
    size_t enquedFaces;
    std::vector<std::vector<float>> landmarks_results;
    std::vector<cv::Rect> faces_bounding_boxes;

    FacialLandmarksDetection(const std::string &pathToModel,
                             bool doRawOutputMessages);

    std::shared_ptr<ov::Model> read(const ov::Core& core) override;
    void submitRequest();

    void enqueue(const cv::Mat &face);
    std::vector<float> operator[](int idx);
};


struct Load {
    BaseDetection& detector;

    explicit Load(BaseDetection& detector);

    void into(ov::Core& core, const std::string & deviceName) const;
};

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
