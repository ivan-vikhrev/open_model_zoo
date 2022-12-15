// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detectors.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include <utils/config_factory.h>
#include <utils/ocv_common.hpp>
#include <map>

namespace {
constexpr size_t ndetections = 200;
constexpr size_t batchSize = 1;

struct ssdOptionShort {
    int numLayers = 4;
    int inputSizeHeight = 128;
    int inputSizeWidth = 128;
    float anchorOffsetX = 0.5;
    float anchorOffsetY = 0.5;
    std::vector<int> strides = {8, 16, 16, 16};
    double interpolatedScaleAspectRatio = 1.0;
};
}  // namespace


BaseDetection::BaseDetection(const std::string &pathToModel, bool doRawOutputMessages, int nthreads, std::string nstreams, int nireq)
    : pathToModel(pathToModel), doRawOutputMessages(doRawOutputMessages), nthreads(nthreads), nstreams(nstreams), nireq(nireq) {
}

bool BaseDetection::enabled() const  {
    return bool(request);
}

FaceDetection::FaceDetection(const std::string &pathToModel,
                             double detectionThreshold, bool doRawOutputMessages,
                             float bb_enlarge_coefficient, float bb_dx_coefficient, float bb_dy_coefficient,
                             int nthreads,std::string nstreams, int nireq)
    : BaseDetection(pathToModel, doRawOutputMessages, nthreads, nstreams, nireq),
      detectionThreshold(detectionThreshold),
      objectSize(0), width(0), height(0),
      model_input_width(0), model_input_height(0),
      bb_enlarge_coefficient(bb_enlarge_coefficient), bb_dx_coefficient(bb_dx_coefficient),
      bb_dy_coefficient(bb_dy_coefficient) {}

// void FaceDetection::submitRequest(const cv::Mat& frame) {
//     width = static_cast<float>(frame.cols);
//     height = static_cast<float>(frame.rows);
//     resize2tensor(frame, request.get_input_tensor());
//     request.start_async();
// }

std::unique_ptr<tflite::FlatBufferModel> FaceDetection::read() {
    slog::info << "Reading model: " << pathToModel << slog::endl;
    model = tflite::FlatBufferModel::BuildFromFile(pathToModel.c_str());
    if (!model) {
        throw std::runtime_error("Failed to read model " + pathToModel);
    }
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    // std::string inputName = interpreter->GetInputName(0);
    // slog::info << "Inputs:" << slog::endl;
    // slog::info << "\t" << inputName << slog::endl;
    // // interpreter->AllocateTensors();
    // std::string outputNameScores = interpreter->GetOutputName(0);
    // std::string outputNameBoxes = interpreter->GetOutputName(1);
    // slog::info << "Outputs:" << slog::endl;
    // slog::info << "\t" << outputNameScores << slog::endl;
    // slog::info << "\t" << outputNameBoxes << slog::endl;
}

// std::vector<FaceDetection::Result> FaceDetection::fetchResults() {
//     std::vector<FaceDetection::Result> results;
//     request.wait();
//     float *detections = request.get_tensor(output).data<float>();
//     if (!labels_output.empty()) {
//         const int32_t *labels = request.get_tensor(labels_output).data<int32_t>();
//         for (size_t i = 0; i < ndetections; i++) {
//             Result r;
//             r.label = labels[i];
//             r.confidence = detections[i * objectSize + 4];

//             if (r.confidence <= detectionThreshold && !doRawOutputMessages) {
//                 continue;
//             }

//             r.location.x = static_cast<int>(detections[i * objectSize] / model_input_width * width);
//             r.location.y = static_cast<int>(detections[i * objectSize + 1] / model_input_height * height);
//             r.location.width = static_cast<int>(detections[i * objectSize + 2] / model_input_width * width - r.location.x);
//             r.location.height = static_cast<int>(detections[i * objectSize + 3] / model_input_height * height - r.location.y);

//             // Make square and enlarge face bounding box for more robust operation of face analytics networks
//             int bb_width = r.location.width;
//             int bb_height = r.location.height;

//             int bb_center_x = r.location.x + bb_width / 2;
//             int bb_center_y = r.location.y + bb_height / 2;

//             int max_of_sizes = std::max(bb_width, bb_height);

//             int bb_new_width = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);
//             int bb_new_height = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);

//             r.location.x = bb_center_x - static_cast<int>(std::floor(bb_dx_coefficient * bb_new_width / 2));
//             r.location.y = bb_center_y - static_cast<int>(std::floor(bb_dy_coefficient * bb_new_height / 2));

//             r.location.width = bb_new_width;
//             r.location.height = bb_new_height;

//             if (doRawOutputMessages) {
//                 slog::debug << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
//                              "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
//                           << r.location.height << ")"
//                           << ((r.confidence > detectionThreshold) ? " WILL BE RENDERED!" : "") << slog::endl;
//             }
//             if (r.confidence > detectionThreshold) {
//                 results.push_back(r);
//             }
//         }
//     }

//     for (size_t i = 0; i < ndetections; i++) {
//         float image_id = detections[i * objectSize];
//         if (image_id < 0) {
//             break;
//         }
//         Result r;
//         r.label = static_cast<int>(detections[i * objectSize + 1]);
//         r.confidence = detections[i * objectSize + 2];

//         if (r.confidence <= detectionThreshold && !doRawOutputMessages) {
//             continue;
//         }

//         r.location.x = static_cast<int>(detections[i * objectSize + 3] * width);
//         r.location.y = static_cast<int>(detections[i * objectSize + 4] * height);
//         r.location.width = static_cast<int>(detections[i * objectSize + 5] * width - r.location.x);
//         r.location.height = static_cast<int>(detections[i * objectSize + 6] * height - r.location.y);

//         // Make square and enlarge face bounding box for more robust operation of face analytics networks
//         int bb_width = r.location.width;
//         int bb_height = r.location.height;

//         int bb_center_x = r.location.x + bb_width / 2;
//         int bb_center_y = r.location.y + bb_height / 2;

//         int max_of_sizes = std::max(bb_width, bb_height);

//         int bb_new_width = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);
//         int bb_new_height = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);

//         r.location.x = bb_center_x - static_cast<int>(std::floor(bb_dx_coefficient * bb_new_width / 2));
//         r.location.y = bb_center_y - static_cast<int>(std::floor(bb_dy_coefficient * bb_new_height / 2));

//         r.location.width = bb_new_width;
//         r.location.height = bb_new_height;

//         if (doRawOutputMessages) {
//             slog::debug << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
//                          "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
//                       << r.location.height << ")"
//                       << ((r.confidence > detectionThreshold) ? " WILL BE RENDERED!" : "") << slog::endl;
//         }
//         if (r.confidence > detectionThreshold) {
//             results.push_back(r);
//         }
//     }
//     return results;
// }

// FacialLandmarksDetection::FacialLandmarksDetection(const std::string &pathToModel,
//                                                    bool doRawOutputMessages)
//     : BaseDetection(pathToModel, doRawOutputMessages), enquedFaces(0) {
// }

// void FacialLandmarksDetection::submitRequest() {
//     if (!enquedFaces) return;
//     request.set_input_tensor(ov::Tensor{request.get_input_tensor(), {0, 0, 0, 0}, {batchSize, inShape[1], inShape[2], inShape[3]}});
//     request.start_async();
//     enquedFaces = 0;
// }

// void FacialLandmarksDetection::enqueue(const cv::Mat &face) {
//     if (!enabled()) {
//         return;
//     }
//     ov::Tensor batch = request.get_input_tensor();
//     batch.set_shape(inShape);
//     resize2tensor(face, ov::Tensor{batch, {enquedFaces, 0, 0, 0}, {enquedFaces + 1, inShape[1], inShape[2], inShape[3]}});
//     enquedFaces++;
// }

// std::vector<float> FacialLandmarksDetection::operator[](int idx) {
//     std::vector<float> normedLandmarks;

//     request.wait();
//     const ov::Tensor& tensor = request.get_output_tensor();
//     size_t n_lm = tensor.get_shape().at(1);
//     const float *normed_coordinates = request.get_output_tensor().data<float>();

//     if (doRawOutputMessages) {
//         slog::debug << "[" << idx << "] element, normed facial landmarks coordinates (x, y):" << slog::endl;
//     }

//     auto begin = n_lm / 2 * idx;
//     auto end = begin + n_lm / 2;
//     for (auto i_lm = begin; i_lm < end; ++i_lm) {
//         float normed_x = normed_coordinates[2 * i_lm];
//         float normed_y = normed_coordinates[2 * i_lm + 1];

//         if (doRawOutputMessages) {
//             slog::debug <<'\t' << normed_x << ", " << normed_y << slog::endl;
//         }

//         normedLandmarks.push_back(normed_x);
//         normedLandmarks.push_back(normed_y);
//     }

//     return normedLandmarks;
// }

// std::shared_ptr<ov::Model> FacialLandmarksDetection::read(const ov::Core& core) {
//     slog::info << "Reading model: " << pathToModel << slog::endl;
//     std::shared_ptr<ov::Model> model = core.read_model(pathToModel);
//     logBasicModelInfo(model);

//     ov::Shape outShape = model->output().get_shape();
//     if (outShape.size() != 2 && outShape.back() != 70) {
//         throw std::logic_error("Facial Landmarks Estimation network output layer should have 2 dimensions and 70 as"
//                                " the last dimension");
//     }
//     ov::preprocess::PrePostProcessor ppp(model);
//     ppp.input().tensor().
//         set_element_type(ov::element::u8).
//         set_layout("NHWC");
//     ppp.input().preprocess().convert_layout("NCHW");
//     ppp.output().tensor().set_element_type(ov::element::f32);
//     model = ppp.build();
//     inShape = model->input().get_shape();
//     inShape[0] = batchSize;
//     ov::set_batch(model, batchSize); //{1, int64_t(ndetections)});
//     return model;
// }


// Load::Load(BaseDetection& detector) : detector(detector) {
// }

// void Load::into(ov::Core& core, const std::string & deviceName) const {

//     interpreter->SetNumThreads(8);

//     // if (!detector.pathToModel.empty()) {
//     //     auto config = ConfigFactory::getUserConfig("CPU", detector.nireq, detector.nstreams, detector.nthreads);
//     //     ov::CompiledModel cml = core.compile_model(detector.read(core), config.deviceName, config.compiledModelConfig);
//     //     logCompiledModelInfo(cml, detector.pathToModel, deviceName);
//     //     detector.request = cml.create_infer_request();
//     // }
// }


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
