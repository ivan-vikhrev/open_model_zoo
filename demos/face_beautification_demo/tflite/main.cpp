// Copyright (c), 2022, KNS Group LLC (YADRO).
// All Rights Reserved.

// This software contains the intellectual property of YADRO
// or is licensed to YADRO from third parties.  Use of this
// software and the intellectual property contained therein is expressly
// limited to the terms and conditions of the License Agreement under which
// it is provided by YADRO.

#include "face.hpp"
#include "filter.hpp"
#include "models.hpp"

#include <gflags/gflags.h>
#include <utils/images_capture.h>
#include <utils/image_utils.h>
#include <utils/slog.hpp>
#include <monitors/presenter.h>

#include <tensorflow/lite/version.h>
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>

#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
constexpr char h_msg[] = "show the help message and exit";
DEFINE_bool(h, false, h_msg);

constexpr char m_msg[] = "path to an .tflite file with a trained Face Detection model";
DEFINE_string(m, "", m_msg);

constexpr char i_msg[] = "an input to process. The input must be a single image, a folder of images, video file or camera id. Default is 0";
DEFINE_string(i, "0", i_msg);

constexpr char d_msg[] =
    "specify a device to infer on (the list of available devices is shown below). "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. "
    "Default is CPU";
DEFINE_string(d, "CPU", d_msg);

constexpr char lim_msg[] = "number of frames to store in output. If 0 is set, all frames are stored. Default is 1000";
DEFINE_uint32(lim, 1000, lim_msg);

constexpr char loop_msg[] = "enable reading the input in a loop";
DEFINE_bool(loop, false, loop_msg);

constexpr char mlm_msg[] = "path to an .tflite file with a trained Facial Landmarks Estimation model";
DEFINE_string(mlm, "", mlm_msg);

constexpr char o_msg[] = "name of the output file(s) to save";
DEFINE_string(o, "", o_msg);

constexpr char render_msg[] = "enable landmarks rendering";
DEFINE_bool(render, false, render_msg);

constexpr char r_msg[] = "output inference results as raw values";
DEFINE_bool(r, false, r_msg);

constexpr char show_msg[] = "(don't) show output";
DEFINE_bool(show, true, show_msg);

constexpr char t_msg[] = "probability threshold for detections. Default is 0.5";
DEFINE_double(t, 0.5, t_msg);

constexpr char u_msg[] = "resource utilization graphs. Default is cdm. "
    "c - average CPU load, d - load distribution over cores, m - memory usage, h - hide";
DEFINE_string(u, "cdm", u_msg);

constexpr char num_threads_message[] = "specify number of threads.";
DEFINE_int32(nthreads, -1, num_threads_message);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout <<   "\t[ -h]                                         " << h_msg
                  << "\n\t[--help]                                      print help on all arguments"
                  << "\n\t  -m <MODEL FILE>                             " << m_msg
                  << "\n\t[ -i <INPUT>]                                 " << i_msg
                  << "\n\t[ -d <DEVICE>]                                " << d_msg
                  << "\n\t[--lim <NUMBER>]                              " << lim_msg
                  << "\n\t[--loop]                                      " << loop_msg
                  << "\n\t[--mlm <MODEL FILE>]                          " << mlm_msg
                  << "\n\t[ -o <OUTPUT>]                                " << o_msg
                  << "\n\t[ --render]                                   " << render_msg
                  << "\n\t[ -r]                                         " << r_msg
                  << "\n\t[--nthreads <integer>]                        " << num_threads_message
                  << "\n\t[--show] ([--noshow])                         " << show_msg
                  << "\n\t[ -t <NUMBER>]                                " << t_msg
                  << "\n\t[ -u <DEVICE>]                                " << u_msg
                  << "\n\tKey bindings:"
                     "\n\t\tQ, q, Esc - Quit"
                     "\n\t\tP, p, 0, spacebar - Pause"
                     "\n\t\tC - average CPU load, D - load distribution over cores, M - memory usage, H - hide\n";
        showAvailableDevices();
        std::cout << ov::get_openvino_version() << std::endl;
        exit(0);
    } if (FLAGS_i.empty()) {
        throw std::invalid_argument{"-i <INPUT> can't be empty"};
    } if (FLAGS_m.empty()) {
        throw std::invalid_argument{"-m <MODEL FILE> can't be empty"};
    }
    slog::info << "TensorFlow Lite" << slog::endl;
    slog::info << "\tversion " << TFLITE_VERSION_STRING << slog::endl;
}

void  renderResults(cv::Mat img, const std::vector<Face>& faces) {
    for (const auto& face : faces) {
        for (const auto& lm : face.landmarks.getAll()) {
            for (const auto& p : lm) {
                cv::circle(img, p, 2, cv::Scalar(110, 193, 225), -1);
            }
            cv::polylines(img, lm, true, {192, 192, 192});
        }
    }
}

} // namespace

int main(int argc, char *argv[]) {
    std::set_terminate(catcher);
    parse(argc, argv);
    PerformanceMetrics metrics, fdMetrics, lmMetrics, filterMetrics, bilatMetrics, renderMetrics;

    BlazeFace faceDetector(FLAGS_m, FLAGS_t, FLAGS_nthreads);

    FaceMesh facialLandmarksDetector(FLAGS_mlm, FLAGS_nthreads);

    size_t framesCounter = 0;
    std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);

    Presenter presenter(FLAGS_u);

    LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_lim};

    bool showOriginal = false;
    bool renderLandmarks = FLAGS_render;
    bool keepRunning = true;
    while (keepRunning) {
        auto startTime = std::chrono::steady_clock::now();
        cv::Mat frame = cap->read();
        if (frame.empty()) {
            // Input stream is over
            break;
        }
        framesCounter++;

        auto fdStart = std::chrono::steady_clock::now();
        DetectionResult detectionRes = faceDetector.run(frame)->asRef<DetectionResult>();
        fdMetrics.update(fdStart);

        std::vector<Face> faces;
        for (auto& box : detectionRes.boxes) {
            cv::Rect faceRect = cv::Rect(cv::Point{static_cast<int>(box.left), static_cast<int>(box.top)},
                cv::Point{static_cast<int>(box.right), static_cast<int>(box.bottom)});
            auto lmStart = std::chrono::steady_clock::now();
            LandmarksResult landmarksRes = facialLandmarksDetector.run(frame,
                std::make_shared<FaceMeshData>(faceRect, box.leftEye, box.rightEye))->asRef<LandmarksResult>();
            lmMetrics.update(lmStart);
            auto& lm = landmarksRes.landmarks;
            faces.emplace_back(faceRect, box.confidence, lm);
            auto filterStart = std::chrono::steady_clock::now();
            auto res = beautifyFace(frame, faces.back(), bilatMetrics);
            if (!showOriginal) {
                frame = res;
            }
            filterMetrics.update(filterStart);
        }


        auto renderingStart = std::chrono::steady_clock::now();
        if (renderLandmarks) {
            renderResults(frame, faces);
        }
        presenter.drawGraphs(frame);
        renderMetrics.update(renderingStart);

        metrics.update(startTime, frame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);

        videoWriter.write(frame);
        if (FLAGS_show) {
            cv::imshow(argv[0], frame);
            // cv::imshow("Beautified", beautifiedImg);
            int key = cv::waitKey(1);
            if ('L' == key || 'l' == key) {
                renderLandmarks = !renderLandmarks;
            }
            if ('O' == key || 'o' == key) {
                showOriginal = !showOriginal;
            }
            if ('P' == key || 'p' == key || '0' == key || ' ' == key) {
                key = cv::waitKey(0);
            }
            if (27 == key || 'Q' == key || 'q' == key) {
                break;
            }
            presenter.handleKey(key);
        }
    }

    slog::info << "Metrics report:" << slog::endl;
    metrics.logTotal();
    slog::info << "Stages: " << slog::endl;
    slog::info << "\tDecoding:\t" << std::fixed << std::setprecision(1) << cap->getMetrics().getTotal().latency << " ms" << slog::endl;
    slog::info << "\tFace detection:\t" << fdMetrics.getTotal().latency << " ms" << slog::endl;
    slog::info << "\tLandmarks detection:\t" << lmMetrics.getTotal().latency << " ms" << slog::endl;
    slog::info << "\tFilters:\t" << filterMetrics.getTotal().latency << " ms" << slog::endl;
    slog::info << "\t\tBilaterial filter:\t" << bilatMetrics.getTotal().latency << " ms" << slog::endl;
    slog::info << "\tRendering:\t" << renderMetrics.getTotal().latency << " ms" << slog::endl;
    slog::info << presenter.reportMeans() << slog::endl;

    return 0;
}
