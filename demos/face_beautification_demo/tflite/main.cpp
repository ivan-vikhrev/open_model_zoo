// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <limits>

#include <gflags/gflags.h>
#include <utils/images_capture.h>
#include <utils/image_utils.h>
#include <utils/slog.hpp>
#include <monitors/presenter.h>

#include <tensorflow/lite/version.h>

#include "detectors.hpp"
#include "face.hpp"
#include "visualizer.hpp"

namespace {
constexpr char h_msg[] = "show the help message and exit";
DEFINE_bool(h, false, h_msg);
\
constexpr char m_msg[] = "path to an .xml file with a trained Face Detection model";
DEFINE_string(m, "", m_msg);

constexpr char i_msg[] = "an input to process. The input must be a single image, a folder of images, video file or camera id. Default is 0";
DEFINE_string(i, "0", i_msg);

constexpr char bb_enlarge_coef_msg[] = "coefficient to enlarge/reduce the size of the bounding box around the detected face. Default is 1.2";
DEFINE_double(bb_enlarge_coef, 1.2, bb_enlarge_coef_msg);

constexpr char d_msg[] =
    "specify a device to infer on (the list of available devices is shown below). "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. "
    "Default is CPU";
DEFINE_string(d, "CPU", d_msg);

constexpr char dx_coef_msg[] = "coefficient to shift the bounding box around the detected face along the Ox axis";
DEFINE_double(dx_coef, 1, dx_coef_msg);

constexpr char dy_coef_msg[] = "coefficient to shift the bounding box around the detected face along the Oy axis";
DEFINE_double(dy_coef, 1, dy_coef_msg);

constexpr char fps_msg[] = "maximum FPS for playing video";
DEFINE_double(fps, -std::numeric_limits<double>::infinity(), fps_msg);

constexpr char lim_msg[] = "number of frames to store in output. If 0 is set, all frames are stored. Default is 1000";
DEFINE_uint32(lim, 1000, lim_msg);

constexpr char loop_msg[] = "enable reading the input in a loop";
DEFINE_bool(loop, false, loop_msg);

constexpr char mlm_msg[] = "path to an .xml file with a trained Facial Landmarks Estimation model";
DEFINE_string(mlm, "", mlm_msg);

constexpr char o_msg[] = "name of the output file(s) to save";
DEFINE_string(o, "", o_msg);

constexpr char r_msg[] = "output inference results as raw values";
DEFINE_bool(r, false, r_msg);

constexpr char show_msg[] = "(don't) show output";
DEFINE_bool(show, true, show_msg);

constexpr char t_msg[] = "probability threshold for detections. Default is 0.5";
DEFINE_double(t, 0.5, t_msg);

constexpr char u_msg[] = "resource utilization graphs. Default is cdm. "
    "c - average CPU load, d - load distribution over cores, m - memory usage, h - hide";
DEFINE_string(u, "cdm", u_msg);

constexpr char num_threads_message[] = "Optional. Specify count of threads.";
DEFINE_int32(nthreads, -1, num_threads_message);

constexpr char num_streams_message[] = "Optional. Specify count of streams.";

DEFINE_string(nstreams, "", num_streams_message);

constexpr char num_inf_req_message[] = "Optional. Number of infer requests.";
DEFINE_uint32(nireq, 0, num_inf_req_message);


void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout <<   "\t[ -h]                                         " << h_msg
                  << "\n\t[--help]                                      print help on all arguments"
                  << "\n\t  -m <MODEL FILE>                             " << m_msg
                  << "\n\t[ -i <INPUT>]                                 " << i_msg
                  << "\n\t[--bb_enlarge_coef <NUMBER>]                  " << bb_enlarge_coef_msg
                  << "\n\t[ -d <DEVICE>]                                " << d_msg
                  << "\n\t[--dx_coef <NUMBER>]                          " << dx_coef_msg
                  << "\n\t[--dy_coef <NUMBER>]                          " << dy_coef_msg
                  << "\n\t[--fps <NUMBER>]                              " << fps_msg
                  << "\n\t[--lim <NUMBER>]                              " << lim_msg
                  << "\n\t[--loop]                                      " << loop_msg
                  << "\n\t[--mlm <MODEL FILE>]                          " << mlm_msg
                  << "\n\t[ -o <OUTPUT>]                                " << o_msg
                  << "\n\t[ -r]                                         " << r_msg
                  << "\n\t[--nthreads <integer>]                            " << num_threads_message
                  << "\n\t[--nstreams] <integer>]                           " << num_streams_message
                  << "\n\t[--nireq <integer>]                               " << num_inf_req_message
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

void renderResults() {

}

} // namespace

int main(int argc, char *argv[]) {
    std::set_terminate(catcher);
    parse(argc, argv);
    PerformanceMetrics metrics, bMetrics, renderMetrics;;

    // --------------------------- 1. Loading Inference Engine -----------------------------


    // FacialLandmarksDetection facialLandmarksDetector(FLAGS_mlm, FLAGS_r);
    // ---------------------------------------------------------------------------------------------------

    // --------------------------- 2. Reading TFlite models and build interpreter ----------------------
    BlazeFace faceDetector(FLAGS_m, FLAGS_t);
    faceDetector.setNumThreads(FLAGS_nthreads);
    slog::info << "\tThreads number:" <<  FLAGS_nthreads << slog::endl;

    FaceMesh facialLandmarksDetector(FLAGS_mlm);
    facialLandmarksDetector.setNumThreads(FLAGS_nthreads);
    slog::info << "\tThreads number: " <<  FLAGS_nthreads << slog::endl;

    // ----------------------------------------------------------------------------------------------------
    // Timer timer;
    // std::ostringstream out;
    // size_t framesCounter = 0;
    // double msrate = 1000.0 / FLAGS_fps;
    // std::list<Face::Ptr> faces;
    // size_t id = 0;
    size_t framesCounter = 0;
    std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);

    // auto startTime = std::chrono::steady_clock::now();
    // cv::Mat frame = cap->read();
    // if (!frame.data) {
    //     throw std::runtime_error("Can't read an image from the input");
    // }

    Presenter presenter(FLAGS_u);

    Visualizer visualizer{{}};

    LazyVideoWriter videoWriter{FLAGS_o, FLAGS_fps > 0.0 ? FLAGS_fps : cap->fps(), FLAGS_lim};

    // // Detecting all faces on the first frame and reading the next one

    // cv::Mat nextFrame = cap->read();
    bool keepRunning = true;
    while (keepRunning) {
        // timer.start("total");
        // const auto startTimePrevFrame = startTime;
        // cv::Mat prevFrame = std::move(frame);
        // startTime = startTimeNextFrame;
        // frame = std::move(nextFrame);
        auto startTime = std::chrono::steady_clock::now();

        cv::Mat frame = cap->read();
        if (frame.empty()) {
            // Input stream is over
            break;
        }
        framesCounter++;

        DetectionResult detectionRes = faceDetector.run(frame)->asRef<DetectionResult>();
        std::vector<Face> faces;
        for (auto& box : detectionRes.boxes) {
            cv::Rect faceRect = cv::Rect(cv::Point{static_cast<int>(box.left), static_cast<int>(box.top)},
                cv::Point{static_cast<int>(box.right), static_cast<int>(box.bottom)});
            // cv::Rect enlargedRect = FaceMesh::enlargeFaceRoi(faceRect) & cv::Rect({}, frame.size());
            // // cv::imshow("Face", frame(enlargedRect));
            // std::vector<cv::Point2f> dstPoints = {
            //     {0, 0},
            //     {192, 0},
            //     {192, 192},
            //     {0, 192}
            // };
            // std::vector<cv::Point2f> srcPoints = {
            //     enlargedRect.tl(),
            //     enlargedRect.tl() + cv::Point{enlargedRect.width, 0},
            //     enlargedRect.br(),
            //     enlargedRect.br() - cv::Point{enlargedRect.width, 0},
            // };

            // cv::Point2f rectCenter = (enlargedRect.br() + enlargedRect.tl()) * 0.5;

            // srcPoints = FaceMesh::calculateRotation(srcPoints, rectCenter, detectionRes.boxes[0].leftEye, detectionRes.boxes[0].rightEye);

            // auto lambda = cv::getPerspectiveTransform(srcPoints, dstPoints);
            // cv::Mat uprightFace;
            // warpPerspective(frame, uprightFace, lambda, {192, 192});
            // cv::imshow("transformed", uprightFace);
            LandmarksResult landmarksRes = facialLandmarksDetector.run(frame,
                std::make_shared<FaceMeshData>(faceRect, box.leftEye, box.rightEye))->asRef<LandmarksResult>();

            // cv::circle(frameLm, detectionRes.boxes[0].leftEye, 1, cv::Scalar(0, 0, 255), -1);
            // cv::circle(frameLm, detectionRes.boxes[0].rightEye, 1, cv::Scalar(0, 0, 255), -1);

            // cv::Mat resizedImage;
            // cv::resize(frameLm, resizedImage, cv::Size(0, 0), 3.0, 3.0, cv::INTER_CUBIC);
            // cv::imshow("lm", resizedImage);

            // lm.leftEye = FaceMesh::calculateRotation(lm.leftEye, rectCenter, detectionRes.boxes[0].leftEye, detectionRes.boxes[0].rightEye); // landmarksRes.landmarks.leftEye

            // auto frameLm = resizeImageExt(frame(enlargedRect), 192, 192,
            //     RESIZE_MODE::RESIZE_FILL, cv::INTER_LINEAR);

            // for (const auto& l : lm.faceOval) {
            //     cv::circle(frameLm, l, 2, cv::Scalar(0, 255, 255), -1);
            // }
            // for (const auto& l : lm.leftEye) {
            //     cv::circle(frameLm, l, 2, cv::Scalar(0, 255, 255), -1);
            // }
            // for (const auto& l : lm.rightEye) {
            //     cv::circle(frameLm, l, 2, cv::Scalar(0, 255, 255), -1);
            // }
            // for (const auto& l : lm.leftBrow) {
            //     cv::circle(frameLm, l, 2, cv::Scalar(0, 255, 255), -1);
            // }
            // for (const auto& l : lm.rightBrow) {
            //     cv::circle(frameLm, l, 2, cv::Scalar(0, 255, 255), -1);
            // }
            // cv::imshow("lm", frameLm);
            // lm.leftBrow = FaceMesh::calculateRotation(lm.leftBrow, rectCenter,detectionRes.boxes[0].leftEye, detectionRes.boxes[0].rightEye);
            // lm.rightEye = FaceMesh::calculateRotation(lm.rightEye, rectCenter, detectionRes.boxes[0].leftEye, detectionRes.boxes[0].rightEye);
            // lm.rightBrow = FaceMesh::calculateRotation(lm.rightBrow, rectCenter,detectionRes.boxes[0].leftEye, detectionRes.boxes[0].rightEye);
            // lm.faceOval = FaceMesh::calculateRotation(lm.faceOval, rectCenter, detectionRes.boxes[0].leftEye, detectionRes.boxes[0].rightEye);
            auto& lm = landmarksRes.landmarks;
            // auto lmToFrameCoordinates = [topLeft = enlargedRect.tl()](std::vector<cv::Point>& lm){
            //     for (auto& l : lm) {
            //         l += topLeft;
            //     }
            // };
            // lmToFrameCoordinates(lm.faceOval);
            // lmToFrameCoordinates(lm.leftEye);
            // lmToFrameCoordinates(lm.rightEye);
            // lmToFrameCoordinates(lm.leftBrow);
            // lmToFrameCoordinates(lm.rightBrow);
            // FacialLandmarks lm = {};
            faces.emplace_back(faceRect, box.confidence, lm);

            // std::cout << rotated << std::endl;
            // for (int i = 0; i < rotated.rows; ++i) {
            //     std::cout << rotated.at<float>(i, 0) << " " <<  rotated.at<float>(i, 1) << std::endl;
            //     cv::circle(frame, 0.5 * (enlargedRect.br() + enlargedRect.tl()) + cv::Point(rotated.at<float>(i, 0) * enlargedRect.width, rotated.at<float>(i, 1) * enlargedRect.width), 3, cv::Scalar(0, 0, 255), -1);
            // }
            // cv::imshow("lm", frame(enlargedRect));
        }
        // presenter.drawGraphs(prevFrame);
        // renderMetrics.update(renderingStart);
        // timer.finish("total");
        // cv::circle(frame, detectionRes.boxes[0].leftEye, 15, cv::Scalar(0, 0, 255), -1);
        // cv::circle(frame, detectionRes.boxes[0].rightEye, 15, cv::Scalar(0, 0, 255), -1);
        // cv::circle(frame, detectionRes.boxes[0].nose, 15, cv::Scalar(0, 0, 255), -1);
        // cv::circle(frame, detectionRes.boxes[0].mouth, 15, cv::Scalar(0, 0, 255), -1);
        // cv::circle(frame, detectionRes.boxes[0].leftTragion, 15, cv::Scalar(0, 0, 255), -1);
        // cv::circle(frame, detectionRes.boxes[0].rightTragion, 15, cv::Scalar(0, 0, 255), -1);

        visualizer.draw(frame, faces);
        presenter.drawGraphs(frame);

        metrics.update(startTime, frame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);

        videoWriter.write(frame);
        // int delay = std::max(1, static_cast<int>(msrate - timer["total"].getLastCallDuration()));
        if (FLAGS_show) {
            cv::imshow(argv[0], frame);
            // cv::imshow("Beautified", beutifiedImg);
            int key = cv::waitKey(1);
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
    // logLatencyPerStage(0.0, 0.0, 0.0, bMetrics.getTotal().latency, renderMetrics.getTotal().latency);
    slog::info << presenter.reportMeans() << slog::endl;

    return 0;
}
