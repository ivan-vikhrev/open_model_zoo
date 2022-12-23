// Copyright (c), 2022, KNS Group LLC (YADRO).
// All Rights Reserved.

// This software contains the intellectual property of YADRO
// or is licensed to YADRO from third parties.  Use of this
// software and the intellectual property contained therein is expressly
// limited to the terms and conditions of the License Agreement under which
// it is provided by YADRO.

# pragma once

#include <opencv2/opencv.hpp>

#include <vector>

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
    std::vector<cv::Point> leftBrow;
    std::vector<cv::Point> leftEye;
    std::vector<cv::Point> rightBrow;
    std::vector<cv::Point> rightEye;
    std::vector<cv::Point> nose;
    std::vector<cv::Point> lips;

    std::vector<std::vector<cv::Point>> getAll() const {
        return {faceOval, leftBrow, leftEye,
            rightBrow, rightEye, nose, lips};
    }
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
