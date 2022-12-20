// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

# pragma once
#include <string>
#include <map>
#include <memory>
#include <utility>
#include <list>
#include <vector>
#include <opencv2/opencv.hpp>

#include "detectors.hpp"

// -------------------------Describe detected face on a frame-------------------------------------------------
using Contour   = std::vector<cv::Point>;

// struct Face {
// public:
//     using Ptr = std::shared_ptr<Face>;

//     explicit Face(size_t id, cv::Rect& location);

//     void updateLandmarks(std::vector<float> values);
//     const std::vector<Contour> getFaceContour();
//     const std::vector<Contour> getFaceElemsContours();
//     const std::vector<float>& getLandmarks();
//     size_t getId();

// public:
//     cv::Rect _location;
//     float _intensity_mean;

// private:
//     size_t _id;
//     std::vector<float> _landmarks;
//     Contour cntFace, cntLeftEye, cntRightEye, cntNose, cntMouth;

// };

// ----------------------------------- Utils -----------------------------------------------------------------
// float calcIoU(cv::Rect& src, cv::Rect& dst);
// float calcMean(const cv::Mat& src);
// Face::Ptr matchFace(cv::Rect rect, std::list<Face::Ptr>& faces);
