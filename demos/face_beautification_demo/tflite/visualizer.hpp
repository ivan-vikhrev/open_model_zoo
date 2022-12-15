// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "face.hpp"

// -------------------------Generic routines for visualization of detection results-------------------------------------------------

// Drawing a photo frame around detected face
class PhotoFrameVisualizer {
public:
    using Ptr = std::shared_ptr<PhotoFrameVisualizer>;

    explicit PhotoFrameVisualizer(int bbThickness = 1, int photoFrameThickness = 2, float photoFrameLength = 0.1);

    void draw(cv::Mat& img, cv::Rect& bb, cv::Scalar color);

private:
    int bbThickness;
    int photoFrameThickness;
    float photoFrameLength;
};


// Drawing detected faces on the frame
class Visualizer {
public:
    enum AnchorType {
        TL = 0,
        TR,
        BL,
        BR
    };

    struct DrawParams {
        cv::Point cell;
        AnchorType barAnchor;
        AnchorType rectAnchor;
        size_t frameIdx;
    };

    explicit Visualizer(cv::Size const& imgSize);

    cv::Mat beautify(cv::Mat img, std::list<Face::Ptr> faces, PerformanceMetrics& m);
    void draw(cv::Mat img, std::list<Face::Ptr> faces);

private:
    void drawFace(cv::Mat& img, Face::Ptr f, bool drawContours = true);
    cv::Mat beautifyFaces(cv::Mat& img, const std::vector<Contour>& facesContours,
        const std::vector<Contour>& facesElemsContours, PerformanceMetrics& m);
    PhotoFrameVisualizer::Ptr photoFrameVisualizer;

    cv::Size imgSize;
    size_t frameCounter;
};
