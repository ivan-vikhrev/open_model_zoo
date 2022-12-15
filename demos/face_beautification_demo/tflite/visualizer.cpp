// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <algorithm>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>

#include "visualizer.hpp"

// PhotoFrameVisualizer
PhotoFrameVisualizer::PhotoFrameVisualizer(int bbThickness, int photoFrameThickness, float photoFrameLength):
    bbThickness(bbThickness), photoFrameThickness(photoFrameThickness), photoFrameLength(photoFrameLength) {
}

void PhotoFrameVisualizer::draw(cv::Mat& img, cv::Rect& bb, cv::Scalar color) {
    cv::rectangle(img, bb, color, bbThickness);

    auto drawPhotoFrameCorner = [&](cv::Point p, int dx, int dy) {
        cv::line(img, p, cv::Point(p.x, p.y + dy), color, photoFrameThickness);
        cv::line(img, p, cv::Point(p.x + dx, p.y), color, photoFrameThickness);
    };

    int dx = static_cast<int>(photoFrameLength * bb.width);
    int dy = static_cast<int>(photoFrameLength * bb.height);

    drawPhotoFrameCorner(bb.tl(), dx, dy);
    drawPhotoFrameCorner(cv::Point(bb.x + bb.width - 1, bb.y), -dx, dy);
    drawPhotoFrameCorner(cv::Point(bb.x, bb.y + bb.height - 1), dx, -dy);
    drawPhotoFrameCorner(cv::Point(bb.x + bb.width - 1, bb.y + bb.height - 1), -dx, -dy);
}

// Visualizer
Visualizer::Visualizer(cv::Size const& imgSize):
        photoFrameVisualizer(std::make_shared<PhotoFrameVisualizer>()), imgSize(imgSize), frameCounter(0) {}


void Visualizer::drawFace(cv::Mat& img, Face::Ptr f, bool drawContours) {

    if (drawContours) {
        const auto faceContours = f->getFaceContour();
        const auto faceElemsContours = f->getFaceElemsContours();
        cv::polylines(img, faceContours, true,{47,173,255});
        cv::polylines(img, faceElemsContours, true, {47,173,255});
    }

    const auto color = cv::Scalar(192, 192, 192);

    std::ostringstream out;

    auto textPos = cv::Point2f(static_cast<float>(f->_location.x), static_cast<float>(f->_location.y - 20));
    putHighlightedText(img, out.str(), textPos, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, color, 2);

    auto& normed_landmarks = f->getLandmarks();
    size_t n_lm = normed_landmarks.size();
    for (size_t i_lm = 0UL; i_lm < n_lm / 2; ++i_lm) {
        float normed_x = normed_landmarks[2 * i_lm];
        float normed_y = normed_landmarks[2 * i_lm + 1];

        int x_lm = f->_location.x + static_cast<int>(f->_location.width * normed_x);
        int y_lm = f->_location.y + static_cast<int>(f->_location.height * normed_y);
        cv::circle(img, cv::Point(x_lm, y_lm), 1 + static_cast<int>(0.012 * f->_location.width), cv::Scalar(0, 255, 255), -1);
    }


    photoFrameVisualizer->draw(img, f->_location, color);
}


cv::Mat fillPolyContours(int width, int height, const std::vector<Contour>& contours) {
    cv::Mat out{height, width, CV_8UC1, cv::Scalar(0)};
    cv::fillPoly(out, contours, {255, 255, 255});
    // out.convertTo(out, CV_8U);
    return out;
}

inline cv::Mat unsharpMask(const cv::Mat &src, const int sigma, const float strength) {
    cv::Mat blurred;
    cv::medianBlur(src, blurred, sigma);
    cv::Mat laplacian;
    cv::Laplacian(blurred, laplacian, CV_8U);

    return (src - (laplacian * strength));
}

inline cv::Mat mask3C(const cv::Mat &src, const cv::Mat &mask) {
    cv::Mat bgr[3];
    cv::split(src, bgr);
    cv::Mat masked0;
    bgr[0].copyTo(masked0, mask);
    cv::Mat masked1;
    bgr[1].copyTo(masked1, mask);
    cv::Mat masked2;
    bgr[2].copyTo(masked2, mask);
    cv::Mat res;
    std::vector<cv::Mat> channels{masked0, masked1, masked2};
    cv::merge(channels, res);
    return res;
}

cv::Mat Visualizer::beautifyFaces(cv::Mat& img, const std::vector<Contour>& facesContours,
    const std::vector<Contour>& facesElemsContours, PerformanceMetrics& m) {

    cv::Mat mskSharp = fillPolyContours(img.size().width, img.size().height, facesElemsContours);
    // cv::imshow("mask", mskSharp);
    // cv::waitKey(0);
    cv::Mat mskSharpG;
    cv::GaussianBlur(mskSharp, mskSharpG, {5, 5}, 0.0);

    cv::Mat mskBlur = fillPolyContours(img.size().width, img.size().height, facesContours);
    cv::Mat mskBlurG;
    cv::GaussianBlur(mskBlur, mskBlurG, {5, 5}, 0.0);

    cv::Mat mask;
    mskBlurG.copyTo(mask, mskSharpG);
    cv::Mat mskBlurFinal = mskBlurG - mask;
    cv::Mat mskFacesGaussed = mskBlurFinal + mskSharpG;

    cv::Mat mskFacesWhite;
    cv::threshold(mskFacesGaussed, mskFacesWhite, 0, 255, cv::THRESH_BINARY);
    cv::Mat mskNoFaces;
    cv::bitwise_not(mskFacesWhite, mskNoFaces);

    auto startTime = std::chrono::steady_clock::now();
    cv::Mat imgBilat;
    cv::bilateralFilter(img, imgBilat, 9, 30.0, 30.0);
    m.update(startTime);
    cv::Mat imgSharp = unsharpMask(img, 3, 0.7f);
    cv::Mat imgBilatMasked = mask3C(imgBilat, mskBlurFinal);
    cv::Mat imgSharpMasked = mask3C(imgSharp, mskSharpG);
    cv::Mat imgInMasked = mask3C(img, mskNoFaces);
    cv::Mat imgBeautif = imgBilatMasked + imgSharpMasked + imgInMasked;
    return imgBeautif;
}

cv::Mat Visualizer::beautify(cv::Mat img, std::list<Face::Ptr> faces, PerformanceMetrics& m) {
    frameCounter++;

    std::vector<Contour> facesContours;
    std::vector<Contour> facesElemsContours;
    for (auto&& face : faces) {
        const auto oneFaceContours = face->getFaceContour();
        const auto oneFaceElems = face->getFaceElemsContours();
        facesContours.insert(facesContours.end(), oneFaceContours.begin(), oneFaceContours.end());
        facesElemsContours.insert(facesElemsContours.end(), oneFaceElems.begin(), oneFaceElems.end());
    }
    return beautifyFaces(img, facesContours, facesElemsContours, m);
}

void Visualizer::draw(cv::Mat img, std::list<Face::Ptr> faces) {
    frameCounter++;

    std::vector<Face::Ptr> newFaces;
    for (auto&& face : faces) {
        drawFace(img, face);
    }
}
