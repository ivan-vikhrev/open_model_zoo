// Copyright (c), 2022, KNS Group LLC (YADRO).
// All Rights Reserved.

// This software contains the intellectual property of YADRO
// or is licensed to YADRO from third parties.  Use of this
// software and the intellectual property contained therein is expressly
// limited to the terms and conditions of the License Agreement under which
// it is provided by YADRO.

#include "filter.hpp"
#include "models.hpp"

#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <vector>
#include <string>


cv::Mat fillPolyContours(int width, int height, const std::vector<std::vector<cv::Point>>& contours) {
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

cv::Mat beautifyFace(cv::Mat img, const Face& face, PerformanceMetrics& m) {
    std::vector<cv::Point> faceOval = face.landmarks.faceOval;
    std::vector<std::vector<cv::Point>> faceFeatures = face.getFeatures();
    auto rect = FaceMesh::enlargeFaceRoi(face.box) & cv::Rect({}, img.size());

    cv::Mat mskSharp = fillPolyContours(img.size().width, img.size().height, faceFeatures );
    // cv::imshow("mask", mskSharp);
    // cv::waitKey(0);
    cv::Mat mskSharpG;
    cv::GaussianBlur(mskSharp, mskSharpG, {5, 5}, 0.0);

    cv::Mat mskBlur = fillPolyContours(img.size().width, img.size().height, {faceOval});
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

    cv::Mat imgBilat(img.clone());

    auto time = std::chrono::steady_clock::now();
    cv::bilateralFilter(img(rect), imgBilat(rect), 9, 30.0, 30.0);
    m.update(time);
    // cv::imshow("filter", imgBilat);

    cv::Mat imgSharp = unsharpMask(img, 3, 0.7f);
    cv::Mat imgBilatMasked = mask3C(imgBilat, mskBlurFinal);
    cv::Mat imgSharpMasked = mask3C(imgSharp, mskSharpG);
    cv::Mat imgInMasked = mask3C(img, mskNoFaces);
    cv::Mat imgBeautif = imgBilatMasked + imgSharpMasked + imgInMasked;

    return imgBeautif;
}

// cv::Mat applyFilter(cv::Mat img, const Face& faces, PerformanceMetrics& m) {
//     std::vector<std::vector<cv::Point>> facesOvals;
//     std::vector<std::vector<cv::Point>> facesFeatures;
//     for (auto&& face : faces) {
//         const auto faceOval = face.landmarks.faceOval;
//         const std::vector<std::vector<cv::Point>> features = face.getFeatures();
//         facesOvals.push_back(face.landmarks.faceOval);
//         facesFeatures.insert(facesFeatures.end(), features.begin(), features.end());
//     }
//     return beautifyFaces(img, facesOvals, facesFeatures, m);
// }
