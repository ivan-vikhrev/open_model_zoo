// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <map>
#include <utility>
#include <list>
#include <vector>

#include "face.hpp"

namespace config {
const     cv::Scalar kClrWhite (255, 255, 255);
const     cv::Scalar kClrGreen (  0, 255,   0);
const     cv::Scalar kClrYellow(  0, 255, 255);

constexpr float      kConfThresh   = 0.7f;

const     cv::Size   kGKernelSize(5, 5);
constexpr double     kGSigma       = 0.0;
constexpr int        kBSize        = 9;
constexpr double     kBSigmaCol    = 30.0;
constexpr double     kBSigmaSp     = 30.0;
constexpr int        kUnshSigma    = 3;
constexpr float      kUnshStrength = 0.7f;
constexpr int        kAngDelta     = 1;
constexpr bool       kClosedLine   = true;
} // namespace config

template<typename Tp> inline int toIntRounded(const Tp x) {
    return static_cast<int>(std::lround(x));
}

template<typename Tp> inline double toDouble(const Tp x) {
    return static_cast<double>(x);
}

inline int getLineInclinationAngleDegrees(const cv::Point &ptLeft, const cv::Point &ptRight) {
    const cv::Point residual = ptRight - ptLeft;
    if (residual.y == 0 && residual.x == 0) {
        return 0;
    }
    else {
        return toIntRounded(atan2(toDouble(residual.y), toDouble(residual.x)) * 180.0 / CV_PI);
    }
}

inline Contour getEyeEllipse(const cv::Point &ptLeft, const cv::Point &ptRight) {
    Contour cntEyeBottom;
    const cv::Point ptEyeCenter((ptRight + ptLeft) / 2);
    const int angle = getLineInclinationAngleDegrees(ptLeft, ptRight);
    const int axisX = toIntRounded(cv::norm(ptRight - ptLeft) / 2.0);
    // According to research, in average a Y axis of an eye is approximately
    //  1/3 of an X one.
    const int axisY = axisX / 3;
    // We need the lower part of an ellipse:
    static constexpr int kAngEyeStart = 0;
    static constexpr int kAngEyeEnd   = 180;
    cv::ellipse2Poly(ptEyeCenter, cv::Size(axisX, axisY), angle, kAngEyeStart, kAngEyeEnd, config::kAngDelta,
                     cntEyeBottom);
    return cntEyeBottom;
}

inline Contour getForeheadEllipse(const cv::Point &ptJawLeft,
                                          const cv::Point &ptJawRight,
                                          const cv::Point &ptJawLower) {
    Contour cntForehead;
    // The point amid the top two points of a jaw:
    const cv::Point ptFaceCenter((ptJawLeft + ptJawRight) / 2);
    // This will be the center of the ellipse.
    // The angle between the jaw and the vertical:
    const int angFace = getLineInclinationAngleDegrees(ptJawLeft, ptJawRight);
    // This will be the inclination of the ellipse
    // Counting the half-axis of the ellipse:
    const double jawWidth  = cv::norm(ptJawLeft - ptJawRight);
    // A forehead width equals the jaw width, and we need a half-axis:
    const int axisX =  toIntRounded(jawWidth / 2.0);
    const double jawHeight = cv::norm(ptFaceCenter - ptJawLower);
    // According to research, in average a forehead is approximately 2/3 of
    //  a jaw:
    const int axisY = toIntRounded(jawHeight * 2 / 3.0);
    // We need the upper part of an ellipse:
    static constexpr int kAngForeheadStart = 180;
    static constexpr int kAngForeheadEnd   = 360;
    cv::ellipse2Poly(ptFaceCenter, cv::Size(axisX, axisY), angFace, kAngForeheadStart, kAngForeheadEnd,
                     config::kAngDelta, cntForehead);
    return cntForehead;
}

inline Contour getPatchedEllipse(const cv::Point &ptLeft,
                                         const cv::Point &ptRight,
                                         const cv::Point &ptUp,
                                         const cv::Point &ptDown) {
    // Shared characteristics for both half-ellipses:
    const cv::Point ptMouthCenter((ptLeft + ptRight) / 2);
    const int angMouth = getLineInclinationAngleDegrees(ptLeft, ptRight);
    const int axisX    = toIntRounded(cv::norm(ptRight - ptLeft) / 2.0);

    // The top half-ellipse:
    Contour cntMouthTop;
    const int axisYTop = toIntRounded(cv::norm(ptMouthCenter - ptUp));
    // We need the upper part of an ellipse:
    static constexpr int angTopStart = 180;
    static constexpr int angTopEnd   = 360;
    cv::ellipse2Poly(ptMouthCenter, cv::Size(axisX, axisYTop), angMouth, angTopStart, angTopEnd, config::kAngDelta, cntMouthTop);

    // The bottom half-ellipse:
    Contour cntMouth;
    const int axisYBot = toIntRounded(cv::norm(ptMouthCenter - ptDown));
    // We need the lower part of an ellipse:
    static constexpr int angBotStart = 0;
    static constexpr int angBotEnd   = 180;
    cv::ellipse2Poly(ptMouthCenter, cv::Size(axisX, axisYBot), angMouth, angBotStart, angBotEnd, config::kAngDelta, cntMouth);

    // Pushing the upper part to vctOut
    std::copy(cntMouthTop.cbegin(), cntMouthTop.cend(), std::back_inserter(cntMouth));
    return cntMouth;
}

Face::Face(size_t id, cv::Rect& location):
    _location(location), _intensity_mean(0.f), _id(id) {
}

// void Face::updateLandmarks(std::vector<float> values) {
//     _landmarks = std::move(values);
// }

void Face::updateLandmarks(std::vector<float> landmarks) {
    _landmarks = landmarks;
    std::vector<cv::Point> normed_landmarks;
    size_t n_lm = _landmarks.size();
    for (size_t i_lm = 0UL; i_lm < n_lm / 2; ++i_lm) {
        float normed_x = _landmarks[2 * i_lm];
        float normed_y = _landmarks[2 * i_lm + 1];

        int x_lm = _location.x + static_cast<int>(_location.width * normed_x);
        int y_lm = _location.y + static_cast<int>(_location.height * normed_y);
        normed_landmarks.emplace_back(x_lm, y_lm);
    }
    static constexpr int kNumFaceElems = 18;
    static constexpr int kNumTotal     = 35;
    // Contour cntLeftEye, cntRightEye, cntNose, cntMouth;
    // A left eye:
    // Approximating the lower eye contour by half-ellipse (using eye points) and storing in cntLeftEye:
    cntLeftEye = getEyeEllipse(normed_landmarks[1], normed_landmarks[0]);
    // Pushing the left eyebrow clock-wise:
    cntLeftEye.insert(cntLeftEye.end(), {normed_landmarks[12], normed_landmarks[13],
                                            normed_landmarks[14]});
    // A right eye:
    // Approximating the lower eye contour by half-ellipse (using eye points) and storing in vctRightEye:
    cntRightEye = getEyeEllipse(normed_landmarks[2], normed_landmarks[3]);
    // Pushing the right eyebrow clock-wise:
    cntRightEye.insert(cntRightEye.end(), {normed_landmarks[15], normed_landmarks[16],
                                            normed_landmarks[17]});
    // A nose:
    // Storing the nose points clock-wise
    cntNose.clear();
    cntNose.insert(cntNose.end(), {normed_landmarks[4], normed_landmarks[7],
                                    normed_landmarks[5], normed_landmarks[6]});
    // A mouth:
    // Approximating the mouth contour by two half-ellipses (using mouth points) and storing in vctMouth:
    cntMouth = getPatchedEllipse(normed_landmarks[8], normed_landmarks[9],
                                    normed_landmarks[10], normed_landmarks[11]);
    // // Storing all the elements in a vector:
    // vctElemsContours.insert(vctElemsContours.end(), {cntLeftEye, cntRightEye, cntNose, cntMouth});

    // The face contour:
    // Approximating the forehead contour by half-ellipse (using jaw points) and storing in vctFace:
    cntFace = getForeheadEllipse(normed_landmarks[kNumFaceElems],normed_landmarks[kNumFaceElems + 16], normed_landmarks[kNumFaceElems + 8]);

    // The ellipse is drawn clock-wise, but jaw contour points goes vice versa, so it's necessary to push
    //  cntJaw from the end to the begin using a reverse iterator:
    std::copy(normed_landmarks.crbegin(), normed_landmarks.crend() - kNumFaceElems, std::back_inserter(cntFace));

}

const std::vector<float>& Face::getLandmarks() {
    return _landmarks;
}

const std::vector<Contour> Face::getFaceContour() {
    return {cntFace};
}

const std::vector<Contour> Face::getFaceElemsContours() {
    return {cntLeftEye, cntRightEye, cntNose, cntMouth};
}

size_t Face::getId() {
    return _id;
}

float calcIoU(cv::Rect& src, cv::Rect& dst) {
    cv::Rect i = src & dst;
    cv::Rect u = src | dst;

    return static_cast<float>(i.area()) / static_cast<float>(u.area());
}

float calcMean(const cv::Mat& src) {
    cv::Mat tmp;
    cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    cv::Scalar mean = cv::mean(tmp);

    return static_cast<float>(mean[0]);
}

Face::Ptr matchFace(cv::Rect rect, std::list<Face::Ptr>& faces) {
    Face::Ptr face(nullptr);
    float maxIoU = 0.55f;
    for (auto&& f : faces) {
        float iou = calcIoU(rect, f->_location);
        if (iou > maxIoU) {
            face = f;
            maxIoU = iou;
        }
    }

    return face;
}
