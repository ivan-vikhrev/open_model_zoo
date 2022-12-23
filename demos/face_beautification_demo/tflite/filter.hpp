// Copyright (c), 2022, KNS Group LLC (YADRO).
// All Rights Reserved.

// This software contains the intellectual property of YADRO
// or is licensed to YADRO from third parties.  Use of this
// software and the intellectual property contained therein is expressly
// limited to the terms and conditions of the License Agreement under which
// it is provided by YADRO.

#pragma once

#include "face.hpp"

#include <utils/performance_metrics.hpp>

#include <opencv2/opencv.hpp>

#include <vector>

cv::Mat applyFilter(cv::Mat& img, const std::vector<Face>& faces, PerformanceMetrics& m);

