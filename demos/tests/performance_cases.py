# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from cases import BASE, single_option_cases
from parsers import PerformanceParser

THREADS_NUM = os.cpu_count()


DEMOS = [
    BASE["interactive_face_detection_demo/cpp_gapi"].add_parser(PerformanceParser),
    BASE["interactive_face_detection_demo/cpp"].add_parser(PerformanceParser),
    BASE["object_detection_demo/python"].only_models(["person-detection-0200", "yolo-v2-tf"])
    # TODO: create large -i for performance scenario
    #    .update_option({'-i': DataPatternArg('action-recognition')})
    .add_test_cases(
        single_option_cases("-nireq", "3", "5"),
        single_option_cases("-nstreams", "3", "4"),
        single_option_cases("-nthreads", str(THREADS_NUM), str(THREADS_NUM - 2)),
    ).add_parser(PerformanceParser),
    BASE["bert_named_entity_recognition_demo/python"]
    .update_option({"--dynamic_shape": None})
    .only_devices(["CPU"])
    .add_parser(PerformanceParser),
    BASE["gpt2_text_prediction_demo/python"]
    .update_option({"--dynamic_shape": None})
    .only_devices(["CPU"])
    .add_parser(PerformanceParser),
    BASE["speech_recognition_wav2vec_demo/python"]
    .update_option({"--dynamic_shape": None})
    .only_devices(["CPU"])
    .add_parser(PerformanceParser),
]
