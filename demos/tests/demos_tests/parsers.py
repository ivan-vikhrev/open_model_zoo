# Copyright (c) 2022 Intel Corporation
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


"""
Parsers for output od demo processing
"""

import re
from pathlib import Path

from demos_tests.providers import ClassProvider


class Parser(ClassProvider):
    __provider_type__ = "parser"

    def __init__(self, path_to_results=None):
        pass

    def __call__(self, subdirectory, demo_results):
        raise NotImplementedError


class BasicParser(Parser):
    __provider__ = "basic"

    def __call__(self, subdirectory, demo_results):
        return demo_results


class PerformanceParser(Parser):
    __provider__ = "perf"

    def __init__(self, path_to_results=None):
        if path_to_results:
            self.result_dir = Path(path_to_results)
        else:
            self.result_dir = Path("results")
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, subdirectory, demo_results):
        filename = Path(self.result_dir, subdirectory.replace("/", "_") + ".csv")
        # creating or cleaning file which will be used for writing perf results
        filename.open(mode="w").close()
        for output, test_case, device in demo_results:
            result = self.parse_metrics(output)
            self.write_to_csv(filename, result, test_case, device)

    @staticmethod
    def parse_metrics(output):
        def get_metric(name):
            pattern = re.compile(r"{}: (([0-9]+)\.[0-9]+)".format(name))
            metric = pattern.search(" ".join(output.split()))
            return metric.group(1) if metric else "N/A"

        stages_to_parse = ("Latency", "FPS", "Decoding", "Preprocessing", "Inference", "Postprocessing", "Rendering")
        return {name: get_metric(name) for name in stages_to_parse}

    @staticmethod
    def write_to_csv(filename, result, test_case, device):
        result["Requests"] = test_case.options.get("-nireq", "-")
        result["Streams"] = test_case.options.get("-nstreams", "-")
        result["Threads"] = test_case.options.get("-nthreads", "-")

        model_keys = [key for key in test_case.options if key.startswith(("-m", "--m"))]

        if filename.stat().st_size == 0:
            models_col = [f"Model {key}" for key in model_keys]
            precisions_col = [f"Precision {key}" for key in model_keys]
            columns = ",".join(["Device", *precisions_col, *models_col, *result.keys()])
            with open(filename, "w") as f:
                print(columns, file=f)

        precisions = [test_case.options[key].precision for key in model_keys]
        models_names = [test_case.options[key].name for key in model_keys]
        data = ",".join([device, *precisions, *models_names, *result.values()])
        with filename.open(mode="a") as f:
            print(data, file=f)
