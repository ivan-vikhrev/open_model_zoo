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
Utils for demo tests
"""

import collections
import itertools
from pathlib import Path
from typing import List, Union

import yaml
from demos_tests.args import AbstractArg, ModelArg

# Functions to read and parsing yaml files


def read_yaml(file: Union[str, Path]):
    with open(file, "r") as content:
        return yaml.safe_load(content)


TestCase = collections.namedtuple("TestCase", ["options", "extra_models"], defaults=([], []))


def join_cases(*args):
    options = {}
    for case in args:
        options.update(case)
    return options


def combine_cases(*args) -> List[dict]:
    return [
        join_cases(*combination)
        for combination in itertools.product(*[[arg] if isinstance(arg, dict) else arg for arg in args])
    ]


def handle_single_option(option_name: str, value) -> dict:
    if value is None:
        return {}
    result_value = value
    if option_name.startswith("model") and isinstance(value, str):
        result_value = ModelArg(value)

    return {option_name: result_value}


def handle_multi_option(key, *args) -> List[dict]:
    list_of_options = []
    for arg in args:
        list_of_options.append(handle_single_option(key, arg))
    return list_of_options


def get_test_options_from_config(options: dict) -> List[dict]:
    if not options or options == "None":
        return [{}]
    general_level_options = {}
    result_options = [{}]
    for key, value in options.items():
        if "split" not in key and not isinstance(value, List):
            general_level_options.update(handle_single_option(key, value))
        else:
            if "split" in key:
                split_args = []
                for elem in value:
                    split_args.extend(get_test_options_from_config(elem))
                result_options = combine_cases(result_options, split_args)
            else:
                result_options = combine_cases(result_options, handle_multi_option(key, *value))
    result_options = combine_cases(result_options, general_level_options)
    return result_options


def create_test_cases(config: dict, implementation) -> List[TestCase]:
    test_options = get_test_options_from_config(config)
    test_cases = []

    for demo_flags in test_options:
        extra_models = demo_flags.get("extra_models", [])
        if isinstance(extra_models, AbstractArg):
            extra_models = extra_models.vector
        demo_flags.pop("extra_models", [])
        case_options = correct_demo_flags(demo_flags, implementation)
        test_case = TestCase(options=case_options, extra_models=extra_models)
        test_cases.append(test_case)

    return test_cases


# Correction demo options according to demo flags

FLAGS_long_short = {"model": "m", "architecture_type": "at", "device": "d"}
FLAGS_long_short_CPP = {"input": "i", "output": "o"}
FLAGS_short_keys_PYTHON = ["nireq", "nstreams", "nthreads", "fg"]


def correct_demo_flags(flags: dict, implementation: str):
    options = {}
    for flag, value in flags.items():
        if flag.startswith(tuple(FLAGS_long_short.keys())):
            for full, short in FLAGS_long_short.items():
                flag = flag.replace(full, short)
            flag = "-" + flag
        else:
            if implementation == "python":
                if flag in FLAGS_short_keys_PYTHON:
                    flag = "-" + flag
                else:
                    flag = "--" + flag
            else:
                if flag.startswith(tuple(FLAGS_long_short_CPP.keys())):
                    for full, short in FLAGS_long_short_CPP.items():
                        flag = flag.replace(full, short)
                flag = "-" + flag
        options[flag] = value
    return options
