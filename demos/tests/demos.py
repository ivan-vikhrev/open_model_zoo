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
Classes for description tested demo
"""

import sys

from args import ModelArg
from parsers import Parser
from providers import ClassProvider
from utils import create_test_cases


class Demo(ClassProvider):
    __provider_type__ = "demo"

    def __init__(self, name, model_keys=None, device_keys=None, test_cases=None, parser_name="basic"):
        if self.__provider__:
            self.subdirectory = name + "/" + self.__provider__
        else:
            self.subdirectory = name
        if device_keys:
            self.device_keys = ["-" + key for key in device_keys]
        else:
            self.device_keys = ["-d"]
        if model_keys:
            self.model_keys = ["-" + key for key in model_keys]
        else:
            self.model_keys = ["-m"]

        self.test_cases = test_cases
        if Parser.check_provider(parser_name):
            self.parser_name = parser_name
        else:
            raise ValueError(f"There is no parser with name {parser_name}")

        self._exec_name = self.subdirectory.replace("/", "_")
        self.supported_devices = None

    def models_lst_path(self, source_dir):
        return source_dir / self.subdirectory / "models.lst"

    def device_args(self, device_list):
        if len(self.device_keys) == 0:
            return {"CPU": []}
        if self.supported_devices:
            device_list = list(set(device_list) & set(self.supported_devices))
        return {device: [arg for key in self.device_keys for arg in [key, device]] for device in device_list}

    def get_models(self, case):
        return ((case.options[key], key) for key in self.model_keys if key in case.options)

    def update_case(self, case, updated_options, with_replacement=False):
        if not updated_options:
            return
        new_options = case.options.copy()
        for key, value in updated_options.items():
            new_options[key] = value
        new_case = case._replace(options=new_options)
        if with_replacement:
            self.test_cases.remove(case)
        self.test_cases.append(new_case)

    def set_devices(self, devices):
        self.supported_devices = devices

    def set_precisions(self, precisions, model_info):
        for case in self.test_cases[:]:
            updated_options = {p: {} for p in precisions}

            for model, key in self.get_models(case):
                if not isinstance(model, ModelArg):
                    continue
                supported_p = list(set(precisions) & set(model_info[model.name]["precisions"]))
                if len(supported_p):
                    model.precision = supported_p[0]
                    for p in supported_p[1:]:
                        updated_options[p][key] = ModelArg(model.name, p)
                else:
                    print(
                        "Warning: {} model does not support {} precisions and will not be tested\n".format(
                            model.name, ",".join(precisions)
                        )
                    )
                    self.test_cases.remove(case)
                    break

            for p in precisions:
                self.update_case(case, updated_options[p])


class CppDemo(Demo):
    __provider__ = "cpp"

    def __init__(self, name, model_keys=None, device_keys=None, test_cases=None, parser_name="basic"):
        super().__init__(name, model_keys, device_keys, test_cases, parser_name)

        self._exec_name = self._exec_name.replace("_cpp", "")

    def fixed_args(self, source_dir, build_dir):
        return [str(build_dir / self._exec_name)]


class GapiDemo(CppDemo):
    __provider__ = "cpp_gapi"


class PythonDemo(Demo):
    __provider__ = "python"

    def __init__(self, name, model_keys=None, device_keys=None, test_cases=None, parser_name="basic"):
        super().__init__(name, model_keys, device_keys, test_cases, parser_name)

        self._exec_name = self._exec_name.replace("_python", "")

    def fixed_args(self, source_dir, build_dir):
        cpu_extension_path = build_dir / "lib/libcpu_extension.so"

        return [
            sys.executable,
            str(source_dir / self.subdirectory / (self._exec_name + ".py")),
            *(["-l", str(cpu_extension_path)] if cpu_extension_path.exists() else []),
        ]


def create_demos_from_yaml(config):
    list_of_demos = config["demos"]
    demo_classes = []
    for demo_info in list_of_demos:
        name = demo_info["name"]
        parameters = demo_info["parameters"]
        implementation = parameters["implementation"]
        test_cases = create_test_cases(demo_info["cases"], implementation)
        parameters.pop("implementation")
        parameters["test_cases"] = test_cases
        demo_classes.append(Demo.provide(implementation, name, **parameters))

    return demo_classes
