"""
Copyright (c) 2022 Intel Corporation
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import collections
from collections.abc import Sequence
import itertools
from typing import List, Union
from pathlib import Path
import yaml

from args import DataDirectoryOrigFileNamesArg, ModelArg, ModelFileArg, OMZ_DIR, TestDataArg


# Constructions for registering classes (one of the variants of factories)

class UnregisteredProviderException(ValueError):
    def __init__(self, provider, root_provider):
        self.provider = provider
        self.root_provider = root_provider
        self.message = 'Requested provider {} not registered for {}'.format(provider, root_provider)
        super().__init__(self.message)

class BaseProvider:
    providers = {}
    __provider_type__ = None
    __provider__ = None

    @classmethod
    def provide(cls, provider, *args, **kwargs):
        root_provider = cls.resolve(provider)
        return root_provider(*args, **kwargs)

    @classmethod
    def resolve(cls, name):
        if name not in cls.providers:
            raise UnregisteredProviderException(name, cls.__provider_type__)
        return cls.providers[name]


class ClassProviderMeta(type):
    def __new__(mcs, name, bases, attrs, **kwargs):
        cls = type.__new__(mcs, name, bases, attrs)
        # do not create container for abstract provider
        if '_is_base_provider' in attrs:
            return cls

        if '__provider_type__' in attrs:
            cls.providers = {}
        else:
            cls.register_provider(cls)

        return cls


class ClassProvider(BaseProvider, metaclass=ClassProviderMeta):
    _is_base_provider = True

    @classmethod
    def get_provider_name(cls):
        return getattr(cls, '__provider__', cls.__name__)

    @classmethod
    def register_provider(cls, provider):
        provider_name = cls.get_provider_name()
        if not provider_name:
            return
        cls.providers[provider_name] = provider


# Functions to read and parsing yaml files

def read_yaml(file: Union[str, Path]):
    with open(file, 'r') as content:
        return yaml.safe_load(content)

TestCase = collections.namedtuple('TestCase', ['options', 'extra_models'], defaults = ([], []))


def join_cases(*args):
    options = {}
    for case in args: options.update(case)
    # extra_models = set()
    # for case in args: extra_models.update(case.extra_models)
    return options # TestCase(options=options, extra_models=list(case.extra_models))


def combine_cases(*args) -> List[dict]:
    return [join_cases(*combination)
        for combination in itertools.product(*[[arg] if isinstance(arg, dict) else arg for arg in args])]


def handle_single_option(option_name, value) -> dict:
    if value is None or value == 'None':
        return {}
    if isinstance(value, dict):
        return {option_name: ModelFileArg(**value)}
    result_value = value
    if 'model' in option_name:
        result_value = ModelArg(value)
    if option_name == 'input':
        if '.' in value:
            result_value = TestDataArg(value)
        else:
            result_value = DataDirectoryOrigFileNamesArg(value)
    if option_name in ['labels', 'c', 'm_tr_ss']:
        if not Path(value).exists():
            result_value = str(OMZ_DIR / value)
    return {option_name: result_value}


def handle_multi_option(key, *args) -> List[dict]:
    list_of_options = []
    for arg in args:
        list_of_options.append(handle_single_option(key, arg))
    return list_of_options


def get_test_options_from_config(options: dict) -> List[dict]:
    if not options or options == 'None':
        return [{}]
    general_level_options = {}
    result_options = [{}]
    for key, value in options.items():
        if 'split' not in key and not isinstance(value, List):
            general_level_options.update(handle_single_option(key, value))
        else:
            if 'split' in key:
                split_args = []
                for elem in value:
                    split_args.extend(get_test_options_from_config(elem))
                result_options = combine_cases(result_options, split_args)
            else:
                result_options = combine_cases(result_options, handle_multi_option(key, *value))
    result_options = combine_cases(result_options, general_level_options)
    return result_options


# Correction demo options according to demo flags

FLAGS_CPP = {
    'input': 'i',
    'model': 'm',
    'architecture_type': 'at'
}

def correct_demo_flags(set_of_flags, implementation):
    new_flags = []
    for flags in set_of_flags:
        options = {}
        for flag, value in flags.items():
            if implementation == 'python':
                flag = '--' + flag
            else:
                if flag.startswith(tuple(FLAGS_CPP.keys())):
                    for full, short in FLAGS_CPP.items():
                        flag = flag.replace(full, short)
                flag = '--' + flag
            options[flag] = value
        new_flags.append(options)
    return new_flags
