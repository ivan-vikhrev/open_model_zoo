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

from args import AbstractArg
from jsonschema import Draft202012Validator, validators
from utils import read_yaml

# Functions for additional types


def all_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


def add_custom_types(validator):
    type_checker = validator.TYPE_CHECKER
    # Add types

    for type in all_subclasses(AbstractArg):
        type_checker = type_checker.redefine(type.__name__, lambda checker, value, type=type: isinstance(value, type))

    CustomValidator = validators.extend(
        validator,
        type_checker=type_checker,
    )

    return CustomValidator


# Validation


def validate(config) -> None:
    class_validator = add_custom_types(Draft202012Validator)
    schema = read_yaml("validation/schema.yml")
    custom_validator = class_validator(schema)

    custom_validator.validate(config)
