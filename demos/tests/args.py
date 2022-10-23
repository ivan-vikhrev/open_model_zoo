# Copyright (c) 2019-2022 Intel Corporation
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

import collections
import shutil
from pathlib import Path

import yaml

ArgContext = collections.namedtuple(
    "ArgContext", ["test_data_dir", "dl_dir", "model_info", "data_sequences", "data_sequence_dir"]
)

RequestedModel = collections.namedtuple("RequestedModel", ["name", "precisions"])

OMZ_DIR = Path(__file__).parents[2].resolve()


class AbstractArg(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader

    def resolve(self, context):
        raise NotImplementedError

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(node.value)


# Types for input arg


class TestDataArg(AbstractArg):
    yaml_tag = "!TestData"

    def __init__(self, rel_path):
        self.rel_path = rel_path

    def resolve(self, context):
        return str(context.test_data_dir / self.rel_path)


class OmzDataArg(AbstractArg):
    yaml_tag = "!OmzData"

    def __init__(self, rel_path):
        self.rel_path = rel_path
        self.context = OMZ_DIR

    def resolve(self, context):
        return str(self.context / self.rel_path)


class image_net_arg(TestDataArg):
    yaml_tag = "!image_net_arg"

    def __init__(self, image_id):
        self.rel_path = f"ILSVRC2012_img_val/ILSVRC2012_val_{image_id}.JPEG"


class brats_arg(TestDataArg):
    yaml_tag = "!brats_arg"

    def __init__(self, image_id):
        self.rel_path = f"BraTS/{image_id}"


class image_retrieval_arg(TestDataArg):
    yaml_tag = "!image_retrieval_arg"

    def __init__(self, image_id):
        self.rel_path = f"Image_Retrieval/{image_id}"


class DataPatternArg(AbstractArg):
    yaml_tag = "!DataPattern"

    def __init__(self, sequence_name):
        self.sequence_name = sequence_name

    def resolve(self, context):
        seq_dir = context.data_sequence_dir / self.sequence_name
        seq = [Path(data.resolve(context)) for data in context.data_sequences[self.sequence_name]]

        assert len({data.suffix for data in seq}) == 1, "all images in the sequence must have the same extension"
        assert "%" not in seq[0].suffix

        name_format = "input-%04d" + seq[0].suffix

        if not seq_dir.is_dir():
            seq_dir.mkdir(parents=True)

            for index, data in enumerate(context.data_sequences[self.sequence_name]):
                shutil.copyfile(data.resolve(context), str(seq_dir / (name_format % index)))

        return str(seq_dir / name_format)


class DataDirectoryArg(AbstractArg):
    yaml_tag = "!DataDirectory"

    def __init__(self, sequence_name):
        self.backend = DataPatternArg(sequence_name)

    def resolve(self, context):
        pattern = self.backend.resolve(context)
        return str(Path(pattern).parent)


class DataDirectoryOrigFileNamesArg(AbstractArg):
    yaml_tag = "!DataDirectoryOrigFileNames"

    def __init__(self, sequence_name):
        self.sequence_name = sequence_name

    def resolve(self, context):
        seq_dir = context.data_sequence_dir / self.sequence_name
        seq = [data.resolve(context) for data in context.data_sequences[self.sequence_name]]

        if not seq_dir.is_dir():
            seq_dir.mkdir(parents=True)

            for seq_item in seq:
                shutil.copyfile(seq_item, str(seq_dir / Path(seq_item).name))

        return str(seq_dir)


class DataListOfFilesArg(AbstractArg):
    yaml_tag = "!DataListOfFiles"

    def __init__(self, sequence_name):
        self.sequence_name = sequence_name

    def resolve(self, context):
        seq = [data.resolve(context) for data in context.data_sequences[self.sequence_name]]

        result_list = []

        for seq_item in seq:
            result_list.append(seq_item)

        return result_list


# Types for model args


class AbstractModelArg(AbstractArg):
    @property
    def required_models(self):
        return []


class ModelArg(AbstractModelArg):
    yaml_tag = "!Model"

    def __init__(self, name, precision=None):
        self.name = name
        self.precision = precision

    def resolve(self, context):
        return str(
            context.dl_dir / context.model_info[self.name]["subdirectory"] / self.precision / (self.name + ".xml")
        )

    @property
    def required_models(self):
        return [RequestedModel(self.name, [])]


class ModelFileArg(AbstractModelArg):
    yaml_tag = "!ModelFile"

    def __init__(self, name, file_name):
        self.name = name
        self.file_name = file_name

    def resolve(self, context):
        return str(context.dl_dir / context.model_info[self.name]["subdirectory"] / self.file_name)

    @property
    def required_models(self):
        return [RequestedModel(self.name, [])]

    @classmethod
    def from_yaml(cls, loader, node):
        return loader.construct_yaml_object(node, cls)


# Type for mean and scale values


class VectorArg(AbstractArg):
    yaml_tag = "!Vector"
    yaml_loader = yaml.SafeLoader

    def __init__(self, vector: list) -> None:
        self.vector = vector

    def resolve(self, context):
        result_vector = [elem.resolve(context) if isinstance(elem, AbstractArg) else elem for elem in self.vector]
        return result_vector

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(loader.construct_sequence(node))
