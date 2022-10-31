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
Constructions for registering classes (one of the variants of factories)
"""


class UnregisteredProviderException(ValueError):
    def __init__(self, provider, root_provider):
        self.provider = provider
        self.root_provider = root_provider
        self.message = "Requested provider {} not registered for {}".format(provider, root_provider)
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
    def check_provider(cls, name):
        if name not in cls.providers:
            return False
        return True

    @classmethod
    def resolve(cls, name):
        if name not in cls.providers:
            raise UnregisteredProviderException(name, cls.__provider_type__)
        return cls.providers[name]


class ClassProviderMeta(type):
    def __new__(mcs, name, bases, attrs, **kwargs):
        cls = type.__new__(mcs, name, bases, attrs)
        # do not create container for abstract provider
        if "_is_base_provider" in attrs:
            return cls

        if "__provider_type__" in attrs:
            cls.providers = {}
        else:
            cls.register_provider(cls)

        return cls


class ClassProvider(BaseProvider, metaclass=ClassProviderMeta):
    _is_base_provider = True

    @classmethod
    def get_provider_name(cls):
        return getattr(cls, "__provider__", cls.__name__)

    @classmethod
    def register_provider(cls, provider):
        provider_name = cls.get_provider_name()
        if not provider_name:
            return
        cls.providers[provider_name] = provider
