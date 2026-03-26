"""Plugin interfaces for extending the framework."""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Any


@runtime_checkable
class ModelPlugin(Protocol):
    name: str

    def build(self, **kwargs: Any):
        ...


@runtime_checkable
class LossPlugin(Protocol):
    name: str

    def build(self, **kwargs: Any):
        ...


@runtime_checkable
class OptimizerPlugin(Protocol):
    name: str

    def build(self, params, **kwargs: Any):
        ...


@runtime_checkable
class PreprocessorPlugin(Protocol):
    name: str

    def process(self, *args, **kwargs):
        ...
