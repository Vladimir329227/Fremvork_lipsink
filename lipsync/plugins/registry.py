"""Plugin registry and dynamic discovery via entry points."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any


@dataclass
class PluginManifest:
    name: str
    kind: str
    object_path: str
    version: str = "unknown"


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, dict[str, Any]] = {
            "model": {},
            "loss": {},
            "optimizer": {},
            "preprocessor": {},
        }

    def register(self, kind: str, name: str, plugin: Any) -> None:
        kind = kind.lower()
        if kind not in self._plugins:
            raise KeyError(f"Unknown plugin kind '{kind}'")
        self._plugins[kind][name] = plugin

    def get(self, kind: str, name: str) -> Any:
        return self._plugins[kind][name]

    def list(self, kind: str | None = None) -> dict[str, list[str]] | list[str]:
        if kind is None:
            return {k: sorted(v.keys()) for k, v in self._plugins.items()}
        return sorted(self._plugins[kind].keys())

    def discover_entry_points(self) -> list[PluginManifest]:
        """Discover plugins registered as Python entry points.

        Supported groups:
        - lipsync.plugins.model
        - lipsync.plugins.loss
        - lipsync.plugins.optimizer
        - lipsync.plugins.preprocessor
        """
        manifests: list[PluginManifest] = []
        groups = {
            "lipsync.plugins.model": "model",
            "lipsync.plugins.loss": "loss",
            "lipsync.plugins.optimizer": "optimizer",
            "lipsync.plugins.preprocessor": "preprocessor",
        }
        eps = entry_points()
        for ep_group, kind in groups.items():
            for ep in eps.select(group=ep_group):
                obj = ep.load()
                self.register(kind, ep.name, obj)
                manifests.append(
                    PluginManifest(
                        name=ep.name,
                        kind=kind,
                        object_path=f"{ep.module}:{ep.attr}",
                        version=getattr(obj, "__version__", "unknown"),
                    )
                )
        return manifests


GLOBAL_PLUGIN_REGISTRY = PluginRegistry()
