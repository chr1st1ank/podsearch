"""Semantic search over podcast episodes - profiling demo."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("podsearch")
except PackageNotFoundError:
    __version__ = "0.1.0"
