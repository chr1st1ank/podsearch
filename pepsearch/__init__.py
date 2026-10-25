"""Semantic search over the PEPs - profiling demo."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pepsearch")
except PackageNotFoundError:
    __version__ = "0.1.0"
