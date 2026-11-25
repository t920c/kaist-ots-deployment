"""Compatibility loader for NVIDIA's renamed Windows TensorRT wheels.

TensorRT 10.14 installs its Python API as ``tensorrt_bindings`` while this
Ultralytics fork imports ``tensorrt``.  The Windows wheel also enumerates its
DLLs in an unsafe order, so load the three direct dependencies explicitly.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


_dll_handles = []
_libs_dir = next(
    (
        candidate
        for entry in sys.path
        if entry
        for candidate in (Path(entry) / "tensorrt_libs",)
        if candidate.is_dir()
    ),
    None,
)
if _libs_dir is None:
    raise ModuleNotFoundError("TensorRT runtime directory 'tensorrt_libs' was not found.")

for _name in ("nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll"):
    _dll = _libs_dir / _name
    if not _dll.is_file():
        raise FileNotFoundError(f"Required TensorRT DLL was not found: {_dll}")
    _dll_handles.append(ctypes.WinDLL(str(_dll)))

import tensorrt_bindings as _bindings  # noqa: E402
from tensorrt_bindings import *  # noqa: E402,F401,F403

__version__ = _bindings.__version__

