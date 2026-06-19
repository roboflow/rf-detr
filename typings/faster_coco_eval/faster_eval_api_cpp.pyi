from __future__ import annotations

import typing

import pybind11_stubgen.typing_ext

__all__: list[str] = [
    "COCOevalAccumulate",
    "COCOevalEvaluateAccumulate",
    "COCOevalEvaluateImages",
    "Dataset",
    "ImageEvaluation",
    "InstanceAnnotation",
    "calc_auc",
    "get_compiler_version",
]

class Dataset:
    def __getstate__(self) -> tuple: ...
    def __init__(self) -> None: ...
    def __len__(self) -> int: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def append(self, arg0: float, arg1: float, arg2: dict) -> None: ...
    def append_ref(self, arg0: float, arg1: float, arg2: typing.Any) -> None: ...
    def clean(self) -> None: ...
    def clear_cache_entry(self, arg0: float, arg1: float) -> None: ...
    def get(self, arg0: float, arg1: float) -> list[dict]: ...
    def get_cpp_annotations(self, arg0: float, arg1: float) -> list[InstanceAnnotation]: ...
    def get_cpp_instances(
        self, arg0: list[float], arg1: list[float], arg2: bool
    ) -> list[list[list[InstanceAnnotation]]]: ...
    def get_instances(self, arg0: list[float], arg1: list[float], arg2: bool) -> list[list[list[dict]]]: ...
    def load_tuple(self, arg0: tuple) -> None: ...
    def make_tuple(self) -> tuple: ...

class ImageEvaluation:
    def __getstate__(self) -> tuple: ...
    def __init__(self) -> None: ...
    def __setstate__(self, arg0: tuple) -> None: ...

class InstanceAnnotation:
    def __init__(self, arg0: int, arg1: float, arg2: float, arg3: bool, arg4: bool, arg5: bool) -> None: ...

def COCOevalAccumulate(arg0: typing.Any, arg1: list[...]) -> dict:
    """
    Accumulates evaluation statistics.
    """

def COCOevalEvaluateAccumulate(
    arg0: typing.Any,
    arg1: list[list[list[list[float]]]],
    arg2: ...,
    arg3: ...,
    arg4: list[float],
    arg5: list[float],
    arg6: bool,
) -> dict:
    """
    Performs evaluation and accumulation in one step.
    """

def COCOevalEvaluateImages(
    arg0: list[typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(2)]],
    arg1: int,
    arg2: list[float],
    arg3: list[list[list[list[float]]]],
    arg4: ...,
    arg5: ...,
    arg6: list[float],
    arg7: list[float],
    arg8: bool,
) -> list[...]:
    """
    Evaluates images based on detections and ground truth.
    """

def calc_auc(arg0: list[float], arg1: list[float]) -> float:
    """
    Calculates area under curve (AUC) for PR curve.
    """

def get_compiler_version() -> str:
    """
    Returns the compiler version used for compilation.
    """

__version__: str = "1.7.2"
