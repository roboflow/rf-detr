from __future__ import annotations
import numpy
import typing
__all__: list[str] = ['RLE', 'area', 'bbIou', 'calculateRleForAllAnnotations', 'decode', 'encode', 'erode_3x3', 'frBbox', 'frPoly', 'frPyObjects', 'frUncompressedRLE', 'get_compiler_version', 'iou', 'merge', 'rleDecode', 'rleEncode', 'rleFrBbox', 'rleFrPoly', 'rleFrString', 'rleIou', 'rleToBbox', 'rleToString', 'rleToUncompressedRLE', 'segmToRle', 'toBbox', 'toBoundary', 'toUncompressedRLE']
class RLE:
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: list[int]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: int, arg2: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: list[float], arg1: int, arg2: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: list[float], arg1: int, arg2: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: list[RLE], arg1: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: dict) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing.Any, arg1: int, arg2: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: tuple[int, int, str]) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def area(self) -> int:
        ...
    def erode_3x3(self, arg0: int) -> RLE:
        ...
    def toBbox(self) -> list[float]:
        ...
    def toBoundary(self, arg0: float) -> RLE:
        ...
    def toDict(self) -> dict:
        ...
    def toString(self) -> str:
        ...
def _frString(arg0: list[dict]) -> list[RLE]:
    """
    Mask::_frString
    """
def _toString(arg0: list[RLE]) -> list[dict]:
    """
    Mask::_toString
    """
def area(arg0: list[dict]) -> numpy.ndarray[numpy.uint64]:
    """
    Mask::area
    """
def bbIou(arg0: list[float], arg1: list[float], arg2: int, arg3: int, arg4: list[int]) -> list[float]:
    """
    Mask::bbIou
    """
def calculateRleForAllAnnotations(arg0: list[dict], arg1: dict[int, tuple[int, int]], arg2: bool, arg3: bool, arg4: float, arg5: int) -> None:
    """
    Mask::calculateRleForAllAnnotations
    """
def decode(arg0: list[dict]) -> numpy.ndarray[numpy.uint8]:
    """
    Mask::decode
    """
def encode(arg0: numpy.ndarray[numpy.uint8]) -> list[dict]:
    """
    Mask::encode
    """
def erode_3x3(arg0: list[dict], arg1: int) -> list[dict]:
    """
    Mask::erode_3x3
    """
def frBbox(arg0: list[list[float]], arg1: int, arg2: int) -> list[dict]:
    """
    Mask::frBbox
    """
def frPoly(arg0: list[list[float]], arg1: int, arg2: int) -> list[dict]:
    """
    Mask::frPoly
    """
def frPyObjects(arg0: typing.Any, arg1: int, arg2: int) -> dict | list[dict]:
    """
    Mask::frPyObjects
    """
def frUncompressedRLE(arg0: list[dict]) -> list[dict]:
    """
    Mask::frUncompressedRLE
    """
def get_compiler_version() -> str:
    """
    get_compiler_version
    """
def iou(arg0: typing.Any, arg1: typing.Any, arg2: list[int]) -> numpy.ndarray[numpy.float64] | list[float]:
    """
    Mask::iou
    """
@typing.overload
def merge(arg0: list[dict], arg1: int) -> dict:
    """
    Mask::merge
    """
@typing.overload
def merge(arg0: list[dict]) -> dict:
    """
    Mask::merge
    """
def rleDecode(arg0: list[RLE]) -> numpy.ndarray[numpy.uint8]:
    """
    Mask::rleDecode
    """
def rleEncode(arg0: numpy.ndarray[numpy.uint8], arg1: int, arg2: int, arg3: int) -> list[RLE]:
    """
    Mask::rleEncode
    """
def rleFrBbox(arg0: list[float], arg1: int, arg2: int, arg3: int) -> list[RLE]:
    """
    Mask::rleFrBbox
    """
def rleFrPoly(arg0: list[float], arg1: int, arg2: int, arg3: int) -> RLE:
    """
    Mask::rleFrPoly
    """
def rleFrString(arg0: str, arg1: int, arg2: int) -> RLE:
    """
    Mask::rleFrString
    """
def rleIou(arg0: list[RLE], arg1: list[RLE], arg2: int, arg3: int, arg4: list[int]) -> list[float]:
    """
    Mask::rleIou
    """
def rleToBbox(arg0: list[RLE], arg1: int | None) -> numpy.ndarray:
    """
    Mask::rleToBbox
    """
def rleToString(arg0: RLE) -> bytes:
    """
    Mask::rleToString
    """
def rleToUncompressedRLE(arg0: list[RLE]) -> list[dict]:
    """
    Mask::rleToUncompressedRLE
    """
def segmToRle(arg0: typing.Any, arg1: int, arg2: int) -> dict | typing.Any:
    """
    Mask::segmToRle
    """
def toBbox(arg0: list[dict]) -> numpy.ndarray[numpy.float64]:
    """
    Mask::toBbox
    """
def toBoundary(arg0: list[dict], arg1: float) -> list[dict]:
    """
    Mask::toBoundary
    """
def toUncompressedRLE(arg0: list[dict]) -> list[dict]:
    """
    Mask::toUncompressedRLE
    """
__version__: str = '1.7.2'
