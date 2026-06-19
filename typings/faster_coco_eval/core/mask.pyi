from __future__ import annotations

import typing

import cv2 as cv2
import numpy
import numpy as np

__all__: list[str] = [
    "ValidRleType",
    "area",
    "calculateRleForAllAnnotations",
    "cv2",
    "decode",
    "encode",
    "frPyObjects",
    "iou",
    "merge",
    "np",
    "opencv_available",
    "rleToBoundary",
    "rleToBoundaryCV",
    "segmToRle",
    "toBbox",
]

def _check_opencv():
    """
    Check if OpenCV is available and raise informative error.
    """

def area(rleObjs: typing.Union[dict, typing.List[dict]]) -> numpy.ndarray:
    """
    Compute area of encoded masks.

        Args:
            rleObjs (Union[dict, List[dict]]): Run-length encoding of binary mask(s).

        Returns:
            np.ndarray: Area(s) of run-length encodings.

    """

def calculateRleForAllAnnotations(
    anns: typing.List[dict],
    img_sizes: typing.Dict[int, tuple],
    compute_rle: bool,
    compute_boundary: bool,
    boundary_dilation_ratio: float,
    boundary_cpu_count: int,
):
    """
    Calculate run-length encoding for all annotations.

        Args:
            anns (List[dict]): List of annotation dictionaries.
            img_sizes (Dict[int, tuple]): Dictionary mapping image ids to their sizes (height, width).
            compute_rle (bool): Whether to compute run-length encoding.
            compute_boundary (bool): Whether to compute boundary run-length encoding.
            boundary_dilation_ratio (float): Ratio of dilation to apply to the mask boundary.
            boundary_cpu_count (int): Number of CPUs to use for boundary computation.

        Returns:
            None

    """

def decode(rleObjs: typing.Union[dict, typing.List[dict]]) -> numpy.ndarray:
    """
    Decode binary masks encoded via RLE.

        Args:
            rleObjs (Union[dict, List[dict]]): Run-length encoding of binary mask(s).

        Returns:
            np.ndarray: Decoded binary mask(s).

    """

def encode(bimask: numpy.ndarray) -> dict:
    """
    Encode binary mask(s) using RLE.

        Args:
            bimask (np.ndarray): Binary mask. Can be 2D (H, W) or 3D (H, W, N).

        Returns:
            dict: Run-length encoding of the binary mask(s).

    """

def frPyObjects(
    objs: typing.Union[
        typing.List[numpy.ndarray],
        typing.List[typing.List[float]],
        numpy.ndarray,
        typing.List[dict],
        typing.List[float],
        dict,
    ],
    h: int,
    w: int,
) -> typing.Union[dict, typing.List[dict]]:
    """
    Convert a list of objects to RLE format suitable for use in mask API.

        Args:
            objs (Union[ValidRleType, np.ndarray, List[float], dict]): Objects to be converted (polygons, bboxes, etc).
            h (int): Height of the image.
            w (int): Width of the image.

        Returns:
            Union[dict, List[dict]]: Run-length encoding of the objects.

    """

def iou(
    dt: typing.Union[typing.List[numpy.ndarray], typing.List[typing.List[float]], numpy.ndarray, typing.List[dict]],
    gt: typing.Union[typing.List[numpy.ndarray], typing.List[typing.List[float]], numpy.ndarray, typing.List[dict]],
    iscrowd: typing.List[int],
) -> typing.Union[list, numpy.ndarray]:
    """
    Compute intersection over union (IoU) between two sets of run-length
        encoded masks.

        Args:
            dt (ValidRleType): Detected masks, can be a list of RLEs or arrays.
            gt (ValidRleType): Ground truth masks, can be a list of RLEs or arrays.
            iscrowd (List[int]): List of flags indicating whether each ground truth mask is a crowd region.

        Returns:
            Union[list, np.ndarray]: Intersection over union between dt and gt masks.

    """

def merge(rleObjs: typing.List[dict], intersect: int = 0):
    """
    Merge a list of run-length encoded objects.

        Args:
            rleObjs (List[dict]): List of run-length encoding dictionaries of binary masks.
            intersect (int, optional): Flag for type of merge to perform. Defaults to 0.

        Returns:
            dict: Run-length encoding of merged mask.

    """

def rleToBoundary(rle: dict, dilation_ratio: float = 0.02, backend: str = "mask_api") -> dict:
    """
    Convert run-length encoding to boundary rle.

        Args:
            rle (dict): Run-length encoding of a binary mask.
            dilation_ratio (float, optional): Ratio of dilation to apply to the mask. Defaults to 0.02.
            backend (str, optional): Backend to use for conversion. 'mask_api' uses the faster_eval_api_cpp backend,
                'opencv' uses OpenCV for conversion. Defaults to "mask_api".

        Returns:
            dict: Run-length encoding of the boundary mask.

        Raises:
            ImportError: If OpenCV is selected as backend but not available.

    """

def rleToBoundaryCV(rle: dict, dilation_ratio: float = 0.02) -> dict:
    """
    Convert run-length encoding to boundary rle using OpenCV backend.

        Args:
            rle (dict): Run-length encoding of a binary mask.
            dilation_ratio (float, optional): Ratio of dilation to apply to the mask. Defaults to 0.02.

        Returns:
            dict: Run-length encoding of the boundary mask.

    """

def segmToRle(segm: typing.Union[typing.List[float], typing.List[int], dict], w: int, h: int):
    """
    Convert segm array to run-length encoding.

        Args:
            segm (Union[List[float], List[int], dict]): Segmentation map, can be a list of floats, ints or a dictionary.
            w (int): Width of the image.
            h (int): Height of the image.

        Returns:
            dict: Run-length encoding of the segmentation map.

    """

def toBbox(rleObjs: typing.Union[dict, typing.List[dict]]) -> numpy.ndarray:
    """
    Get bounding boxes surrounding encoded masks.

        Args:
            rleObjs (Union[dict, List[dict]]): Run-length encoding of binary mask(s).

        Returns:
            np.ndarray: Bounding box(es) of run-length encodings.

    """

ValidRleType: (
    typing._UnionGenericAlias
)  # value = typing.Union[typing.List[numpy.ndarray], typing.List[typing.List[float]], numpy.ndarray, typing.List[dict]]
opencv_available: bool = True
