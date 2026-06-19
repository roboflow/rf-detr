from __future__ import annotations

import copy as copy
import json as json
import logging as logging
import os as os
import pathlib as pathlib
import time as time
import warnings as warnings
from collections import defaultdict

import numpy
import numpy as np
from faster_coco_eval.core import mask as maskUtils

__all__: list[str] = [
    "COCO",
    "copy",
    "defaultdict",
    "json",
    "logger",
    "logging",
    "maskUtils",
    "np",
    "os",
    "pathlib",
    "time",
    "warnings",
]

class COCO:
    @staticmethod
    def load_json(
        json_file: typing.Union[str, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath, dict, list],
        use_deepcopy: typing.Optional[bool] = False,
    ) -> dict:
        """
        Load a json file.

                Args:
                    json_file (Union[str, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath, dict, list]): Path to the json file or data dict/list.
                    use_deepcopy (Optional[bool], optional): If True, use deep copy. Defaults to False.

                Returns:
                    dict: Loaded json data.

        """
    def __init__(
        self,
        annotation_file: typing.Union[str, dict, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath, NoneType] = None,
        use_deepcopy: bool = False,
        print_function: typing.Callable = logging.Logger.debug,
    ):
        """
        Constructor of Microsoft COCO helper class.

                Args:
                    annotation_file (Optional[Union[str, dict, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath]], optional): Path to annotation file or annotation dict. Defaults to None.
                    use_deepcopy (bool, optional): Whether to copy the dict annotations. Defaults to False.
                    print_function (Callable, optional): Function to use for printing messages. Defaults to logger.debug.

        """
    def __iter__(self):
        """
        Iterate over the annotations.

                Yields:
                    Tuple[str, Any]: Key-value pair of the annotation data.

        """
    def __repr__(self) -> str:
        """
        String representation for COCO class.

                Returns:
                    str: Representation with author and version.

        """
    def annToMask(self, ann: dict) -> numpy.ndarray:
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE
                to binary mask.

                Args:
                    ann (dict): Annotation information.

                Returns:
                    np.ndarray: Binary mask of the annotation.

        """
    def annToRLE(self, ann: dict) -> dict:
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.

                Args:
                    ann (dict): Annotation information.

                Returns:
                    dict: Run-length encoding of the annotation.

        """
    def createIndex(self):
        """
        Create index for coco annotation data.

                Creates internal indices for the COCO dataset to enable fast
                lookups. Builds mappings between images, annotations and
                categories.

        """
    def download(self, tarDir=None, imgIds=list()):
        """
        Deprecated: Download images (no longer supported).

                Args:
                    tarDir (Any, optional): Target directory. Not used.
                    imgIds (list, optional): Image ids. Not used.

                Raises:
                    DeprecationWarning: Always raised, function is deprecated.

        """
    def dump(self, output_file: typing.Union[str, os.PathLike]):
        """
        Dump annotations to a json file.

                Args:
                    output_file (Union[str, os.PathLike]): Path to the output json file.

        """
    def getAnnIds(
        self,
        imgIds: typing.List[int] = list(),
        catIds: typing.List[int] = list(),
        areaRng: typing.List[float] = list(),
        iscrowd: bool = None,
    ) -> typing.List[int]:
        """
        Get ann ids that satisfy given filter conditions.

                Args:
                    imgIds (List[int], optional): Get anns for given images. Defaults to [].
                    catIds (List[int], optional): Get anns for given categories. Defaults to [].
                    areaRng (List[float], optional): Get anns for given area range (e.g. [0, inf]). Defaults to [].
                    iscrowd (bool, optional): Get anns for given crowd label (False or True). Defaults to None.

                Returns:
                    List[int]: Integer array of ann ids that satisfy the criteria.

        """
    def getCatIds(
        self, catNms: typing.List[str] = list(), supNms: typing.List[str] = list(), catIds: typing.List[int] = list()
    ) -> typing.List[int]:
        """
        Get category ids that satisfy given filter conditions.

                Args:
                    catNms (List[str], optional): Get categories for given cat names. Defaults to [].
                    supNms (List[str], optional): Get categories for given supercategory names. Defaults to [].
                    catIds (List[int], optional): Get categories for given ids. Defaults to [].

                Returns:
                    List[int]: Integer array of cat ids.

        """
    def getImgIds(self, imgIds: typing.List[int] = list(), catIds: typing.List[int] = list()) -> typing.List[int]:
        """
        Get image ids that satisfy given filter conditions.

                Args:
                    imgIds (List[int], optional): Get images for given ids. Defaults to [].
                    catIds (List[int], optional): Get images with all given categories. Defaults to [].

                Returns:
                    List[int]: Integer array of img ids.

        """
    def get_ann_ids(
        self,
        img_ids: typing.List[int] = list(),
        cat_ids: typing.List[int] = list(),
        area_rng: typing.List[float] = list(),
        iscrowd: bool = None,
    ) -> typing.List[int]:
        """
        Get ann ids that satisfy given filter conditions.

                Args:
                    img_ids (List[int], optional): Get anns for given imgs. Defaults to [].
                    cat_ids (List[int], optional): Get anns for given cats. Defaults to [].
                    area_rng (List[float], optional): Get anns with area less than this. Defaults to [].
                    iscrowd (bool, optional): Get anns for given crowd label. Defaults to None.

                Returns:
                    List[int]: Integer array of ann ids.

        """
    def get_cat_ids(
        self,
        cat_names: typing.List[str] = list(),
        sup_names: typing.List[str] = list(),
        cat_ids: typing.List[int] = list(),
    ) -> typing.List[int]:
        """
        Get cat ids that satisfy given filter conditions.

                Args:
                    cat_names (List[str], optional): Get cats for given names. Defaults to [].
                    sup_names (List[str], optional): Get cats for given supercategory names. Defaults to [].
                    cat_ids (List[int], optional): Get cats for given ids. Defaults to [].

                Returns:
                    List[int]: Integer array of cat ids.

        """
    def get_img_ids(self, img_ids: typing.List[int] = list(), cat_ids: typing.List[int] = list()) -> typing.List[int]:
        """
        Get img ids that satisfy given filter conditions.

                Args:
                    img_ids (List[int], optional): Get imgs for given ids. Defaults to [].
                    cat_ids (List[int], optional): Get imgs with all given cats. Defaults to [].

                Returns:
                    List[int]: Integer array of img ids.

        """
    def info(self):
        """
        Print information about the annotation file.

                Prints the info section of the annotation file using the print
                function.

        """
    def loadAnns(self, ids: typing.Union[typing.List[int], int] = list()) -> typing.List[dict]:
        """
        Load annotations with the specified ids.

                Args:
                    ids (Union[List[int], int], optional): Integer ids specifying annotations. Defaults to [].

                Returns:
                    List[dict]: Loaded annotation objects.

        """
    def loadCats(self, ids: typing.Union[typing.List[int], int] = list()) -> typing.List[dict]:
        """
        Load categories with the specified ids.

                Args:
                    ids (Union[List[int], int], optional): Integer ids specifying categories. Defaults to [].

                Returns:
                    List[dict]: Loaded category objects.

        """
    def loadImgs(self, ids: typing.Union[typing.List[int], int] = list()) -> typing.List[dict]:
        """
        Load images with the specified ids.

                Args:
                    ids (Union[List[int], int], optional): Integer ids specifying images. Defaults to [].

                Returns:
                    List[dict]: Loaded image objects.

        """
    def loadNumpyAnnotations(self, data: numpy.ndarray) -> typing.List[dict]:
        """
        Convert result data from array to anns.

                Args:
                    data (np.ndarray): 2d array where each row contains [imageID, x1, y1, w, h, score, class]

                Returns:
                    List[dict]: Converted annotations as a list of dicts.

        """
    def loadRes(
        self,
        resFile: typing.Union[str, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath, dict, list, numpy.ndarray],
        min_score: float = 0.0,
    ) -> COCO:
        """
        Load result file and return a result api object.

                Args:
                    resFile (Union[str, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath, dict, list, np.ndarray]): File name of result file or numpy array.
                    min_score (float, optional): Minimum score to consider a result. Defaults to 0.0.

                Returns:
                    COCO: Result api object.

        """
    def load_anns(self, ids: typing.List[int]) -> typing.List[dict]:
        """
        Load anns with the specified ids.

                Args:
                    ids (List[int]): Integer ids specifying anns.

                Returns:
                    List[dict]: Loaded annotation objects.

        """
    def load_cats(self, ids: typing.List[int]) -> typing.List[dict]:
        """
        Load cats with the specified ids.

                Args:
                    ids (List[int]): Integer ids specifying cats.

                Returns:
                    List[dict]: Loaded category objects.

        """
    def load_imgs(self, ids: typing.List[int]) -> typing.List[dict]:
        """
        Load imgs with the specified ids.

                Args:
                    ids (List[int]): Integer ids specifying imgs.

                Returns:
                    List[dict]: Loaded image objects.

        """
    def showAnns(self, anns: typing.List[dict], draw_bbox: typing.Optional[bool] = False):
        """
        Display the specified annotations.

                Args:
                    anns (List[dict]): Annotations to display.
                    draw_bbox (Optional[bool], optional): Whether to display bbox. Defaults to False.

        """
    def to_dict(self, separate_fn: bool = False) -> dict:
        """
        Convert to a standard python dictionary.

                Args:
                    separate_fn (bool, optional): Whether to separate the fn category. Defaults to False.

                Returns:
                    dict: Standard python dictionary containing the COCO data.

        """
    @property
    def cat_img_map(self) -> dict:
        """
        Return a mapping from category ids to image ids.

                Returns:
                    dict: Mapping from category ids to image ids.

        """
    @cat_img_map.setter
    def cat_img_map(self, value: dict):
        """
        Set the mapping from category ids to image ids.

                Args:
                    value (dict): Mapping from category ids to image ids.

        """
    @property
    def img_ann_map(self) -> dict:
        """
        Return a mapping from image ids to annotation ids.

                Returns:
                    dict: Mapping from image ids to annotation ids.

        """
    @img_ann_map.setter
    def img_ann_map(self, value: dict):
        """
        Set the mapping from image ids to annotation ids.

                Args:
                    value (dict): Mapping from image ids to annotation ids.

        """
    @property
    def print_function(self) -> typing.Callable:
        """
        Get the function used for printing/logging messages.

                Returns:
                    Callable: Print/log function.

        """
    @print_function.setter
    def print_function(self, value: typing.Callable):
        """
        Set the function used for printing/logging messages.

                Args:
                    value (Callable): Function to use for printing messages.

        """

def _isArrayLike(obj):
    """
    Check if the object is array-like.

        Args:
            obj (Any): Object to check.

        Returns:
            bool: True if object behaves like an array, False otherwise.

    """

__author__: str = "MiXaiLL76"
__version__: str = "1.7.2"
logger: logging.Logger  # value = <Logger faster_coco_eval.core.coco (INFO)>
