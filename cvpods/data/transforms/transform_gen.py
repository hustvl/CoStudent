#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import inspect
import math
import pprint
import random
from re import I
import sys
from abc import ABCMeta, abstractmethod
from tkinter.tix import Tree
from unittest import result
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as tfs
import torch
import torchvision
from torchvision.transforms import functional as F

from cvpods.structures import Boxes, BoxMode, pairwise_iou

from ..registry import TRANSFORMS
from .auto_aug import AutoAugmentTransform
from .transform_util import bb_intersection_over_union, get_image_size

from .transform import (  # isort:skip
    ScaleTransform,
    AffineTransform,
    BlendTransform,
    IoUCropTransform,
    CropTransform,
    CropPadTransform,
    HFlipTransform,
    NoOpTransform,
    VFlipTransform,
    DistortTransform,
    RandomSwapChannelsTransform,
    ExpandTransform,
    ExtentTransform,
    ResizeTransform,
    PILColorTransform,
    # Transforms used in ssl
    GaussianBlurTransform,
    GaussianBlurConvTransform,
    SolarizationTransform,
    LightningTransform,
    ComposeTransform,
    TorchTransform,
    # LabSpaceTransform,
    PadTransform,
    FiveCropTransform,
    JigsawCropTransform,
)

__all__ = [
    "Pad",
    "PILColorTransformGen",
    "RandomScale",
    "Expand",
    "MinIoURandomCrop",
    "RandomSwapChannels",
    "CenterAffine",
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomCropWithInstance",
    "RandomCropWithMaxAreaLimit",
    "RandomCropPad",
    "RandomExtent",
    "RandomFlip",
    "RandomSaturation",
    "RandomLighting",
    "RandomDistortion",
    "Resize",
    "ResizeShortestEdge",
    "ResizeLongestEdge",
    "ShuffleList",
    "RandomList",
    "RepeatList",
    "RepeatListv2",
    "OrderList",
    "TransformGen",
    "TorchTransformGen",
    # transforms used in ssl
    "GaussianBlur",
    "GaussianBlurConv",
    "Solarization",
    "Lightning",
    "RandomFiveCrop",
    "AutoAugment",
    "RandAutoAugment",
    "JigsawCrop",
    "ColorJiter",
    "OneOf",
    "RandFlip",
    "RandResizeShortestEdge",
    "RandTranslate",
    "RandRotate",
    "RandShear",
    "RandErase",
]

"""
Record the geometric transformation information used in the augmentation in a transformation matrix.
"""
cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

def _get_shear_matrix(magnitude, direction='horizontal'):
    """Generate the shear matrix for transformation.

    Args:
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".

    Returns:
        ndarray: The shear matrix with dtype float32.
    """
    if direction == 'horizontal':
        shear_matrix = np.float32([[1, magnitude, 0], [0, 1, 0]])
    elif direction == 'vertical':
        shear_matrix = np.float32([[1, 0, 0], [magnitude, 1, 0]])
    return shear_matrix

def imshear(img,
            magnitude,
            direction='horizontal',
            border_value=0,
            interpolation='bilinear'):
    """Shear an image.

    Args:
        img (ndarray): Image to be sheared with format (h, w)
            or (h, w, c).
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".
        border_value (int | tuple[int]): Value used in case of a
            constant border.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The sheared image.
    """
    assert direction in ['horizontal',
                         'vertical'], f'Invalid direction: {direction}'
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]
    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, \
            'Expected the num of elements in tuple equals the channels' \
            'of input image. Found {} vs {}'.format(
                len(border_value), channels)
    else:
        raise ValueError(
            f'Invalid type {type(border_value)} for `border_value`')
    shear_matrix = _get_shear_matrix(magnitude, direction)
    sheared = cv2.warpAffine(
        img,
        shear_matrix,
        (width, height),
        # Note case when the number elements in `border_value`
        # greater than 3 (e.g. shearing masks whose channels large
        # than 3) will raise TypeError in `cv2.warpAffine`.
        # Here simply slice the first 3 values in `border_value`.
        borderValue=border_value[:3],
        flags=cv2_interp_codes[interpolation])
    return sheared
def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             border_value=0,
             interpolation='bilinear',
             auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        interpolation (str): Same as :func:`resize`.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        matrix, (w, h),
        flags=cv2_interp_codes[interpolation],
        borderValue=border_value)
    return rotated


def _get_translate_matrix(offset, direction='horizontal'):
    """Generate the translate matrix.

    Args:
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either
            "horizontal" or "vertical".

    Returns:
        ndarray: The translate matrix with dtype float32.
    """
    if direction == 'horizontal':
        translate_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
    elif direction == 'vertical':
        translate_matrix = np.float32([[1, 0, 0], [0, 1, offset]])
    return translate_matrix


def imtranslate(img,
                offset,
                direction='horizontal',
                border_value=0,
                interpolation='bilinear'):
    """Translate an image.

    Args:
        img (ndarray): Image to be translated with format
            (h, w) or (h, w, c).
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either "horizontal"
            or "vertical".
        border_value (int | tuple[int]): Value used in case of a
            constant border.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The translated image.
    """
    assert direction in ['horizontal',
                         'vertical'], f'Invalid direction: {direction}'
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]
    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, \
            'Expected the num of elements in tuple equals the channels' \
            'of input image. Found {} vs {}'.format(
                len(border_value), channels)
    else:
        raise ValueError(
            f'Invalid type {type(border_value)} for `border_value`.')
    translate_matrix = _get_translate_matrix(offset, direction)
    translated = cv2.warpAffine(
        img,
        translate_matrix,
        (width, height),
        # Note case when the number elements in `border_value`
        # greater than 3 (e.g. translating masks whose channels
        # large than 3) will raise TypeError in `cv2.warpAffine`.
        # Here simply slice the first 3 values in `border_value`.
        borderValue=border_value[:3],
        flags=cv2_interp_codes[interpolation])
    return translated

def bbox2fields():
    """The key correspondence from bboxes to labels, masks and
    segmentations."""
    bbox2label = {"gt_bboxes": "gt_labels", "gt_bboxes_ignore": "gt_labels_ignore"}
    bbox2mask = {"gt_bboxes": "gt_masks", "gt_bboxes_ignore": "gt_masks_ignore"}
    bbox2seg = {
        "gt_bboxes": "gt_semantic_seg",
    }
    return bbox2label, bbox2mask, bbox2seg

@TRANSFORMS.register()
class TransformGen(metaclass=ABCMeta):
    """
    TransformGen takes an image of type uint8 in range [0, 255], or
    floating point in range [0, 1] or [0, 255] as input.

    It creates a :class:`Transform` based on the given image, sometimes with randomness.
    The transform can then be used to transform images
    or other data (boxes, points, annotations, etc.) associated with it.

    The assumption made in this class
    is that the image itself is sufficient to instantiate a transform.
    When this assumption is not true, you need to create the transforms by your own.

    A list of `TransformGen` can be applied with :func:`apply_transform_gens`.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def get_transform(self, img, annotations=None):
        raise NotImplementedError

    def __call__(self, img, annotations=None, **kwargs):
        tfm = self.get_transform(img, annotations)
        img, annotations = tfm(img, annotations, **kwargs)
        return img, annotations

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyTransformGen(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL
                    and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(
                        name
                    )
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__

class GeometricAugmentation(TransformGen):
    def __init__(
        self,
        img_fill_val=125,
        seg_ignore_label=255,
        min_size=0,
        prob: float = 1.0,
        random_magnitude: bool = True,
        record: bool = False,
    ):
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, "img_fill_val as tuple must have 3 elements."
            img_fill_val = tuple([float(val) for val in img_fill_val])
        assert np.all(
            [0 <= val <= 255 for val in img_fill_val]
        ), "all elements of img_fill_val should between range [0,255]."
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.min_size = min_size
        self.prob = prob
        self.random_magnitude = random_magnitude
        self.record = record
        self.magnitude = None

    def __call__(self, img, annotations,**kwargs):
        if np.random.random() < self.prob:
            magnitude: dict = self.get_magnitude(img,annotations)
            if self.record:
                if "aug_info" not in annotations[0]:
                    annotations[0]["aug_info"] = []
                annotations[0]["aug_info"].append(self.get_aug_info(**magnitude))
            img,annotations = self.apply(img,annotations, **magnitude)

            self._filter_invalid(annotations, min_size=self.min_size)
        return img, annotations

    def get_transform(self, img, annotations=None):
        pass

    def get_magnitude(self, img, annotations) -> dict:
        raise NotImplementedError()

    def apply(self, img, annotations, **kwargs):
        raise NotImplementedError()

    def enable_record(self, mode: bool = True):
        self.record = mode

    def get_aug_info(self, **kwargs):
        aug_info = dict(type=self.__class__.__name__)
        aug_info.update(
            dict(
                # make op deterministic
                prob=1.0,
                random_magnitude=False,
                record=False,
                img_fill_val=self.img_fill_val,
                seg_ignore_label=self.seg_ignore_label,
                min_size=self.min_size,
            )
        )
        aug_info.update(kwargs)
        return aug_info
##TODO
    def _filter_invalid(self, annotations, min_size=0):
        """Filter bboxes and masks too small or translated out of image."""
        if min_size is None:
            return annotations
        valid_annotations = []
        # from pdb import set_trace
        # set_trace()
        for i,annotation in enumerate(annotations):
            if annotation['bbox'].ndim == 2 and len(annotation['bbox']) ==1:
                annotation['bbox'] = annotation['bbox'][0]
            bbox_w = annotation['bbox'][2] - annotation['bbox'][0]
            bbox_h = annotation['bbox'][3] - annotation['bbox'][1]
            valid = (bbox_w > min_size) & (bbox_h > min_size)
            if valid:
                valid_annotations.append(annotation)

        return valid_annotations

    def __repr__(self):
        return f"""{self.__class__.__name__}(
        img_fill_val={self.img_fill_val},
        seg_ignore_label={self.seg_ignore_label},
        min_size={self.magnitude},
        prob: float = {self.prob},
        random_magnitude: bool = {self.random_magnitude},
        )"""


class GTrans(object):
    @classmethod
    def inverse(cls, results):
        # compute the inverse
        return results["transform_matrix"].I  # 3x3

    @classmethod
    def apply(self, results, operator, **kwargs):
        if not hasattr(self, f"_get_{operator}_matrix"):
            results["transform_matrix"] = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            trans_matrix = getattr(self, f"_get_{operator}_matrix")(**kwargs)
            if "transform_matrix" not in results:
                results["transform_matrix"] = trans_matrix
            else:
                base_transformation = results["transform_matrix"]
                results["transform_matrix"] = np.dot(trans_matrix, base_transformation)

    @classmethod
    def apply_cv2_matrix(self, results, cv2_matrix):
        if cv2_matrix.shape[0] == 2:
            mat = np.concatenate(
                [cv2_matrix, np.array([0, 0, 1]).reshape((1, 3))], axis=0
            )
        else:
            mat = cv2_matrix
        base_transformation = results["transform_matrix"]
        results["transform_matrix"] = np.dot(mat, base_transformation)
        return results

    @classmethod
    def _get_rotate_matrix(cls, degree=None, cv2_rotation_matrix=None, inverse=False):
        # TODO: this is rotated by zero point
        if degree is None and cv2_rotation_matrix is None:
            raise ValueError(
                "At least one of degree or rotation matrix should be provided"
            )
        if degree:
            if inverse:
                degree = -degree
            rad = degree * np.pi / 180
            sin_a = np.sin(rad)
            cos_a = np.cos(rad)
            return np.array([[cos_a, sin_a, 0], [-sin_a, cos_a, 0], [0, 0, 1]])  # 2x3
        else:
            mat = np.concatenate(
                [cv2_rotation_matrix, np.array([0, 0, 1]).reshape((1, 3))], axis=0
            )
            if inverse:
                mat = mat * np.array([[1, -1, -1], [-1, 1, -1], [1, 1, 1]])
            return mat

    @classmethod
    def _get_shift_matrix(cls, dx=0, dy=0, inverse=False):
        if inverse:
            dx = -dx
            dy = -dy
        # from pdb import set_trace
        # set_trace()
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

    @classmethod
    def _get_shear_matrix(
        cls, degree=None, magnitude=None, direction="horizontal", inverse=False
    ):
        if magnitude is None:
            assert degree is not None
            rad = degree * np.pi / 180
            magnitude = np.tan(rad)

        if inverse:
            magnitude = -magnitude
        if direction == "horizontal":
            shear_matrix = np.float32([[1, magnitude, 0], [0, 1, 0], [0, 0, 1]])
        else:
            shear_matrix = np.float32([[1, 0, 0], [magnitude, 1, 0], [0, 0, 1]])
        return shear_matrix

    @classmethod
    def _get_flip_matrix(cls, shape, direction="horizontal", inverse=False):
        h, w = shape
        # from pdb import set_trace
        # set_trace()
        if direction == "horizontal":
            flip_matrix = np.float32([[-1, 0, w], [0, 1, 0], [0, 0, 1]])
        else:
            flip_matrix = np.float32([[1, 0, 0], [0, - 1, 0], [0, h, 1]])
        return flip_matrix

    @classmethod
    def _get_scale_matrix(cls, sx, sy, inverse=False):
        if inverse:
            sx = 1 / sx
            sy = 1 / sy
        return np.float32([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

@TRANSFORMS.register()
class RandTranslate(GeometricAugmentation):
    def __init__(self, x=None, y=None, **kwargs):

        super().__init__(**kwargs)
        self.x = x
        self.y = y
        if self.x is None and self.y is None:
            self.prob = 0.0

    def get_magnitude(self, img, annotations):
        magnitude = {}
        if self.random_magnitude:
            if isinstance(self.x, (list, tuple)):
                assert len(self.x) == 2
                x = np.random.random() * (self.x[1] - self.x[0]) + self.x[0]
                magnitude["x"] = x
            if isinstance(self.y, (list, tuple)):
                assert len(self.y) == 2
                y = np.random.random() * (self.y[1] - self.y[0]) + self.y[0]
                magnitude["y"] = y
        else:
            if self.x is not None:
                assert isinstance(self.x, (int, float))
                magnitude["x"] = self.x
            if self.y is not None:
                assert isinstance(self.y, (int, float))
                magnitude["y"] = self.y
        return magnitude

    def apply(self,img, annotations, x=None, y=None):
        # ratio to pixel
        h, w, c = img.shape
        if x is not None:
            x = w * x
        if y is not None:
            y = h * y
        if x is not None:
            # translate horizontally
            img,annotations = self._translate(img,annotations, x)
        if y is not None:
            # translate veritically
            img,annotations = self._translate(img,annotations, y, direction="vertical")
        return img,annotations

    def _translate(self,img, annotations, offset, direction="horizontal"):
        if self.record:
            GTrans.apply(
                annotations[0],
                "shift",
                dx=offset if direction == "horizontal" else 0,
                dy=offset if direction == "vertical" else 0,
            )
        img = self._translate_img(img, offset, direction=direction)
        annotations = self._translate_bboxes(img,annotations, offset, direction=direction)
        # fill_val defaultly 0 for BitmapMasks and None for PolygonMasks.
        # self._translate_masks(annotations, offset, direction=direction)
        # self._translate_seg(
        #     annotations, offset, fill_val=self.seg_ignore_label, direction=direction
        # )
        return img,annotations

    def _translate_img(self, img, offset, direction="horizontal"):

        return imtranslate(img, offset, direction,border_value=self.img_fill_val).astype(img.dtype)

    def _translate_bboxes(self,img, annotations, offset, direction="horizontal"):
        """Shift bboxes horizontally or vertically, according to offset."""
        h, w, c = img.shape

        for annotation in annotations:
            min_x, min_y, max_x, max_y = np.split(
                annotation['bbox'], annotation['bbox'].shape[-1], axis=-1
            )
            if direction == "horizontal":
                min_x = np.maximum(0, min_x + offset)
                max_x = np.minimum(w, max_x + offset)
            elif direction == "vertical":
                min_y = np.maximum(0, min_y + offset)
                max_y = np.minimum(h, max_y + offset)

            # the boxes translated outside of image will be filtered along with
            # the corresponding masks, by invoking ``_filter_invalid``.
            annotation['bbox'] = np.concatenate([min_x, min_y, max_x, max_y], axis=-1)

        return annotations
    # def _translate_masks(self, results, offset, direction="horizontal", fill_val=0):
    #     """Translate masks horizontally or vertically."""
    #     h, w, c = results["img_shape"]
    #     for key in results.get("mask_fields", []):
    #         masks = annotation['bbox']
    #         annotation['bbox'] = masks.translate((h, w), offset, direction, fill_val)

    # def _translate_seg(self, results, offset, direction="horizontal", fill_val=255):
    #     """Translate segmentation maps horizontally or vertically."""
    #     for key in results.get("seg_fields", []):
    #         seg = annotation['bbox'].copy()
    #         annotation['bbox'] = mmcv.imtranslate(seg, offset, direction, fill_val).astype(
    #             seg.dtype
    #         )

    def __repr__(self):
        repr_str = super().__repr__()
        return ("\n").join(
            repr_str.split("\n")[:-1]
            + [f"x={self.x}", f"y={self.y}"]
            + repr_str.split("\n")[-1:]
        )


@TRANSFORMS.register()
class RandRotate(GeometricAugmentation):
    def __init__(self, angle=None, center=None, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.angle = angle
        self.center = center
        self.scale = scale
        if self.angle is None:
            self.prob = 0.0

    def get_magnitude(self, img, annotations):
        magnitude = {}
        if self.random_magnitude:
            if isinstance(self.angle, (list, tuple)):
                assert len(self.angle) == 2
                angle = (
                    np.random.random() * (self.angle[1] - self.angle[0]) + self.angle[0]
                )
                magnitude["angle"] = angle
        else:
            if self.angle is not None:
                assert isinstance(self.angle, (int, float))
                magnitude["angle"] = self.angle

        return magnitude

    def apply(self, img,annotations, angle: float = None, **kwargs):

        h, w = img.shape[:2]
        center = self.center
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        img = self._rotate_img(img, angle, center, self.scale)
        rotate_matrix = cv2.getRotationMatrix2D(center, -angle, self.scale)
        if not self.record:
            from pdb import set_trace
            set_trace()
        if self.record:
            GTrans.apply(annotations[0], "rotate", cv2_rotation_matrix=rotate_matrix)
        annotations = self._rotate_bboxes(img,annotations, rotate_matrix)

        return img, annotations

    def _rotate_img(self, img, angle, center=None, scale=1.0):
        """Rotate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            angle (float): Rotation angle in degrees, positive values
                mean clockwise rotation. Same in ``mmcv.imrotate``.
            center (tuple[float], optional): Center point (w, h) of the
                rotation. Same in ``mmcv.imrotate``.
            scale (int | float): Isotropic scale factor. Same in
                ``mmcv.imrotate``.
        """
        return imrotate(img, angle, center, scale, border_value=self.img_fill_val
            ).astype(img.dtype)


    def _rotate_bboxes(self,img, annotations, rotate_matrix):
        """Rotate the bboxes."""

        h, w, c = img.shape
        for annotation in annotations:
            min_x, min_y, max_x, max_y = np.split(
                annotation['bbox'], annotation['bbox'].shape[-1], axis=-1
            )
            coordinates = np.stack(
                [[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]]
            )  # [4, 2, nb_bbox, 1]
            coordinates = coordinates.reshape(4,2,-1,1)
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coordinates = np.concatenate(
                (
                    coordinates,
                    np.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype),
                ),
                axis=1,
            )  # [4, 3, nb_bbox, 1]
            coordinates = coordinates.transpose((2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = np.matmul(rotate_matrix, coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            min_x, min_y = (
                np.min(rotated_coords[:, :, 0], axis=1),
                np.min(rotated_coords[:, :, 1], axis=1),
            )
            max_x, max_y = (
                np.max(rotated_coords[:, :, 0], axis=1),
                np.max(rotated_coords[:, :, 1], axis=1),
            )
            min_x, min_y = (
                np.clip(min_x, a_min=0, a_max=w),
                np.clip(min_y, a_min=0, a_max=h),
            )
            max_x, max_y = (
                np.clip(max_x, a_min=min_x, a_max=w),
                np.clip(max_y, a_min=min_y, a_max=h),
            )
            annotation['bbox'] = np.stack([min_x, min_y, max_x, max_y], axis=-1).astype(
                annotation['bbox'].dtype
            )

        return annotations

    def __repr__(self):
        repr_str = super().__repr__()
        return ("\n").join(
            repr_str.split("\n")[:-1]
            + [f"angle={self.angle}", f"center={self.center}", f"scale={self.scale}"]
            + repr_str.split("\n")[-1:]
        )


@TRANSFORMS.register()
class RandShear(GeometricAugmentation):
    def __init__(self, x=None, y=None, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.interpolation = interpolation
        if self.x is None and self.y is None:
            self.prob = 0.0

    def get_magnitude(self, img, annotations):
        magnitude = {}
        if self.random_magnitude:
            if isinstance(self.x, (list, tuple)):
                assert len(self.x) == 2
                x = np.random.random() * (self.x[1] - self.x[0]) + self.x[0]
                magnitude["x"] = x
            if isinstance(self.y, (list, tuple)):
                assert len(self.y) == 2
                y = np.random.random() * (self.y[1] - self.y[0]) + self.y[0]
                magnitude["y"] = y
        else:
            if self.x is not None:
                assert isinstance(self.x, (int, float))
                magnitude["x"] = self.x
            if self.y is not None:
                assert isinstance(self.y, (int, float))
                magnitude["y"] = self.y
        return magnitude

    def apply(self,img, annotations, x=None, y=None):
        if x is not None:
            # translate horizontally
            img, annotations = self._shear(img, annotations, np.tanh(-x * np.pi / 180))
        if y is not None:
            # translate veritically
            img, annotations = self._shear(img, annotations, np.tanh(y * np.pi / 180), direction="vertical")
        return img, annotations

    def _shear(self, img, annotations, magnitude, direction="horizontal"):
        if not self.record:
            from pdb import set_trace
            set_trace()
        if self.record:
            GTrans.apply(annotations[0], "shear", magnitude=magnitude, direction=direction)
        img = self._shear_img(img, magnitude, direction, interpolation=self.interpolation)
        annotations = self._shear_bboxes(img, annotations, magnitude, direction=direction)
        # fill_val defaultly 0 for BitmapMasks and None for PolygonMasks.
        return img, annotations
    def _shear_img(
        self, img, magnitude, direction="horizontal", interpolation="bilinear"
    ):
        """Shear the image.

        Args:
            results (dict): Result dict from loading pipeline.
            magnitude (int | float): The magnitude used for shear.
            direction (str): The direction for shear, either "horizontal"
                or "vertical".
            interpolation (str): Same as in :func:`mmcv.imshear`.
        """
        return imshear(
                img,
                magnitude,
                direction,
                border_value=self.img_fill_val,
                interpolation=interpolation,
            ).astype(img.dtype)

    def _shear_bboxes(self, img, annotations, magnitude, direction="horizontal"):
        """Shear the bboxes."""
        h, w, c = img.shape
        if direction == "horizontal":
            shear_matrix = np.stack([[1, magnitude], [0, 1]]).astype(
                np.float32
            )  # [2, 2]
        else:
            shear_matrix = np.stack([[1, 0], [magnitude, 1]]).astype(np.float32)
        for annotation in annotations:
            min_x, min_y, max_x, max_y = np.split(
                annotation['bbox'], annotation['bbox'].shape[-1], axis=-1
            )
            coordinates = np.stack(
                [[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]]
            )  # [4, 2, nb_box, 1]
            coordinates = coordinates.reshape(4,2,-1,1)
            coordinates = (
                coordinates[..., 0].transpose((2, 1, 0)).astype(np.float32)
            )  # [nb_box, 2, 4]
            new_coords = np.matmul(
                shear_matrix[None, :, :], coordinates
            )  # [nb_box, 2, 4]
            min_x = np.min(new_coords[:, 0, :], axis=-1)
            min_y = np.min(new_coords[:, 1, :], axis=-1)
            max_x = np.max(new_coords[:, 0, :], axis=-1)
            max_y = np.max(new_coords[:, 1, :], axis=-1)
            min_x = np.clip(min_x, a_min=0, a_max=w)
            min_y = np.clip(min_y, a_min=0, a_max=h)
            max_x = np.clip(max_x, a_min=min_x, a_max=w)
            max_y = np.clip(max_y, a_min=min_y, a_max=h)
            annotation['bbox'] = np.stack([min_x, min_y, max_x, max_y], axis=-1).astype(
                annotation['bbox'].dtype
            )

        return annotations
   

    def __repr__(self):
        repr_str = super().__repr__()
        return ("\n").join(
            repr_str.split("\n")[:-1]
            + [f"x_magnitude={self.x}", f"y_magnitude={self.y}"]
            + repr_str.split("\n")[-1:]
        )


@TRANSFORMS.register()
class RandErase(GeometricAugmentation):
    def __init__(
        self,
        n_iterations=None,
        size=None,
        squared: bool = True,
        patches=None,
        **kwargs,
    ):
        kwargs.update(min_size=None)
        super().__init__(**kwargs)
        self.n_iterations = n_iterations
        self.size = size
        self.squared = squared
        self.patches = patches

    def get_magnitude(self, img, annotations):
        magnitude = {}
        if self.random_magnitude:
            n_iterations = self._get_erase_cycle()
            patches = []
            h, w, c = img.shape
            for i in range(n_iterations):
                # random sample patch size in the image
                ph, pw = self._get_patch_size(h, w)
                # random sample patch left top in the image
                px, py = np.random.randint(0, w - pw), np.random.randint(0, h - ph)
                patches.append([px, py, px + pw, py + ph])
            magnitude["patches"] = patches
        else:
            assert self.patches is not None
            magnitude["patches"] = self.patches

        return magnitude

    def _get_erase_cycle(self):
        if isinstance(self.n_iterations, int):
            n_iterations = self.n_iterations
        else:
            assert (
                isinstance(self.n_iterations, (tuple, list))
                and len(self.n_iterations) == 2
            )
            n_iterations = np.random.randint(*self.n_iterations)
        return n_iterations

    def _get_patch_size(self, h, w):
        if isinstance(self.size, float):
            assert 0 < self.size < 1
            return int(self.size * h), int(self.size * w)
        else:
            assert isinstance(self.size, (tuple, list))
            assert len(self.size) == 2
            assert 0 <= self.size[0] < 1 and 0 <= self.size[1] < 1
            w_ratio = np.random.random() * (self.size[1] - self.size[0]) + self.size[0]
            h_ratio = w_ratio

            if not self.squared:
                h_ratio = (
                    np.random.random() * (self.size[1] - self.size[0]) + self.size[0]
                )
            return int(h_ratio * h), int(w_ratio * w)

    def apply(self, img,annotations, patches: list):
        for patch in patches:
            img = self._erase_image(img, patch, fill_val=self.img_fill_val)
        return img,annotations

    def _erase_image(self, img, patch, fill_val=128):
        tmp = img.copy()
        x1, y1, x2, y2 = patch
        tmp[y1:y2, x1:x2, :] = fill_val
        img = tmp
        return img

def check_dtype(img):
    """
    Check the image data type and dimensions to ensure that transforms can be applied on it.

    Args:
        img (np.array): image to be checked.
    """
    assert isinstance(
        img, np.ndarray
    ), "[TransformGen] Needs an numpy array, but got a {}!".format(type(img))
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[TransformGen] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [2, 3], img.ndim








@TRANSFORMS.register()
class RandomFlip(TransformGen):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError(
                "Cannot do both horiz and vert. Please use two Flip instead."
            )
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        self.flip = do
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()

@TRANSFORMS.register()
class RandFlip(RandomFlip):
    def __init__(self, record=False, **kwargs):
        super().__init__(**kwargs)
        self.record = record

    def __call__(self, img, annotations=None, **kwargs):
        img, annotations = super().__call__(img, annotations, **kwargs)
        if self.record:
            annotations[0]["flip"] = self.flip
            if "aug_info" not in annotations[0]:
                annotations[0]["aug_info"] = []
            if self.flip: 
                # from pdb import set_trace
                # set_trace()
                GTrans.apply(
                    annotations[0],
                    "flip",
                    direction='horizontal' if self.horizontal else 'vertical',
                    shape=img.shape[:2],
                )
                annotations[0]["aug_info"].append(
                    dict(
                        type=self.__class__.__name__,
                        record=self.record,
                        flip_ratio=1.0,
                        direction='horizontal' if self.horizontal else 'vertical',
                    )
                )
            else:
                annotations[0]["aug_info"].append(
                    dict(
                        type=self.__class__.__name__,
                        record=self.record,
                        flip_ratio=0.0,
                        direction="None",
                    )
                )
        return img, annotations


    def enable_record(self, mode: bool = True):
        self.record = mode


@TRANSFORMS.register()
class TorchTransformGen(TransformGen):
    """
    Wrapper transfrom of transforms in torchvision.
    It convert img (np.ndarray) to PIL image, and convert back to np.ndarray after transform.
    """
    def __init__(self, tfm):
        super().__init__()
        self.tfm = tfm

    def get_transform(self, img, annotations=None):
        return TorchTransform(self.tfm)


@TRANSFORMS.register()
class PILColorTransformGen(TransformGen):
    """
    Wrapper transfrom of transforms in torchvision.
    It convert img (np.ndarray) to PIL image, and convert back to np.ndarray after transform.
    """
    def __init__(self, tfm):
        super().__init__()
        self.tfm = tfm

    def get_transform(self, img, annotations=None):
        return PILColorTransform(self.tfm)


@TRANSFORMS.register()
class RandomFiveCrop(TransformGen):
    """
    Random select rand_in crops from the fixed 5 crops: center, left-top, right-top,
        left-bottom, and right-bottom.
    """
    def __init__(self, size, rand_in=[0, -1]):
        """
        Args:
            size (int): desired crop size.
            rand_in (List[int]): select crops according to indexes in rand_in, then random select
                from those crops.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return FiveCropTransform(self.size, self.rand_in)


@TRANSFORMS.register()
class RandomDistortion(TransformGen):
    """
    Random distort image's hue, saturation and exposure.
    """

    def __init__(self, hue, saturation, exposure, image_format="BGR"):
        """
        RandomDistortion Initialization.
        Args:
            hue (float): value of hue
            saturation (float): value of saturation
            exposure (float): value of exposure
        """
        assert image_format in ["RGB", "BGR"]
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return DistortTransform(
            self.hue, self.saturation, self.exposure, self.image_format)


@TRANSFORMS.register()
class CenterAffine(TransformGen):
    """
    Affine Transform for CenterNet
    """

    def __init__(self, boarder, output_size, pad_value=[0, 0, 0], random_aug=True):
        """
        output_size (w, h) shape
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        img_shape = img.shape[:2]
        center, scale = self.generate_center_and_scale(img_shape)
        src, dst = self.generate_src_and_dst(center, scale, self.output_size)
        return AffineTransform(src, dst, self.output_size, self.pad_value)

    @staticmethod
    def _get_boarder(boarder, size):
        """
        This func may be rewirite someday
        """
        i = 1
        size //= 2
        while size <= boarder // i:
            i *= 2
        return boarder // i

    def generate_center_and_scale(self, img_shape):
        """
        generate center
        shpae : (h, w)
        """
        height, width = img_shape
        center = np.array([width / 2, height / 2], dtype=np.float32)
        scale = float(max(img_shape))
        if self.random_aug:
            scale = scale * np.random.choice(np.arange(0.6, 1.4, 0.1))
            h_boarder = self._get_boarder(self.boarder, height)
            w_boarder = self._get_boarder(self.boarder, width)
            center[0] = np.random.randint(low=w_boarder, high=width - w_boarder)
            center[1] = np.random.randint(low=h_boarder, high=height - h_boarder)
        else:
            pass

        return center, scale

    @staticmethod
    def generate_src_and_dst(center, scale, output_size):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src_w = scale[0]
        src_dir = [0, src_w * -0.5]
        src[0, :] = center
        src[1, :] = src[0, :] + src_dir
        src[2, :] = src[1, :] + (src_dir[1], -src_dir[0])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst_w, dst_h = output_size
        dst_dir = [0, dst_w * -0.5]
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = dst[0, :] + dst_dir
        dst[2, :] = dst[1, :] + (dst_dir[1], -dst_dir[0])

        return src, dst


@TRANSFORMS.register()
class GaussianBlur(TransformGen):
    """
    Gaussian blur transform.
    """
    def __init__(self, sigma, p=1.0):
        """
        Args:
            sigma (List(float)): sigma of gaussian
            p (float): probability of perform this augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return GaussianBlurTransform(self.sigma, self.p)


@TRANSFORMS.register()
class Solarization(TransformGen):
    def __init__(self, threshold=128, p=0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return SolarizationTransform(self.threshold, self.p)


@TRANSFORMS.register()
class GaussianBlurConv(TransformGen):
    def __init__(self, kernel_size, p):
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return GaussianBlurConvTransform(self.kernel_size, self.p)


@TRANSFORMS.register()
class Resize(TransformGen):
    """
    Resize image to a target size
    """

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int.
            interp: PIL interpolation method.
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return ResizeTransform(
            img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.interp
        )




@TRANSFORMS.register()
class ResizeLongestEdge(TransformGen):
    """
    Scale the longer edge to the given size.
    """

    def __init__(self, long_edge_length, sample_style="range", interp=Image.BILINEAR,
                 jitter=(0.0, 32)):
        """
        Args:
            long_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            sample_style (str): either "range" or "choice".
            interp: PIL interpolation method.
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(long_edge_length, int):
            long_edge_length = (long_edge_length, long_edge_length)
        self._init(locals())

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        if self.is_range:
            size = np.random.randint(
                self.long_edge_length[0], self.long_edge_length[1] + 1
            )
        else:
            size = np.random.choice(self.long_edge_length)
        if size == 0:
            return NoOpTransform()

        if self.jitter[0] > 0:
            dw = self.jitter[0] * w
            dh = self.jitter[0] * h
            size = max(h, w) + np.random.uniform(low=-max(dw, dh), high=max(dw, dh))
            size -= size % self.jitter[1]

        scale = size * 1.0 / max(h, w)
        if h < w:
            newh, neww = scale * h, size
        else:
            newh, neww = size, scale * w

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        return ResizeTransform(h, w, newh, neww, self.interp)


@TRANSFORMS.register()
class ResizeShortestEdge(TransformGen):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self,
        short_edge_length,
        max_size=sys.maxsize,
        sample_style="range",
        interp=Image.BILINEAR,
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
            interp: PIL interpolation method.
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]

        if self.is_range:
            size = np.random.randint(
                self.short_edge_length[0], self.short_edge_length[1] + 1
            )
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(h, w)
            newh = h * scale
            neww = w * scale
        self.scale_factor = (scale,scale)
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        return ResizeTransform(h, w, newh, neww, self.interp)


@TRANSFORMS.register()
class RandResizeShortestEdge(ResizeShortestEdge):
    def __init__(self, record=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs #
        self.record = record

    def __call__(self, img, annotations=None, **kwargs):
        

        img, annotations = super().get_transform(img,annotations)(img, annotations)
        if self.record:
            
            GTrans.apply(annotations[0], "scale", sx=self.scale_factor[0], sy=self.scale_factor[1])

            if "aug_info" not in annotations[0]:
                annotations[0]["aug_info"] = []
            new_h, new_w = img.shape[:2]
            annotations[0]["aug_info"].append(
                dict(
                    type=self.__class__.__name__,
                    record=False,
                    img_scale=(new_w, new_h),
                    keep_ratio=False
                )
            )
        return img, annotations

    def enable_record(self, mode: bool = True):
        self.record = mode

@TRANSFORMS.register()
class RandomCrop(TransformGen):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self, crop_type: str, crop_size, strict_mode=True):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
            strict_mode (bool): if `True`, the target `crop_size` must be smaller than
                the original image size.
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute"]
        self._init(locals())

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        if self.strict_mode:
            assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(
                self
            )
        offset_range_h = max(h - croph, 0)
        offset_range_w = max(w - cropw, 0)
        h0 = np.random.randint(offset_range_h + 1)
        w0 = np.random.randint(offset_range_w + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return self.crop_size
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


@TRANSFORMS.register()
class RandomCropWithInstance(RandomCrop):
    """
    Make sure the cropping region contains the center of a random instance from annotations.
    """

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        if self.strict_mode:
            assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(
                self
            )
        offset_range_h = max(h - croph, 0)
        offset_range_w = max(w - cropw, 0)
        # Make sure there is always at least one instance in the image
        assert annotations is not None, "Can not get annotations infos."
        instance = np.random.choice(annotations)
        bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)
        bbox = torch.tensor(bbox)
        center_xy = (bbox[:2] + bbox[2:]) / 2.0

        offset_range_h_min = max(center_xy[1] - croph, 0)
        offset_range_w_min = max(center_xy[0] - cropw, 0)
        offset_range_h_max = min(offset_range_h, center_xy[1] - 1)
        offset_range_w_max = min(offset_range_w, center_xy[0] - 1)

        h0 = np.random.randint(offset_range_h_min, offset_range_h_max + 1)
        w0 = np.random.randint(offset_range_w_min, offset_range_w_max + 1)
        return CropTransform(w0, h0, cropw, croph)


@TRANSFORMS.register()
class RandomCropWithMaxAreaLimit(RandomCrop):
    """
    Find a cropping window such that no single category occupies more than
    `single_category_max_area` in `sem_seg`.

    The function retries random cropping 10 times max.
    """

    def __init__(self, crop_type: str, crop_size, strict_mode=True,
                 single_category_max_area=1.0, ignore_value=255):
        super().__init__(crop_type, crop_size, strict_mode)
        self._init(locals())

    def get_transform(self, img, annotations=None):
        if self.single_category_max_area >= 1.0:
            crop_tfm = super().get_transform(img, annotations)
        else:
            h, w = img.shape[:2]
            assert "sem_seg" in annotations[0]
            sem_seg = annotations[0]["sem_seg"]
            croph, cropw = self.get_crop_size((h, w))
            for _ in range(10):
                y0 = np.random.randint(h - croph + 1)
                x0 = np.random.randint(w - cropw + 1)
                sem_seg_temp = sem_seg[y0: y0 + croph, x0: x0 + cropw]
                labels, cnt = np.unique(sem_seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_value]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.single_category_max_area:
                    break
            crop_tfm = CropTransform(x0, y0, cropw, croph)
        return crop_tfm


@TRANSFORMS.register()
class RandomCropPad(RandomCrop):
    """
    Randomly crop and pad a subimage out of an image.
    """
    def __init__(self,
                 crop_type: str,
                 crop_size,
                 img_value=None,
                 seg_value=None):
        super().__init__(crop_type, crop_size, strict_mode=False)
        self._init(locals())

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        h0 = np.random.randint(h - croph + 1) if h >= croph else 0
        w0 = np.random.randint(w - cropw + 1) if w >= cropw else 0
        dh = min(h, croph)
        dw = min(w, cropw)
        # print(w0, h0, dw, dh)
        return CropPadTransform(w0, h0, dw, dh, cropw, croph, self.img_value,
                                self.seg_value)


@TRANSFORMS.register()
class RandomExtent(TransformGen):
    """
    Outputs an image by cropping a random "subrect" of the source image.

    The subrect can be parameterized to include pixels outside the source image,
    in which case they will be set to zeros (i.e. black). The size of the output
    image will vary with the size of the random subrect.
    """

    def __init__(self, scale_range, shift_range):
        """
        Args:
            scale_range (l, h): Range of input-to-output size scaling factor.
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        img_h, img_w = img.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

        # Apply a random scaling to the src_rect.
        src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(
                int(src_rect[3] - src_rect[1]),
                int(src_rect[2] - src_rect[0]),
            ),
        )


@TRANSFORMS.register()
class RandomContrast(TransformGen):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max, prob=1.0):
        """
        Args:
            intensity_min (float): Minimum augmentation.
            intensity_max (float): Maximum augmentation.
            prob (float): probability of transforms image contrast.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        do = self._rand_range() < self.prob
        if do:
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            return BlendTransform(src_image=img.mean(), src_weight=1 - w, dst_weight=w)
        else:
            return NoOpTransform()


@TRANSFORMS.register()
class RandomBrightness(TransformGen):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max, prob=1.):
        """
        Args:
            intensity_min (float): Minimum augmentation.
            intensity_max (float): Maximum augmentation.
            prob (float): probability of transforms image brightness.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        do = self._rand_range() < self.prob
        if do:
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)
        else:
            return NoOpTransform()


@TRANSFORMS.register()
class RandomSaturation(TransformGen):
    """
    Randomly transforms image saturation.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max, prob=1.0):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
            prob (float): probability of transforms image saturation.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        do = self._rand_range() < self.prob
        if do:
            assert img.shape[-1] == 3, "Saturation only works on RGB images"
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
            return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)
        else:
            return NoOpTransform()


@TRANSFORMS.register()
class RandomLighting(TransformGen):
    """
    Randomly transforms image color using fixed PCA over ImageNet.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self._init(locals())
        self.eigen_vecs = np.array(
            [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def get_transform(self, img, annotations=None):
        assert img.shape[-1] == 3, "Saturation only works on RGB images"
        weights = np.random.normal(scale=self.scale, size=3)
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals),
            src_weight=1.0,
            dst_weight=1.0,
        )


@TRANSFORMS.register()
class RandomSwapChannels(TransformGen):
    """
    Randomly swap image channels.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): probability of swap channels.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        _, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            return RandomSwapChannelsTransform()
        else:
            return NoOpTransform()


@TRANSFORMS.register()
class MinIoURandomCrop(TransformGen):
    """
    Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        """
        Args:
            min_ious (tuple): minimum IoU threshold for all intersections with bounding boxes
            min_crop_size (float): minimum crop's size
                (i.e. h,w := a*h, a*w, where a >= min_crop_size).
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations):
        """
        Args:
            img (ndarray): of shape HxWxC(RGB). The array can be of type uint8
                in range [0, 255], or floating point in range [0, 255].
            annotations (list[dict[str->str]]):
                Each item in the list is a bbox label of an object. The object is
                    represented by a dict,
                which contains:
                 - bbox (list): bbox coordinates, top left and bottom right.
                 - bbox_mode (str): bbox label mode, for example: `XYXY_ABS`,
                    `XYWH_ABS` and so on...
        """
        sample_mode = (1, *self.min_ious, 0)
        h, w = img.shape[:2]

        boxes = list()
        for obj in annotations:
            boxes.append(BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS))
        boxes = torch.tensor(boxes)

        while True:
            mode = np.random.choice(sample_mode)
            if mode == 1:
                return NoOpTransform()

            min_iou = mode
            for i in range(50):
                new_w = np.random.uniform(self.min_crop_size * w, w)
                new_h = np.random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = np.random.uniform(w - new_w)
                top = np.random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))

                overlaps = pairwise_iou(
                    Boxes(patch.reshape(-1, 4)),
                    Boxes(boxes.reshape(-1, 4))
                )

                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1])
                        * (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                if not mask.any():
                    continue
                return IoUCropTransform(int(left), int(top), int(new_w), int(new_h))


@TRANSFORMS.register()
class Expand(TransformGen):
    """
    Random Expand the image & bboxes.
    """

    def __init__(self, ratio_range=(1, 4), mean=(0, 0, 0), prob=0.5):
        """
        Args:
            ratio_range (tuple): range of expand ratio.
            mean (tuple): mean value of dataset.
            prob (float): probability of applying this transformation.
        """
        super().__init__()
        self._init(locals())
        self.min_ratio, self.max_ratio = ratio_range

    def get_transform(self, img, annotations=None):
        if np.random.uniform(0, 1) > self.prob:
            return NoOpTransform()
        h, w, c = img.shape
        ratio = np.random.uniform(self.min_ratio, self.max_ratio)
        left = int(np.random.uniform(0, w * ratio - w))
        top = int(np.random.uniform(0, h * ratio - h))
        return ExpandTransform(left, top, ratio, self.mean)


@TRANSFORMS.register()
class Lightning(TransformGen):
    """
    Lightning augmentation, used in classification.

    .. code-block:: python
        tfm = LightningTransform(
            alpha_std=0.1,
            eig_val=np.array([[0.2175, 0.0188, 0.0045]]),
            eig_vec=np.array([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203]
            ]),
        )
    """
    def __init__(self, alpha_std, eig_val, eig_vec):
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return LightningTransform(self.alpha_std, self.eig_val, self.eig_vec)


@TRANSFORMS.register()
class RandomScale(TransformGen):
    """
    Randomly scale the image according to the specified output size and scale ratio range.

    This transform has the following three steps:

        1. select a random scale factor according to the specified scale ratio range.
        2. recompute the accurate scale_factor using rounded scaled image size.
        3. select non-zero random offset (x, y) if scaled image is larger than output_size.
    """

    def __init__(self, output_size, ratio_range=(0.1, 2), interp="BILINEAR"):
        """
        Args:
            output_size (tuple): image output size.
            ratio_range (tuple): range of scale ratio.
            interp (str): the interpolation method. Options includes:
              * "NEAREST"
              * "BILINEAR"
              * "BICUBIC"
              * "LANCZOS"
              * "HAMMING"
              * "BOX"
        """
        super().__init__()
        self._init(locals())
        self.min_ratio, self.max_ratio = ratio_range
        if isinstance(self.output_size, int):
            self.output_size = [self.output_size] * 2

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        output_h, output_w = self.output_size

        # 1. Select a random scale factor.
        random_scale_factor = np.random.uniform(self.min_ratio, self.max_ratio)

        scaled_size_h = int(random_scale_factor * output_h)
        scaled_size_w = int(random_scale_factor * output_w)

        # 2. Recompute the accurate scale_factor using rounded scaled image size.
        image_scale_h = scaled_size_h * 1.0 / h
        image_scale_w = scaled_size_w * 1.0 / w
        image_scale = min(image_scale_h, image_scale_w)

        # 3. Select non-zero random offset (x, y) if scaled image is larger than output_size.
        scaled_h = int(h * 1.0 * image_scale)
        scaled_w = int(w * 1.0 * image_scale)

        return ScaleTransform(h, w, scaled_h, scaled_w, self.interp)


@TRANSFORMS.register()
class AutoAugment(TransformGen):
    """
    Convert any of AutoAugment into a cvpods-fashion Transform such that can be configured in
        config.py
    """
    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        """
        Args:
            name (str): any type of transforms list in _RAND_TRANSFORMS.
            prob (float): probability of perform current augmentation.
            magnitude (int): intensity / magnitude of each augmentation.
            hparams (dict): hyper-parameters required by each augmentation.
        """

        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return AutoAugmentTransform(self.name, self.prob, self.magnitude, self.hparams)

PARAMETER_MAX = 10

@TRANSFORMS.register()
class RandAutoAugment(AutoAugment):

    def __init__(self, name, prob=1, magnitude=10, hparams=None,magnitude_limit=10,record=False):
        super().__init__(name,prob,magnitude)
        assert 0 <= prob <= 1, f"probability should be in (0,1) but get {prob}"
        assert (
            magnitude <= PARAMETER_MAX
        ), f"magnitude should be small than max value {PARAMETER_MAX} but get {magnitude}"
        self._init(locals())

    def get_transform(self, img, annotations=None):
        if np.random.random() < self.prob:
            magnitude = self.magnitude
            magnitude = np.random.randint(1, magnitude)

            if self.record:
                if "aug_info" not in annotations[0]:
                    annotations[0]["aug_info"] = []
                annotations[0]["aug_info"].append(self.get_aug_info(magnitude=magnitude))
        return AutoAugmentTransform(self.name, self.prob, magnitude, self.hparams)
    
    def get_aug_info(self, **kwargs):
        aug_info = dict(type=self.__class__.__name__)
        aug_info.update(
            dict(
                prob=1.0,
                random_magnitude=False,
                record=False,
                magnitude=self.magnitude,
                name=self.name
            )
        )
        aug_info.update(kwargs)
        return aug_info

    def enable_record(self, mode: bool = True):
        self.record = mode

@TRANSFORMS.register()
class Pad(TransformGen):
    """
    Pad image with `pad_value` to the specified `target_h` and `target_w`.

    Adds `top` rows of `pad_value` on top, `left` columns of `pad_value` on the left,
    and then pads the image on the bottom and right with `pad_value` until it has
    dimensions `target_h`, `target_w`.

    This op does nothing if `top` and `left` is zero and the image already has size
    `target_h` by `target_w`.
    """

    def __init__(self, top, left, target_h, target_w, pad_value=0):
        """
        Args:
            top (int): number of rows of `pad_value` to add on top.
            left (int): number of columns of `pad_value` to add on the left.
            target_h (int): height of output image.
            target_w (int): width of output image.
            pad_value (int): the value used to pad the image.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return PadTransform(self.top, self.left, self.target_h, self.target_w, self.pad_value)


@TRANSFORMS.register()
class RandomList(TransformGen):
    """
    Random select subset of provided augmentations.
    """
    def __init__(self, transforms, num_layers=2, choice_weights=None):
        """
        Args:
            transforms (List[TorchTransformGen]): list of transforms need to be performed.
            num_layers (int): parameters of np.random.choice.
            choice_weights (optional, float): parameters of np.random.choice.
        """
        self.transforms = transforms
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def get_transform(self, img, annotations=None):
        tfms = np.random.choice(
            self.transforms,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights)
        return ComposeTransform(tfms)


@TRANSFORMS.register()
class ShuffleList(TransformGen):
    """
    Randomly shuffle the `transforms` order.
    """

    def __init__(self, transforms, record: bool = False):
        """
        Args:
            transforms (list[TransformGen]): List of transform to be shuffled.
        """
        super().__init__()
        self.transforms = transforms
        self.record = record
        self.enable_record(record)
    def get_transform(self, img, annotations=None):
        np.random.shuffle(self.transforms)
        return ComposeTransform(self.transforms)

    def enable_record(self, mode: bool = True):
        # enable children to record
        self.record = mode
        for transform in self.transforms:
            transform.enable_record(mode)

@TRANSFORMS.register()
class OrderList(TransformGen):
    def __init__(self, transforms, record: bool = False):
        self.transforms = transforms
        self.record = record
        self.enable_record(record)
    def get_transform(self, img, annotations=None):
        return ComposeTransform(self.transforms)

    def enable_record(self, mode: bool = True):
        # enable children to record
        self.record = mode
        for transform in self.transforms:
            transform.enable_record(mode)

# @TRANSFORMS.register()
# class ShuffledOrderList(OrderList):
#     def get_transform(self, img, annotations=None):
#         order = np.random.permutation(len(self.transforms))
#         transform = [self.transforms[idx] for idx in order]
#         return ComposeTransform(transform)

@TRANSFORMS.register()
class OneOf(TransformGen):
    def __init__(self, transforms, record: bool = False):
        self.transforms = transforms
        self.record = record
        self.enable_record(record)

    def get_transform(self, img, annotations=None):
        pass

    def __call__(self, img, annotations=None, **kwargs):
        transform = np.random.choice(self.transforms)
        return transform(img, annotations, **kwargs)

    def enable_record(self, mode: bool = True):
        # enable children to record
        self.record = mode
        for transform in self.transforms:

            transform.enable_record(mode)

@TRANSFORMS.register()
class RepeatListv2(TransformGen):
    def __init__(self, transforms, repeat_times, ranges=[(0, 1)]):
        super().__init__()
        self.transforms = transforms
        self.times = repeat_times
        self.iou_ranges = ranges

    def get_transform(self, img, annotations=None):
        return ComposeTransform(self.transforms).transforms

    def __call__(self, img, annotations=None,**kwargs):
        repeat_imgs = []
        repeat_annotations = []

        for idx, tfm in enumerate(self.transforms):
            if isinstance(tfm.tfm, torchvision.transforms.RandomResizedCrop):
                size = tfm.tfm.size
                scale = tfm.tfm.scale
                ratio = tfm.tfm.ratio
                interp = tfm.tfm.interpolation

                width, height = get_image_size(img)
                area = height * width

                satisfied = False
                cnt = 0
                while not satisfied and cnt < 50:
                    cnt += 1
                    crop_params = []
                    for t in range(self.times):
                        params = None
                        for _ in range(10):
                            target_area = random.uniform(*scale) * area
                            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
                            aspect_ratio = math.exp(random.uniform(*log_ratio))

                            w = int(round(math.sqrt(target_area * aspect_ratio)))
                            h = int(round(math.sqrt(target_area / aspect_ratio)))

                            if 0 < w <= width and 0 < h <= height:
                                i = random.randint(0, height - h)
                                j = random.randint(0, width - w)
                                params = [i, j, h, w, size, interp]
                                break
                        if not params:
                            # Fallback to central crop
                            in_ratio = float(width) / float(height)
                            if (in_ratio < min(ratio)):
                                w = width
                                h = int(round(w / min(ratio)))
                            elif (in_ratio > max(ratio)):
                                h = height
                                w = int(round(h * max(ratio)))
                            else:  # whole image
                                w = width
                                h = height
                            i = (height - h) // 2
                            j = (width - w) // 2
                            params = [i, j, h, w, size, interp]
                        crop_params.append(params)

                    if self.times == 2:
                        boxes = [
                            [param[0], param[1], param[0] + param[2], param[1] + param[3]]
                            for param in crop_params]
                        iou = bb_intersection_over_union(boxes[0], boxes[1])
                        for rg in self.iou_ranges:
                            if iou >= rg[0] and iou <= rg[1]:
                                satisfied = True
                                break
                    else:
                        satisfied = True
                break
        else:
            img, annotations = tfm(img, annotations)

        for crop_param in crop_params:
            tmp_img = np.array(F.resized_crop(Image.fromarray(img), *crop_param))
            tmp_annotations = annotations
            for tfm in self.transforms[idx + 1:]:
                tmp_img, tmp_annotations = tfm(tmp_img, tmp_annotations)

            repeat_imgs.append(tmp_img)
            repeat_annotations.append(tmp_annotations)

        repeat_imgs = np.stack(repeat_imgs, axis=0)

        return repeat_imgs, repeat_annotations


@TRANSFORMS.register()
class RepeatList(TransformGen):
    """
    Forward several times of provided transforms for a given image.
    """
    def __init__(self, transforms, repeat_times):
        """
        Args:
            transforms (list[TransformGen]): List of transform to be repeated.
            repeat_times (int): number of duplicates desired.
        """
        super().__init__()
        self.transforms = transforms
        self.times = repeat_times

    def get_transform(self, img, annotations=None):
        return ComposeTransform(self.transforms)

    def __call__(self, img, annotations=None, **kwargs):
        repeat_imgs = []
        repeat_annotations = []
        for t in range(self.times):
            tmp_img, tmp_anno = self.get_transform(img)(img, annotations, **kwargs)
            repeat_imgs.append(tmp_img)
            repeat_annotations.append(tmp_anno)
        repeat_imgs = np.stack(repeat_imgs, axis=0)
        return repeat_imgs, repeat_annotations


@TRANSFORMS.register()
class JigsawCrop(TransformGen):
    """
    Crop an image into n_grid times n_grid crops, no overlaps.
    """
    def __init__(self, n_grid=3, img_size=255, crop_size=64):
        """
        Args:
            n_grid (int): crop image into n_grid x n_grid crops.
            img_size (int): image scale size.
            crop_size (int): crop size.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return JigsawCropTransform(self.n_grid, self.img_size, self.crop_size)

    def __call__(self, img, annotations=None, **kwargs):
        crops, annos = self.get_transform(img)(img, annotations, **kwargs)
        return np.stack(crops, axis=0), annos

@TRANSFORMS.register()
class ColorJiter(TransformGen):

    def __init__(self,scale_factor=1, backend='cv2'):
        self._init(locals())

    def __call__(self, img, annotations=None,**kwargs):
        s = self.scale_factor

        color_jitter = tfs.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        color_jitter_img = tfs.RandomApply([color_jitter], p=0.8)(Image.fromarray(img))

        return np.array(color_jitter_img), annotations


    def get_transform(self, img, annotations=None):
        pass

    def enable_record(self,mode=False):
        pass