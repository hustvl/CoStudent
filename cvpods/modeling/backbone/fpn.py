#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2019-2021 Megvii Inc. All rights reserved.
import math

import torch.nn.functional as F
from torch import nn

from cvpods.layers import Conv2d, ShapeSpec, get_norm
from cvpods.modeling.nn_utils import weight_init
from cvpods.utils.compat_wrapper import deprecated

from .backbone import Backbone
from .mobilenet import build_mobilenetv2_backbone
from .resnet import build_resnet_backbone
from .shufflenet import build_shufflenetv2_backbone
from .timm_backbone import build_timm_backbone

__all__ = [
    "_assert_strides_are_log2_contiguous",
    "build_fpn_backbone",
    "build_retinanet_fpn_backbone",
    "build_resnet_fpn_backbone",
    "build_shufflenetv2_fpn_backbone",
    "build_mobilenetv2_fpn_backbone",
    "build_timm_fpn_backbone",
    "build_retinanet_resnet_fpn_backbone",
    "build_retinanet_mobilenetv2_fpn_backbone",
    "build_retinanet_timm_fpn_backbone",
    "build_mobilenetv2_fpn_backbone",
    "build_retinanet_mobilenetv2_fpn_backbone",
    "FPN",
    "LastLevelP6P7"
]


class FPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(in_strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in Retinanet and follow-up network to generate extra layers
    P6 and P7 from C5/P5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        """
        Args:
            in_feature: input feature name, e.g. "res5" stands for C5 features,
                "p5" stands for P5 feature.
        """
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


# TODO refine backbone name into cfg
def build_fpn_backbone(cfg, input_shape: ShapeSpec, backbone_name="resnet"):
    """
    Args:
        cfg: a cvpods CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if backbone_name == "resnet":
        bottom_up = build_resnet_backbone(cfg, input_shape)
    elif backbone_name == "shufflev2":
        bottom_up = build_shufflenetv2_backbone(cfg, input_shape)
    elif backbone_name == "mobilev2":
        bottom_up = build_mobilenetv2_backbone(cfg, input_shape)
    elif backbone_name == "timm":
        bottom_up = build_timm_backbone(cfg, input_shape)
    else:
        raise ValueError("No such backbone: {}".format(backbone_name))

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


def build_retinanet_fpn_backbone(cfg, input_shape: ShapeSpec, backbone_name="resnet"):
    """
    Args:
        cfg: a cvpods CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if backbone_name == "resnet":
        bottom_up = build_resnet_backbone(cfg, input_shape)
    elif backbone_name == "shufflev2":
        bottom_up = build_shufflenetv2_backbone(cfg, input_shape)
    elif backbone_name == "mobilev2":
        bottom_up = build_mobilenetv2_backbone(cfg, input_shape)
    elif backbone_name == "timm":
        bottom_up = build_timm_backbone(cfg, input_shape)
    else:
        raise ValueError("No such backbone: {}".format(backbone_name))

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    block_in_feature = cfg.MODEL.FPN.BLOCK_IN_FEATURES
    if block_in_feature == "p5":
        in_channels_p6p7 = out_channels
    else:
        assert block_in_feature in bottom_up.output_shape()
        in_channels_p6p7 = bottom_up.output_shape()[block_in_feature].channels

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels, in_feature=block_in_feature),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


def build_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    return build_fpn_backbone(cfg, input_shape)


def build_shufflenetv2_fpn_backbone(cfg, input_shape: ShapeSpec):
    return build_fpn_backbone(cfg, input_shape, "shufflev2")


def build_mobilenetv2_fpn_backbone(cfg, input_shape: ShapeSpec):
    return build_fpn_backbone(cfg, input_shape, "mobilev2")


def build_timm_fpn_backbone(cfg, input_shape: ShapeSpec):
    return build_fpn_backbone(cfg, input_shape, "timm")


def build_retinanet_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    return build_retinanet_fpn_backbone(cfg, input_shape)


def build_retinanet_mobilenetv2_fpn_backbone(cfg, input_shape: ShapeSpec):
    return build_retinanet_fpn_backbone(cfg, input_shape, "mobilev2")


def build_retinanet_timm_fpn_backbone(cfg, input_shape: ShapeSpec):
    return build_retinanet_fpn_backbone(cfg, input_shape, "timm")


@deprecated("Use build_retinanet_resnet_fpn_backbone instead.")
def build_retinanet_resnet_fpn_p5_backbone(cfg, input_shape: ShapeSpec):
    return build_retinanet_resnet_fpn_backbone(cfg, input_shape)
