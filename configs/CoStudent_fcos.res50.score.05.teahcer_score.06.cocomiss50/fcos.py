from email.mime import base
import logging
import math
from typing import List
import torch
from cvpods.modeling.losses import sigmoid_focal_loss_jit, iou_loss
from torch import nn
import torch.nn.functional as F
import os
import cv2
from cvpods.layers import ShapeSpec, batched_nms, cat
from cvpods.structures import Boxes, ImageList, Instances, pairwise_iou
from cvpods.utils import log_first_n
from cvpods.data.build import build_dataset
from cvpods.modeling.box_regression import Shift2BoxTransform
from cvpods.modeling.postprocessing import detector_postprocess
import numpy as np
from pycocotools.coco import COCO
import json
from cvpods.structures import BoxMode





def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):


    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def Revision_GT(Instance1,gt_Instance2):
    '''
    Based on Instance1 to revise Instance2,
    we revise gt_Instance2 by ious,labels when one of Instances dont have scores,
    Namely,(Instance1[pred_output1],gt_Instance2[gt_info]),depending on their ious,labels

    '''
    assert Instance1.image_size == gt_Instance2.image_size
    image_size = Instance1.image_size

    refine_gt_Instance = Instances(tuple(image_size))
    missing_Instance = Instances(tuple(image_size))

    bboxes1 = Instance1.pred_boxes.tensor
    scores1 = Instance1.scores
    classes1 = Instance1.pred_classes

    bboxes2 = gt_Instance2.gt_boxes.tensor.clone()
    classes2 = gt_Instance2.gt_classes.clone()

    ious = bbox_overlaps(bboxes1,bboxes2)

    while(True):
        refine_gt_inds =  (ious>=0.5).any(dim=0)

        if torch.where(refine_gt_inds)[0].numel() == 0:
            break

        refine_inds = ious.max(dim=0)[1]

        refine_pred_classes = classes1[refine_inds]
        refine_pred_classes = refine_pred_classes == classes2

        diff_labels_inds = (~refine_gt_inds | ~refine_pred_classes) & refine_gt_inds

        diff_labels_inds0 = torch.where(diff_labels_inds)[0]

        if diff_labels_inds0.numel()>0:
            index = [refine_inds[diff_labels_inds],diff_labels_inds0]

            input_zeros = torch.zeros((diff_labels_inds0.numel())).to(ious.device)

            ious.index_put_(index,input_zeros)

        else:
            break

    if torch.where(refine_gt_inds)[0].numel() > 0:
        refine_inds = refine_inds[refine_gt_inds]
        refine_gt_inds_repeat = torch.where(refine_gt_inds)[0].reshape(-1,1).repeat(1,4)
        bboxes2.scatter_(dim=0,index=refine_gt_inds_repeat,src=bboxes1[refine_inds])

    refine_gt_Instance.gt_boxes = Boxes((bboxes2+bboxes2.abs())/2)
    refine_gt_Instance.gt_classes = classes2


    missing_inds = (ious<0.5).all(dim=1)
    missing_Instance.pred_boxes = Boxes(bboxes1[missing_inds])
    missing_Instance.pred_classes = classes1[missing_inds]
    missing_Instance.scores = scores1[missing_inds]

    return missing_Instance,refine_gt_Instance


def Revision_PRED(Instance1,Instance2):
    '''
    Based on Instance1 to revise Instance2,
    we revise Instance2 by scores and ious when both of Instances have scores,
    Namely,(Instance1[pred_output1],Instance2[pred_output2]),depending on their scores and ious

    '''
    assert Instance1.image_size == Instance2.image_size
    image_size = Instance1.image_size

    refine_gt_Instance = Instances(tuple(image_size))
    missing_Instance = Instances(tuple(image_size))

    bboxes1 = Instance1.pred_boxes.tensor
    scores1 = Instance1.scores
    classes1 = Instance1.pred_classes

    bboxes2 = Instance2.pred_boxes.tensor.clone()
    scores2 = Instance2.scores.clone()
    classes2 = Instance2.pred_classes.clone()

    ious = bbox_overlaps(bboxes1,bboxes2)

    if len(ious)==0 or len(ious[0])==0:
        return Instances.cat([Instance1,Instance2])

    while(True):

        refine_gt_inds = (ious > 0.5).any(dim=0)
        refine_inds = ious.max(dim=0)[1]

        refine_pred_scores = scores1[refine_inds]
        need_refine = refine_pred_scores >= scores2

        lower_scores_inds = (~refine_gt_inds | ~need_refine) & refine_gt_inds

        lower_scores_inds0 = torch.where(lower_scores_inds)[0]

        if lower_scores_inds0.numel()>0:
            index = [refine_inds[lower_scores_inds],lower_scores_inds0]

            input_zeros = torch.zeros((lower_scores_inds0.numel())).to(ious.device) + 0.5

            ious.index_put_(index,input_zeros)

        else:
            break

    refine_inds = refine_inds[refine_gt_inds]
    refine_gt_inds = torch.where(refine_gt_inds)[0]
    refine_gt_inds_repeat = refine_gt_inds.reshape(-1,1).repeat(1,4)

    bboxes2.scatter_(dim=0,index=refine_gt_inds_repeat,src=bboxes1[refine_inds])
    classes2.scatter_(dim=0,index=refine_gt_inds,src=classes1[refine_inds])
    scores2.scatter_(dim=0,index=refine_gt_inds,src=scores1[refine_inds])

    refine_gt_Instance.pred_boxes = Boxes((bboxes2+bboxes2.abs())/2)
    refine_gt_Instance.pred_classes = classes2
    refine_gt_Instance.scores = scores2


    missing_inds = (ious<0.5).all(dim=1)
    missing_Instance.pred_boxes = Boxes(bboxes1[missing_inds])
    missing_Instance.pred_classes = classes1[missing_inds]
    missing_Instance.scores = scores1[missing_inds]

    return Instances.cat([missing_Instance,refine_gt_Instance])


def bbox2points(box):
    min_x, min_y, max_x, max_y = torch.split(box[:, :4], [1, 1, 1, 1], dim=1)

    return torch.cat(
                [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dim=1
            ).reshape(
                -1, 2
            )  # n*4,2

def points2bbox(point, max_w, max_h):
    point = point.reshape(-1, 4, 2)
    if point.size()[0] > 0:
        min_xy = point.min(dim=1)[0]
        max_xy = point.max(dim=1)[0]
        xmin = min_xy[:, 0].clamp(min=0, max=max_w)
        ymin = min_xy[:, 1].clamp(min=0, max=max_h)
        xmax = max_xy[:, 0].clamp(min=0, max=max_w)
        ymax = max_xy[:, 1].clamp(min=0, max=max_h)
        min_xy = torch.stack([xmin, ymin], dim=1)
        max_xy = torch.stack([xmax, ymax], dim=1)
        return torch.cat([min_xy, max_xy], dim=1)  # n,4
    else:
        return point.new_zeros(0, 4)

def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat_retina(box_cls,
                                                  box_delta,
                                                  num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


def permute_all_cls_and_box_to_N_HWA_K_and_concat_fcos(box_cls,
                                                  box_delta,
                                                  box_center,
                                                  num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_delta and the centerness
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    box_center = cat(box_center_flattened, dim=1).view(-1, 1)
    return box_cls, box_delta, box_center


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FCOS(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1708.02002).
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.pseudo_score_thres = cfg.MODEL.FCOS.PSEUDO_SCORE_THRES
        self.matching_iou_thres = cfg.MODEL.FCOS.MATCHING_IOU_THRES

        # Inference parameters:
        self.score_threshold = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = FCOSHead(cfg, feature_shapes)
        self.shift_generator = cfg.build_shift_generator(cfg, feature_shapes)

        # Matching and loss
        self.shift2box_transform = Shift2BoxTransform(
            weights=cfg.MODEL.FCOS.BBOX_REG_WEIGHTS)
        self.object_sizes_of_interest = cfg.MODEL.FCOS.OBJECT_SIZES_OF_INTEREST

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.in_normalizer = lambda x: x * pixel_std + pixel_mean
        self.to(self.device)


        
    def freeze_all(self):
        for module in self.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                value.requires_grad = False

   
    def forward(self, batched_inputs,teacher_model=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """

        images_raw, images_nor,images_str = self.preprocess_image(batched_inputs,teacher_model)
        
        if self.training:
            gt_instances_raw = [x["instances_Raw"].to(self.device) for x in batched_inputs]
            gt_instances_nor = [x["instances_Nor"].to(self.device) for x in batched_inputs]
            gt_instances_str = [x["instances_Str"].to(self.device) for x in batched_inputs]
            
            features_raw = teacher_model.backbone(images_raw.tensor)
            features_nor = self.backbone(images_nor.tensor) 
            features_str = self.backbone(images_str.tensor)

            features_raw = [features_raw[f] for f in self.in_features]
            features_nor = [features_nor[f] for f in self.in_features]
            features_str = [features_str[f] for f in self.in_features]

            shifts_raw = self.shift_generator(features_raw)
            shifts_nor = self.shift_generator(features_nor)
            shifts_str = self.shift_generator(features_str)

            TM_Raw = [x["transform_matrix_Raw"] for x in batched_inputs]
            TM_Nor = [x["transform_matrix_Nor"] for x in batched_inputs]
            TM_Str = [x["transform_matrix_Str"] for x in batched_inputs]

        else:
            gt_instances_raw = None
            gt_instances_nor = [x["instances"].to(self.device) for x in batched_inputs]
            gt_instances_str = None
            features_raw = None
            features_nor = self.backbone(images_nor.tensor)
            features_nor = [features_nor[f] for f in self.in_features]
            features_str = None
            shifts_raw = None
            shifts_nor = self.shift_generator(features_nor)
            shifts_str = None
            TM_Raw = None
            TM_Nor = None
            TM_Str = None
        
        cls, delta, center, cls_aug, delta_aug, center_aug = self.head(features_nor, features_str)
        
        if self.training:
            with torch.no_grad():
                head1_preds = self.pseudo_gt_generate(
                    [ar.clone() for ar in cls],
                    [br.clone() for br in delta],
                    [cr.clone() for cr in center],
                    shifts_nor,
                    images_nor
                )
                head2_preds = self.pseudo_gt_generate(
                    [ar.clone() for ar in cls_aug],
                    [br.clone() for br in delta_aug],
                    [cr.clone() for cr in center_aug],
                    shifts_str,
                    images_str
                )

                if teacher_model:

                    missing_Instances,refine_gt_Instances,high_score_Instances = self.teacher_revision(
                                    teacher_model,features_raw,gt_instances_raw,shifts_raw,images_raw)

                    high_score_Instances_to_nor = self.cvt_Instance_list(TM_Raw,high_score_Instances,TM_Nor,images_nor.image_sizes)
                    high_score_Instances_to_str = self.cvt_Instance_list(TM_Raw,high_score_Instances,TM_Str,images_str.image_sizes)

                    head1_preds_new = [Revision_PRED(missing_Instance,head1_pred) for missing_Instance,head1_pred in zip(high_score_Instances_to_nor,head1_preds)]
                    head2_preds_new = [Revision_PRED(missing_Instance,head2_pred) for missing_Instance,head2_pred in zip(high_score_Instances_to_str,head2_preds)]
                    
                else:
                    head1_preds_new = head1_preds
                    head2_preds_new = head2_preds
                gt_instances1 = self.merge_ground_truth(gt_instances_nor, head2_preds_new, images_nor, self.matching_iou_thres,TM_Str,TM_Nor)
                gt_instances2 = self.merge_ground_truth(gt_instances_str, head1_preds_new, images_str, self.matching_iou_thres,TM_Nor,TM_Str)

                gt_classes_fcos1, gt_shifts_reg_deltas1, gt_centerness1,valid_img_1 = self.get_fcos_ground_truth(
                    shifts_nor, gt_instances1)

                gt_classes_fcos2, gt_shifts_reg_deltas2, gt_centerness2,valid_img_2 = self.get_fcos_ground_truth(
                    shifts_str, gt_instances2)

            fcos_losses1 = self.fcos_losses1(gt_classes_fcos1, gt_shifts_reg_deltas1, gt_centerness1,
                                           cls, delta, center,valid_img_1)

            fcos_losses2 = self.fcos_losses2(gt_classes_fcos2, gt_shifts_reg_deltas2, gt_centerness2,
                                           cls_aug, delta_aug, center_aug,valid_img_2)

            return dict(fcos_losses1, **fcos_losses2)

        else:
            results = self.inference(cls, delta, center, shifts_nor, images_nor)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images_nor.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    @torch.no_grad()
    def merge_ground_truth(self, targets, predictions, images, iou_thresold,source_TM,target_TM):

        new_targets = []

        for targets_per_image, predictions_per_image, image,source_TM_i,target_TM_i in zip(targets, predictions, images,source_TM,target_TM):
            image_size = image.shape[1:3]

            if len(predictions_per_image.pred_boxes) != 0:
                predictions_per_image_cvt = self.cvt_bbox(source_TM_i,
                                                            predictions_per_image.pred_boxes.tensor,
                                                            target_TM_i,
                                                            image.shape,
                                                            predictions_per_image.pred_classes,
                                                            predictions_per_image.scores
                                                            )
            else:
                predictions_per_image_cvt = predictions_per_image

            iou_matrix = pairwise_iou(targets_per_image.gt_boxes,
                                      predictions_per_image_cvt.pred_boxes)
            iou_filter = iou_matrix > iou_thresold

            target_class_list = (targets_per_image.gt_classes).reshape(-1, 1)
            pred_class_list = (predictions_per_image_cvt.pred_classes).reshape(1, -1)
            class_filter = target_class_list == pred_class_list

            final_filter = iou_filter & class_filter
            unlabel_idxs = torch.sum(final_filter, 0) == 0

            new_target = Instances(image_size)
            new_target.gt_boxes = Boxes.cat([targets_per_image.gt_boxes,
                                             predictions_per_image_cvt.pred_boxes[unlabel_idxs]])
            new_target.gt_classes = torch.cat([targets_per_image.gt_classes,
                                               predictions_per_image_cvt.pred_classes[unlabel_idxs]])
            new_targets.append(new_target)

        return new_targets





    def cvt_bbox(self,source_TM,source_bbox,target_TM,target_img_shape,labels,scores):
        
        source_points = bbox2points(source_bbox[:, :4])
        source_points = torch.cat(
                    [source_points, source_points.new_ones(source_points.shape[0], 1)], dim=1
                )
        M_T = np.matmul(target_TM,np.linalg.inv(source_TM))
        M_T = torch.tensor(M_T).to(source_bbox.device).float()
        target_points = (M_T @ source_points.t()).t()
        _,H,W = target_img_shape
        target_points = target_points[:, :2] / target_points[:, 2:3]
        target_bboxes = points2bbox(target_points, W, H)
        minx,min_y,max_x,max_y = target_bboxes[:,0],target_bboxes[:,1],target_bboxes[:,2],target_bboxes[:,3]
        valid_bboxes = (minx < max_x) & (min_y < max_y)

        target = Instances(target_img_shape[1:3])
        target.pred_boxes = Boxes(target_bboxes[valid_bboxes])
        target.scores = scores[valid_bboxes]
        target.pred_classes = labels[valid_bboxes]
        return target

    def cvt_Instance_list(self,source_TM,source_Instance,target_TM,target_img_shapes):
        assert len(source_TM) == len(source_Instance) == len(target_TM) == len(target_img_shapes)
        target_Instances = []
        if source_Instance[0].has('pred_boxes'):
            bbox_key = 'pred_boxes'
        else:
            bbox_key = 'gt_boxes'
        if source_Instance[0].has('pred_boxes'):
            class_key = 'pred_classes'
        else:
            class_key = 'gt_classes' 

        for i,(instance,source_tm,target_tm,target_img_shape) in enumerate(zip(source_Instance,source_TM,target_TM,target_img_shapes)):
            source_boxes = instance.get(bbox_key).tensor
            
            if len(source_boxes)!=0:
                source_points = bbox2points(source_boxes[:, :4])
                source_points = torch.cat(
                            [source_points, source_points.new_ones(source_points.shape[0], 1)], dim=1
                        )
                M_T = np.matmul(target_tm,np.linalg.inv(source_tm))
                M_T = torch.tensor(M_T).to(source_boxes.device).float()
                target_points = (M_T @ source_points.t()).t()
                H,W = target_img_shape
                target_points = target_points[:, :2] / target_points[:, 2:3]
                target_bboxes = points2bbox(target_points, W, H)
                minx,min_y,max_x,max_y = target_bboxes[:,0],target_bboxes[:,1],target_bboxes[:,2],target_bboxes[:,3]
                valid_bboxes = (minx < max_x) & (min_y < max_y)

                target = Instances(target_img_shape)
                target.set(bbox_key,Boxes(target_bboxes[valid_bboxes]))
                if instance.has('scores'):
                    target.scores = instance.scores[valid_bboxes]
                target.set(class_key,instance.get(class_key)[valid_bboxes])
            else:
                target = instance
    
            target_Instances.append(target)

        return target_Instances

    def fcos_losses1(self, gt_classes, gt_shifts_deltas, gt_centerness,
                    pred_class_logits, pred_shift_deltas, pred_centerness,valid_img):
        """
        Args:
            For `gt_classes`, `gt_shifts_deltas` and `gt_centerness` parameters, see
                :meth:`FCOS.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of shifts across levels, i.e. sum(Hi x Wi)
            For `pred_class_logits`, `pred_shift_deltas` and `pred_centerness`, see
                :meth:`FCOSHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_shift_deltas, pred_centerness = \
            permute_all_cls_and_box_to_N_HWA_K_and_concat_fcos(
                pred_class_logits, pred_shift_deltas, pred_centerness,
                self.num_classes
            )  # Shapes: (N x R, K) and (N x R, 4), respectively.
        
        valid_img = (valid_img>0).reshape(gt_classes.shape[0],-1).repeat(1,gt_classes.shape[1]).flatten()
        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)
        gt_centerness = gt_centerness.view(-1, 1)

        valid_idxs = (gt_classes >= 0) & valid_img
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground)

        # regression loss
        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            gt_centerness[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.iou_loss_type,
            reduction="mean",
        )

        # centerness loss
        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_centerness[foreground_idxs],
            gt_centerness[foreground_idxs],
            reduction="sum",
        ) / max(1, num_foreground)

        return {
            "fcos_cls_loss1": loss_cls,
            "fcos_reg_loss1": loss_box_reg,
            "fcos_center_loss1": loss_centerness
        }

    def fcos_losses2(self, gt_classes, gt_shifts_deltas, gt_centerness,
                    pred_class_logits, pred_shift_deltas, pred_centerness,valid_img):
        """
        Args:
            For `gt_classes`, `gt_shifts_deltas` and `gt_centerness` parameters, see
                :meth:`FCOS.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of shifts across levels, i.e. sum(Hi x Wi)
            For `pred_class_logits`, `pred_shift_deltas` and `pred_centerness`, see
                :meth:`FCOSHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_shift_deltas, pred_centerness = \
            permute_all_cls_and_box_to_N_HWA_K_and_concat_fcos(
                pred_class_logits, pred_shift_deltas, pred_centerness,
                self.num_classes
            )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        valid_img = (valid_img>0).reshape(gt_classes.shape[0],-1).repeat(1,gt_classes.shape[1]).flatten()
        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)
        gt_centerness = gt_centerness.view(-1, 1)

        valid_idxs = (gt_classes >= 0) & valid_img
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground)

        # regression loss
        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            gt_centerness[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.iou_loss_type,
            reduction="mean",
        )

        # centerness loss
        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_centerness[foreground_idxs],
            gt_centerness[foreground_idxs],
            reduction="sum",
        ) / max(1, num_foreground)

        return {
            "fcos_cls_loss2": loss_cls,
            "fcos_reg_loss2": loss_box_reg,
            "fcos_center_loss2": loss_centerness
        }

    @torch.no_grad()
    def get_fcos_ground_truth(self, shifts, targets):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            gt_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
            gt_centerness (Tensor):
                An float tensor (0, 1) of shape (N, R) whose values in [0, 1]
                storing ground-truth centerness for each shift.

        """
        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []

        valid_img = torch.ones(len(shifts)).to(shifts[0][0].device)
        for i,(shifts_per_image, targets_per_image) in enumerate(zip(shifts, targets)):

            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes

            if len(gt_boxes)==0:
                gt_classes_i = torch.zeros((shifts_over_all_feature_maps.shape[0])) + self.num_classes
                gt_classes_i = gt_classes_i.to(shifts_over_all_feature_maps.device).long()
                gt_shifts_reg_deltas_i = torch.zeros((shifts_over_all_feature_maps.shape[0],4)).to(shifts_over_all_feature_maps.device)
                gt_centerness_i = torch.zeros((shifts_over_all_feature_maps.shape[0])).to(shifts_over_all_feature_maps.device)
                valid_img[i] = 0

            else:
                object_sizes_of_interest = torch.cat([
                    shifts_i.new_tensor(size).unsqueeze(0).expand(
                        shifts_i.size(0), -1) for shifts_i, size in zip(
                        shifts_per_image, self.object_sizes_of_interest)
                ], dim=0)

                

                deltas = self.shift2box_transform.get_deltas(
                    shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))

                if self.center_sampling_radius > 0:
                    centers = gt_boxes.get_centers()
                    is_in_boxes = []
                    for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                        radius = stride * self.center_sampling_radius
                        center_boxes = torch.cat((
                            torch.max(centers - radius, gt_boxes.tensor[:, :2]),
                            torch.min(centers + radius, gt_boxes.tensor[:, 2:]),
                        ), dim=-1)
                        center_deltas = self.shift2box_transform.get_deltas(
                            shifts_i, center_boxes.unsqueeze(1))
                        is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                    is_in_boxes = torch.cat(is_in_boxes, dim=1)
                else:
                    # no center sampling, it will use all the locations within a ground-truth box
                    is_in_boxes = deltas.min(dim=-1).values > 0

                max_deltas = deltas.max(dim=-1).values
                # limit the regression range for each location
                is_cared_in_the_level = \
                    (max_deltas >= object_sizes_of_interest[None, :, 0]) & \
                    (max_deltas <= object_sizes_of_interest[None, :, 1])

                gt_positions_area = gt_boxes.area().unsqueeze(1).repeat(
                    1, shifts_over_all_feature_maps.size(0))
                gt_positions_area[~is_in_boxes] = math.inf
                gt_positions_area[~is_cared_in_the_level] = math.inf

                # if there are still more than one objects for a position,
                # we choose the one with minimal area
                positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)

                # ground truth box regression
                gt_shifts_reg_deltas_i = self.shift2box_transform.get_deltas(
                    shifts_over_all_feature_maps, gt_boxes[gt_matched_idxs].tensor)

                # ground truth classes
                has_gt = len(targets_per_image) > 0
                if has_gt:
                    gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                    # Shifts with area inf are treated as background.
                    gt_classes_i[positions_min_area == math.inf] = self.num_classes
                else:
                    gt_classes_i = torch.zeros_like(
                        gt_matched_idxs) + self.num_classes

                # ground truth centerness
                left_right = gt_shifts_reg_deltas_i[:, [0, 2]]
                top_bottom = gt_shifts_reg_deltas_i[:, [1, 3]]
                gt_centerness_i = torch.sqrt(
                    (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                    * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
                )

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
            gt_centerness.append(gt_centerness_i)

        return torch.stack(gt_classes), torch.stack(
            gt_shifts_deltas), torch.stack(gt_centerness),valid_img

    def inference(self, box_cls, box_delta, box_center, shifts, images):
        """
        Arguments:
            box_cls, box_delta, box_center: Same as the output of :meth:`FCOSHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(shifts) == len(images)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]
        # list[Tensor], one per level, each has shape (N, Hi x Wi, K or 4)

        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            box_ctr_per_image = [
                box_ctr_per_level[img_idx] for box_ctr_per_level in box_center
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, box_ctr_per_image,
                shifts_per_image, tuple(image_size))
            results.append(results_per_image)
        return results

    def pseudo_gt_generate(self, box_cls, box_delta, box_center, shifts, images,pseudo_score_thres=None):
        """
        Arguments:
            box_cls, box_delta, box_center: Same as the output of :meth:`FCOSHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(shifts) == len(images)
        results = []
        if pseudo_score_thres:
            thres = pseudo_score_thres
        else:
            thres = self.pseudo_score_thres
        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]
        # list[Tensor], one per level, each has shape (N, Hi x Wi, K or 4)

        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            box_ctr_per_image = [
                box_ctr_per_level[img_idx] for box_ctr_per_level in box_center
            ]
            results_per_image = self.pseudo_gt_generate_per_image(
                box_cls_per_image, box_reg_per_image, box_ctr_per_image,
                shifts_per_image, tuple(image_size),thres)
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, box_center, shifts,
                                    image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_center (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, box_ctr_i, shifts_i in zip(
                box_cls, box_delta, box_center, shifts):
            # (HxWxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            # predict boxes
            predicted_boxes = self.shift2box_transform.apply_deltas(
                box_reg_i, shifts_i)

            box_ctr_i = box_ctr_i.flatten().sigmoid_()[shift_idxs]
            predicted_prob = torch.sqrt(predicted_prob * box_ctr_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all,
                           self.nms_threshold)
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def pseudo_gt_generate_per_image(self, box_cls, box_delta, box_center, shifts,
                                    image_size,thres):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_center (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, box_ctr_i, shifts_i in zip(
                box_cls, box_delta, box_center, shifts):
            # (HxWxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > thres
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            # predict boxes
            predicted_boxes = self.shift2box_transform.apply_deltas(
                box_reg_i, shifts_i)

            box_ctr_i = box_ctr_i.flatten().sigmoid_()[shift_idxs]
            predicted_prob = torch.sqrt(predicted_prob * box_ctr_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all,
                           self.nms_threshold)
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs,teacher_model=None):
        """
        Normalize, pad and batch the input images.
        """
        if self.training:
            images_raw = [x["image_Raw"].to(self.device) for x in batched_inputs]
            images_nor = [x["image_Nor"].to(self.device) for x in batched_inputs]
            images_str = [x["image_Str"].to(self.device) for x in batched_inputs]

            images_raw = [self.normalizer(x) for x in images_raw]
            images_nor = [self.normalizer(x) for x in images_nor]
            images_str = [self.normalizer(x) for x in images_str]

            images_raw = ImageList.from_tensors(images_raw,
                                            teacher_model.backbone.size_divisibility)
            images_nor = ImageList.from_tensors(images_nor,
                                            self.backbone.size_divisibility)
            images_str = ImageList.from_tensors(images_str,
                                            self.backbone.size_divisibility)
        else:
            images_nor = [x["image"].to(self.device) for x in batched_inputs]

            images_nor = [self.normalizer(x) for x in images_nor]

            images_nor = ImageList.from_tensors(images_nor,
                                            self.backbone.size_divisibility)
            images_raw = None
            images_str = None

        return images_raw, images_nor,images_str

    def divide_pseudo_gt(self,high_score_Instances,gt_instances_raw,images_raw):
        
        missing_Instances = []
        refine_gt_Instances = []
        for img_inds,(ins,gt_ins) in enumerate(zip(high_score_Instances,gt_instances_raw)):
            image_size = images_raw.image_sizes[img_inds]
            pred_bboxes = ins.pred_boxes.tensor

            if len(pred_bboxes) == 0:
                
                scores = ins.scores
                pred_classes = ins.pred_classes
                gt_bboxes = gt_ins.gt_boxes.tensor.clone()
                gt_classes = gt_ins.gt_classes.clone()
                missing_Instance = Instances(tuple(image_size))
                refine_gt_Instance = Instances(tuple(image_size))
                missing_Instance.pred_boxes = Boxes(pred_bboxes)
                missing_Instance.pred_classes = pred_classes
                missing_Instance.scores = scores
                refine_gt_Instance.gt_boxes = Boxes(gt_bboxes)
                refine_gt_Instance.gt_classes = gt_classes
            else:

                missing_Instance,refine_gt_Instance = Revision_GT(ins,gt_ins)
            
            missing_Instances.append(missing_Instance)
            refine_gt_Instances.append(refine_gt_Instance)
        
        return missing_Instances, refine_gt_Instances


    def teacher_revision(self,teacher_model,features_raw,gt_instances_raw,shifts_raw,images_raw):

        cls_t, delta_t, center_t,_,_,_ = teacher_model.head(features_raw)
        missing_Instances = None
        refine_gt_Instances = None
        high_score_Instances = self.pseudo_gt_generate(cls_t, delta_t, center_t, shifts_raw, images_raw,0.6)
        
        # missing_Instances , refine_gt_Instances = self.divide_pseudo_gt(high_score_Instances,gt_instances_raw,images_raw)
        
        return missing_Instances , refine_gt_Instances,high_score_Instances

class FCOSHead(nn.Module):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        num_convs = cfg.MODEL.FCOS.NUM_CONVS
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        # fmt: on
        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels,
                                   num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.centerness = nn.Conv2d(in_channels,
                                    1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        # Initialization
        for modules in [
                self.cls_subnet, self.bbox_subnet,
                self.cls_score, self.bbox_pred, self.centerness,
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    def forward(self, features, features_color=None):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
            centerness (list[Tensor]): #lvl tensors, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness at each spatial position.
        """
        logits1 = []
        bbox_reg1 = []
        centerness1 = []
        logits2 = []
        bbox_reg2 = []
        centerness2 = []

        num_scales = [i for i in range(len(self.fpn_strides))]
        if features_color == None:
            features_color = [None for _ in range(len(features))]
        for l, feature, feature_color in zip(num_scales, features, features_color):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)
            logits1.append(self.cls_score(cls_subnet))

            if feature_color is not None:
                cls_subnet_color = self.cls_subnet(feature_color)
                bbox_subnet_color = self.bbox_subnet(feature_color)
                logits2.append(self.cls_score(cls_subnet_color))

            if self.centerness_on_reg:
                centerness1.append(self.centerness(bbox_subnet))
                if feature_color is not None:
                    centerness2.append(self.centerness(bbox_subnet_color))
            else:
                centerness1.append(self.centerness(cls_subnet))
                if feature_color is not None:
                    centerness2.append(self.centerness(cls_subnet_color))

            bbox_pred1 = self.scales[l](self.bbox_pred(bbox_subnet))
            if feature_color is not None:
                bbox_pred2 = self.scales[l](self.bbox_pred(bbox_subnet_color))

            if self.norm_reg_targets:
                bbox_reg1.append(F.relu(bbox_pred1) * self.fpn_strides[l])
                if feature_color is not None:
                    bbox_reg2.append(F.relu(bbox_pred2) * self.fpn_strides[l])
            else:
                bbox_reg1.append(torch.exp(bbox_pred1))
                if feature_color is not None:
                    bbox_reg2.append(torch.exp(bbox_pred2))

        return logits1, bbox_reg1, centerness1, logits2, bbox_reg2, centerness2


        