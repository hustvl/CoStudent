# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
from typing import Any, Dict, List, Tuple, Union
import cv2
import os
import numpy as np
import torch
from cvpods.structures import Boxes
from cvpods.layers import cat,batched_nms
from PIL import Image, ImageDraw,ImageFont
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

class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/Get a field:
       .. code-block:: python
          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)
    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``,
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    def draw(self,img,color,path=None,basename=None,line_size=2,with_label_and_scores=False,cat_ids2color=None,scores_fliter=None,name=None):
        # assert img.shape[1:] == self.image_size
        boxes_name,class_name = Instances.get_bbox_class_name(self)
        bboxes = self.get(boxes_name).tensor
        if isinstance(img,torch.Tensor):
            img = np.array(img.cpu().int()).astype(np.uint8).transpose((1,2,0)).copy()

        if len(bboxes)!=0:
            bboxes = np.array(bboxes.cpu().int()).tolist()
            
            if isinstance(color,tuple):
                colors = [color for _ in range(len(bboxes))]
            
            if with_label_and_scores:
                # from pdb import set_trace
                # set_trace()
                if cat_ids2color is not None:
                    labels = np.array(self.get(class_name).cpu()).tolist()
                    colors = []
                    for label in labels:
                        if label not in cat_ids2color.keys():
                            colors.append((0,0,0))
                        else:
                            colors.append(cat_ids2color[label])
                    
            
            valids = [True for _ in range(len(bboxes))]
            if with_label_and_scores:
                assert self.has('scores')
                
                img = Image.fromarray(img)
       
                draw = ImageDraw.Draw(img)
                labels = np.array(self.get(class_name).cpu()).tolist()
                scores = np.array(self.get('scores').cpu()).tolist()
                if scores_fliter is not None:
                    valids = np.array(self.get('scores').cpu()) > scores_fliter
                if name is not None:
                    texts = ['{}:{:.4f}'.format(name[label],score) for label,score in zip(labels,scores)]
                else:
                    texts = ['{}:{:.4f}'.format(str(label),score) for label,score in zip(labels,scores)]
                font = ImageFont.truetype(
                        font='/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf',
                        size=np.floor(40).astype('int32')
                    ) 
                for text,(left, top,_,_),valid in zip(texts,bboxes,valids):
                    if valid:
                        draw.text((left, top-40), text, (0,0,255),font=font)
                img = np.array(img)

            for bbox,color,valid in zip(bboxes,colors,valids):
                if valid:
                    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, line_size)
        if path:
            assert basename is not None and basename.endswith('.jpg')
            if not os.path.exists(path):
                os.makedirs(path)
            img = img.astype(np.uint8)
            cv2.imwrite(os.path.join(path,basename),img)
            return None
        return img


    def bbox_to_int(self):
        boxes_name,_ = Instances.get_bbox_class_name(self)
        bboxes = self.get(boxes_name).tensor
        self.set(boxes_name,Boxes(torch.round(bboxes)))


    # Tensor-like methods
    def to(self, device: str) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            return len(v)
        return 0

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def sub(ins1,ins2):
        boxes_name1,class_name1 = Instances.get_bbox_class_name(ins1)
        boxes_name2,class_name2 = Instances.get_bbox_class_name(ins2)
        ins1_bboxes =  ins1.get(boxes_name1).tensor
        ins2_bboxes =  ins1.get(boxes_name2).tensor
        (ins1_bboxes[:,None,:]-ins2_bboxes[None,:,:]).abs().sum(-1) <= 4

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    @staticmethod
    def get_bbox_class_name(instance):
        boxes_name = 'gt_boxes'
        class_name = 'gt_classes'
        for k in instance._fields.keys():
            if k.endswith('boxes'):
                boxes_name = k
            if k.endswith('classes'):
                class_name = k
        return boxes_name,class_name


    @staticmethod
    def refine_and_cat(instance_list_t,instance_list_s,iou_thre=0.6):
        
        assert len(instance_list_t) == len(instance_list_s)
        boxes_name,class_name = Instances.get_bbox_class_name(instance_list_t[0])
        boxes_name_s,class_name_s = Instances.get_bbox_class_name(instance_list_s[0])
        image_size = [i.image_size for i in instance_list_t]
        for inx,i in enumerate(instance_list_s) :
            assert i.image_size == image_size[inx]

        ret = []
        
        for inx,(t_i,s_i) in enumerate(zip(instance_list_t,instance_list_s)):
            t_i_boxes = t_i.get(boxes_name).tensor
            s_i_boxes = s_i.get(boxes_name_s).tensor
            if len(t_i_boxes) !=0 and len(s_i_boxes) !=0:

                t_i_classes = t_i.get(class_name)
                t_i_scores =  t_i.get('scores')
                s_i_classes = s_i.get(class_name_s)
                s_i_scores =  s_i.get('scores')

                ret_t = Instances(image_size[inx])
                ret_s = Instances(image_size[inx])
                ious = bbox_overlaps(t_i_boxes,s_i_boxes)
                refine_ious, refine_inds = ious.max(dim=0)
                iou_fliter = refine_ious >= iou_thre
                refine_inds = refine_inds[iou_fliter]


                mask = ious.new_full((t_i_scores.shape),1).bool()
                values = s_i_boxes.clone()
                c_values = s_i_classes.clone()
                s_values = s_i_scores.clone()

                for gt_i,pred_i in zip(torch.where(iou_fliter)[0],refine_inds):
                    score_fliter = t_i_scores[pred_i] > s_i_scores[gt_i]
                
                    if score_fliter:  
                        values[gt_i,:] = t_i_boxes[pred_i,:]
                        c_values[gt_i] = t_i_classes[pred_i]
                        s_values[gt_i] = t_i_scores[pred_i]
                        mask[pred_i] = False

                # bboxes.append(Boxes(values))
                # scores.append(s_values)
                # classes.append(c_values)

                ret_s.set(boxes_name_s, Boxes(values))
                ret_s.set('scores', s_values)
                ret_s.set(class_name_s,c_values)

                ret_t.set(boxes_name, Boxes(t_i_boxes[mask,:].reshape(-1,4)))
                ret_t.set('scores', t_i_scores[mask])
                ret_t.set(class_name,t_i_classes[mask])

                ret.append(Instances.cat([ret_s,ret_t]))
            else:
                ret.append(Instances.cat([t_i,s_i]))


        return ret
        
    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join(self._fields.keys()))
        return s

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=["
        for k, v in self._fields.items():
            s += "{} = {}, ".format(k, v)
        s += "])"
        return s
