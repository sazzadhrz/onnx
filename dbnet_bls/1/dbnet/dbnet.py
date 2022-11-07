# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any, Callable, Dict, List, Tuple, Optional

import os
import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.deform_conv import DeformConv2d


def _addindent(s_, num_spaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class NestedObject:
    def extra_repr(self) -> str:
        return ''

    def __repr__(self):
        # We treat the extra repr like the sub-object, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        if hasattr(self, '_children_names'):
            for key in self._children_names:
                child = getattr(self, key)
                if isinstance(child, list) and len(child) > 0:
                    child_str = ",\n".join([repr(subchild) for subchild in child])
                    if len(child) > 1:
                        child_str = _addindent(f"\n{child_str},", 2) + '\n'
                    child_str = f"[{child_str}]"
                else:
                    child_str = repr(child)
                child_str = _addindent(child_str, 2)
                child_lines.append('(' + key + '): ' + child_str)
        lines = extra_lines + child_lines

        main_str = self.__class__.__name__ + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str



class DetectionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        box_thresh (float): minimal objectness score to consider a box
        bin_thresh (float): threshold to apply to segmentation raw heatmap
        assume straight_pages (bool): if True, fit straight boxes only
    """

    def __init__(
        self,
        box_thresh: float = 0.5,
        bin_thresh: float = 0.5,
        assume_straight_pages: bool = True
    ) -> None:

        self.box_thresh = box_thresh
        self.bin_thresh = bin_thresh
        self.assume_straight_pages = assume_straight_pages
        self._opening_kernel = np.ones((3, 3), dtype=np.uint8)

    def extra_repr(self) -> str:
        return f"bin_thresh={self.bin_thresh}, box_thresh={self.box_thresh}"

    @staticmethod
    def box_score(
        pred: np.ndarray,
        points: np.ndarray,
        assume_straight_pages: bool = True
    ) -> float:
        """Compute the confidence score for a polygon : mean of the p values on the polygon

        Args:
            pred (np.ndarray): p map returned by the model

        Returns:
            polygon objectness
        """
        h, w = pred.shape[:2]

        if assume_straight_pages:
            xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int32), 0, w - 1)
            xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int32), 0, w - 1)
            ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int32), 0, h - 1)
            ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int32), 0, h - 1)
            return pred[ymin:ymax + 1, xmin:xmax + 1].mean()

        else:
            mask = np.zeros((h, w), np.int32)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1.0)
            product = pred * mask
            return np.sum(product) / np.count_nonzero(product)

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def __call__(
        self,
        proba_map,
    ) -> List[List[np.ndarray]]:
        """Performs postprocessing for a list of model outputs

        Args:
            proba_map: probability map of shape (N, H, W, C)

        Returns:
            list of N class predictions (for each input sample), where each class predictions is a list of C tensors
        of shape (*, 5) or (*, 6)
        """

        if proba_map.ndim != 4:
            raise AssertionError(f"arg `proba_map` is expected to be 4-dimensional, got {proba_map.ndim}.")

        # Erosion + dilation on the binary map
        bin_map = [
            [
                cv2.morphologyEx(bmap[..., idx], cv2.MORPH_OPEN, self._opening_kernel)
                for idx in range(proba_map.shape[-1])
            ]
            for bmap in (proba_map >= self.bin_thresh).astype(np.uint8)
        ]

        return [
            [self.bitmap_to_boxes(pmaps[..., idx], bmaps[idx]) for idx in range(proba_map.shape[-1])]
            for pmaps, bmaps in zip(proba_map, bin_map)
        ]



class DBPostProcessor(DetectionPostProcessor):
    """Implements a post processor for DBNet adapted from the implementation of `xuannianz
    <https://github.com/xuannianz/DifferentiableBinarization>`_.

    Args:
        unclip ratio: ratio used to unshrink polygons
        min_size_box: minimal length (pix) to keep a box
        max_candidates: maximum boxes to consider in a single page
        box_thresh: minimal objectness score to consider a box
        bin_thresh: threshold used to binzarized p_map at inference time

    """
    def __init__(
        self,
        box_thresh: float = 0.1,
        bin_thresh: float = 0.3,
        assume_straight_pages: bool = True,
    ) -> None:

        super().__init__(
            box_thresh,
            bin_thresh,
            assume_straight_pages
        )
        self.unclip_ratio = 1.5 if assume_straight_pages else 2.2

    def polygon_to_box(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """Expand a polygon (points) by a factor unclip_ratio, and returns a polygon

        Args:
            points: The first parameter.

        Returns:
            a box in absolute coordinates (xmin, ymin, xmax, ymax) or (4, 2) array (quadrangle)
        """
        if not self.assume_straight_pages:
            # Compute the rectangle polygon enclosing the raw polygon
            rect = cv2.minAreaRect(points)
            points = cv2.boxPoints(rect)
            # Add 1 pixel to correct cv2 approx
            area = (rect[1][0] + 1) * (1 + rect[1][1])
            length = 2 * (rect[1][0] + rect[1][1]) + 2
        else:
            poly = Polygon(points)
            area = poly.area
            length = poly.length
        distance = area * self.unclip_ratio / length  # compute distance to expand polygon
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        _points = offset.Execute(distance)
        # Take biggest stack of points
        idx = 0
        if len(_points) > 1:
            max_size = 0
            for _idx, p in enumerate(_points):
                if len(p) > max_size:
                    idx = _idx
                    max_size = len(p)
            # We ensure that _points can be correctly casted to a ndarray
            _points = [_points[idx]]
        expanded_points = np.asarray(_points)  # expand polygon
        if len(expanded_points) < 1:
            return None
        return cv2.boundingRect(expanded_points) if self.assume_straight_pages else np.roll(
            cv2.boxPoints(cv2.minAreaRect(expanded_points)), -1, axis=0
        )

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
    ) -> np.ndarray:
        """Compute boxes from a bitmap/pred_map

        Args:
            pred: Pred map from differentiable binarization output
            bitmap: Bitmap map computed from pred (binarized)
            angle_tol: Comparison tolerance of the angle with the median angle across the page
            ratio_tol: Under this limit aspect ratio, we cannot resolve the direction of the crop

        Returns:
            np tensor boxes for the bitmap, each box is a 5-element list
                containing x, y, w, h, score for the box
        """
        height, width = bitmap.shape[:2]
        min_size_box = 1 + int(height / 512)
        boxes = []
        # get contours from connected components on the bitmap
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Check whether smallest enclosing bounding box is not too small
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) < min_size_box):
                continue
            # Compute objectness
            if self.assume_straight_pages:
                x, y, w, h = cv2.boundingRect(contour)
                points = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]])
                score = self.box_score(pred, points, assume_straight_pages=True)
            else:
                score = self.box_score(pred, contour, assume_straight_pages=False)

            if score < self.box_thresh:   # remove polygons with a weak objectness
                continue

            if self.assume_straight_pages:
                _box = self.polygon_to_box(points)
            else:
                _box = self.polygon_to_box(np.squeeze(contour))

            # Remove too small boxes
            if self.assume_straight_pages:
                if _box is None or _box[2] < min_size_box or _box[3] < min_size_box:
                    continue
            elif np.linalg.norm(_box[2, :] - _box[0, :], axis=-1) < min_size_box:
                continue

            if self.assume_straight_pages:
                x, y, w, h = _box  # type: ignore[misc]
                # compute relative polygon to get rid of img shape
                xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
                boxes.append([xmin, ymin, xmax, ymax, score])
            else:
                # compute relative box to get rid of img shape, in that case _box is a 4pt polygon
                if not isinstance(_box, np.ndarray) and _box.shape == (4, 2):
                    raise AssertionError("When assume straight pages is false a box is a (4, 2) array (polygon)")
                _box[:, 0] /= width
                _box[:, 1] /= height
                boxes.append(_box)

        if not self.assume_straight_pages:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 4, 2), dtype=pred.dtype)
        else:
            return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=pred.dtype)


class _DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_.

    Args:
        feature extractor: the backbone serving as feature extractor
        fpn_channels: number of channels each extracted feature maps is mapped to
    """

    shrink_ratio = 0.4
    thresh_min = 0.3
    thresh_max = 0.7
    min_size_box = 3
    assume_straight_pages: bool = True

    @staticmethod
    def compute_distance(
        xs: np.array,
        ys: np.array,
        a: np.array,
        b: np.array,
        eps: float = 1e-7,
    ) -> float:
        """Compute the distance for each point of the map (xs, ys) to the (a, b) segment

        Args:
            xs : map of x coordinates (height, width)
            ys : map of y coordinates (height, width)
            a: first point defining the [ab] segment
            b: second point defining the [ab] segment

        Returns:
            The computed distance

        """
        square_dist_1 = np.square(xs - a[0]) + np.square(ys - a[1])
        square_dist_2 = np.square(xs - b[0]) + np.square(ys - b[1])
        square_dist = np.square(a[0] - b[0]) + np.square(a[1] - b[1])
        cosin = (square_dist - square_dist_1 - square_dist_2) / (2 * np.sqrt(square_dist_1 * square_dist_2) + eps)
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_dist_1 * square_dist_2 * square_sin / square_dist)
        result[cosin < 0] = np.sqrt(np.fmin(square_dist_1, square_dist_2))[cosin < 0]
        return result

    def draw_thresh_map(
        self,
        polygon: np.array,
        canvas: np.array,
        mask: np.array,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a polygon treshold map on a canvas, as described in the DB paper

        Args:
            polygon : array of coord., to draw the boundary of the polygon
            canvas : threshold map to fill with polygons
            mask : mask for training on threshold polygons
        """
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise AttributeError("polygon should be a 2 dimensional array of coords")

        # Augment polygon by shrink_ratio
        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(coor) for coor in polygon]  # Get coord as list of tuples
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])

        # Fill the mask with 1 on the new padded polygon
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        # Get min/max to recover polygon after distance computation
        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        # Get absolute polygon for distance computation
        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin
        # Get absolute padded polygon
        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        # Compute distance map to fill the padded polygon
        distance_map = np.zeros((polygon.shape[0], height, width), dtype=polygon.dtype)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.compute_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = np.min(distance_map, axis=0)

        # Clip the padded polygon inside the canvas
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)

        # Fill the canvas with the distances computed inside the valid padded polygon
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymin + 1,
                xmin_valid - xmin:xmax_valid - xmin + 1
            ],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1]
        )

        return polygon, canvas, mask

    def build_target(
        self,
        target: List[np.ndarray],
        output_shape: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if any(t.dtype != np.float32 for t in target):
            raise AssertionError("the expected dtype of target 'boxes' entry is 'np.float32'.")
        if any(np.any((t[:, :4] > 1) | (t[:, :4] < 0)) for t in target):
            raise ValueError("the 'boxes' entry of the target is expected to take values between 0 & 1.")

        input_dtype = target[0].dtype if len(target) > 0 else np.float32

        seg_target = np.zeros(output_shape, dtype=np.uint8)
        seg_mask = np.ones(output_shape, dtype=bool)
        thresh_target = np.zeros(output_shape, dtype=np.float32)
        thresh_mask = np.ones(output_shape, dtype=np.uint8)

        for idx, _target in enumerate(target):
            # Draw each polygon on gt
            if _target.shape[0] == 0:
                # Empty image, full masked
                seg_mask[idx] = False

            # Absolute bounding boxes
            abs_boxes = _target.copy()
            if abs_boxes.ndim == 3:
                abs_boxes[:, :, 0] *= output_shape[-1]
                abs_boxes[:, :, 1] *= output_shape[-2]
                polys = abs_boxes
                boxes_size = np.linalg.norm(abs_boxes[:, 2, :] - abs_boxes[:, 0, :], axis=-1)
                abs_boxes = np.concatenate((abs_boxes.min(1), abs_boxes.max(1)), -1).round().astype(np.int32)
            else:
                abs_boxes[:, [0, 2]] *= output_shape[-1]
                abs_boxes[:, [1, 3]] *= output_shape[-2]
                abs_boxes = abs_boxes.round().astype(np.int32)
                polys = np.stack([
                    abs_boxes[:, [0, 1]],
                    abs_boxes[:, [0, 3]],
                    abs_boxes[:, [2, 3]],
                    abs_boxes[:, [2, 1]],
                ], axis=1)
                boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])

            for box, box_size, poly in zip(abs_boxes, boxes_size, polys):
                # Mask boxes that are too small
                if box_size < self.min_size_box:
                    seg_mask[idx, box[1]: box[3] + 1, box[0]: box[2] + 1] = False
                    continue

                # Negative shrink for gt, as described in paper
                polygon = Polygon(poly)
                distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                subject = [tuple(coor) for coor in poly]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)

                # Draw polygon on gt if it is valid
                if len(shrinked) == 0:
                    seg_mask[idx, box[1]: box[3] + 1, box[0]: box[2] + 1] = False
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                if shrinked.shape[0] <= 2 or not Polygon(shrinked).is_valid:
                    seg_mask[idx, box[1]: box[3] + 1, box[0]: box[2] + 1] = False
                    continue
                cv2.fillPoly(seg_target[idx], [shrinked.astype(np.int32)], 1)

                # Draw on both thresh map and thresh mask
                poly, thresh_target[idx], thresh_mask[idx] = self.draw_thresh_map(poly, thresh_target[idx],
                                                                                  thresh_mask[idx])

        thresh_target = thresh_target.astype(input_dtype) * (self.thresh_max - self.thresh_min) + self.thresh_min

        seg_target = seg_target.astype(input_dtype)
        seg_mask = seg_mask.astype(bool)
        thresh_target = thresh_target.astype(input_dtype)
        thresh_mask = thresh_mask.astype(bool)

        return seg_target, seg_mask, thresh_target, thresh_mask



class FeaturePyramidNetwork(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        deform_conv: bool = False,
    ) -> None:

        super().__init__()

        out_chans = out_channels // len(in_channels)

        conv_layer = DeformConv2d if deform_conv else nn.Conv2d

        self.in_branches = nn.ModuleList([
            nn.Sequential(
                conv_layer(chans, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for idx, chans in enumerate(in_channels)
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out_branches = nn.ModuleList([
            nn.Sequential(
                conv_layer(out_channels, out_chans, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_chans),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2 ** idx, mode='bilinear', align_corners=True),
            ) for idx, chans in enumerate(in_channels)
        ])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if len(x) != len(self.out_branches):
            raise AssertionError
        # Conv1x1 to get the same number of channels
        _x: List[torch.Tensor] = [branch(t) for branch, t in zip(self.in_branches, x)]
        out: List[torch.Tensor] = [_x[-1]]
        for t in _x[:-1][::-1]:
            out.append(self.upsample(out[-1]) + t)

        # Conv and final upsampling
        out = [branch(t) for branch, t in zip(self.out_branches, out[::-1])]

        return torch.cat(out, dim=1)


class DBNet(_DBNet, nn.Module):

    def __init__(
        self,
        feat_extractor: IntermediateLayerGetter,
        head_chans: int = 256,
        deform_conv: bool = False,
        num_classes: int = 1,
        assume_straight_pages: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.cfg = cfg

        conv_layer = DeformConv2d if deform_conv else nn.Conv2d

        self.assume_straight_pages = assume_straight_pages

        self.feat_extractor = feat_extractor
        # Identify the number of channels for the head initialization
        _is_training = self.feat_extractor.training
        self.feat_extractor = self.feat_extractor.eval()
        with torch.no_grad():
            out = self.feat_extractor(torch.zeros((1, 3, 224, 224)))
            fpn_channels = [v.shape[1] for _, v in out.items()]

        if _is_training:
            self.feat_extractor = self.feat_extractor.train()

        self.fpn = FeaturePyramidNetwork(fpn_channels, head_chans, deform_conv)
        # Conv1 map to channels

        self.prob_head = nn.Sequential(
            conv_layer(head_chans, head_chans // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_chans // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_chans // 4, head_chans // 4, 2, stride=2, bias=False),
            nn.BatchNorm2d(head_chans // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_chans // 4, num_classes, 2, stride=2),
        )
        self.thresh_head = nn.Sequential(
            conv_layer(head_chans, head_chans // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_chans // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_chans // 4, head_chans // 4, 2, stride=2, bias=False),
            nn.BatchNorm2d(head_chans // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(head_chans // 4, num_classes, 2, stride=2),
        )

        self.postprocessor = DBPostProcessor(assume_straight_pages=assume_straight_pages)

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith('feat_extractor.'):
                continue
            if isinstance(m, (nn.Conv2d, DeformConv2d)):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[np.ndarray]] = None,
        # return_model_output: bool = False,
        return_model_output: bool = True,
        return_preds: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Extract feature maps at different stages
        feats = self.feat_extractor(x)
        feats = [feats[str(idx)] for idx in range(len(feats))]
        # Pass through the FPN
        feat_concat = self.fpn(feats)
        logits = self.prob_head(feat_concat)

        out: Dict[str, Any] = {}
        if return_model_output or target is None or return_preds:
            prob_map = torch.sigmoid(logits)

        if return_model_output:
            # out["out_map"] = prob_map
            out = prob_map

        # if target is None or return_preds:
        #     # Post-process boxes (keep only text predictions)
        #     out["preds"] = [
        #         preds[0] for preds in self.postprocessor(prob_map.detach().cpu().permute((0, 2, 3, 1)).numpy())
        #     ]

        if target is not None:
            thresh_map = self.thresh_head(feat_concat)

        return out


def _dbnet(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    fpn_layers: List[str],
    backbone_submodule: Optional[str] = None,
    pretrained_backbone: bool = True,
    **kwargs: Any,
) -> DBNet:

    # Starting with Imagenet pretrained params introduces some NaNs in layer3 & layer4 of resnet50
    pretrained_backbone = pretrained_backbone and not arch.split('_')[1].startswith('resnet')
    pretrained_backbone = pretrained_backbone and not pretrained

    # Feature extractor
    backbone = backbone_fn(pretrained_backbone)
    if isinstance(backbone_submodule, str):
        backbone = getattr(backbone, backbone_submodule)
    feat_extractor = IntermediateLayerGetter(
        backbone,
        {layer_name: str(idx) for idx, layer_name in enumerate(fpn_layers)},
    )

    cfg = {
        'input_shape': (3, 1024, 1024),
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'url': 'https://github.com/mindee/doctr/releases/download/v0.3.1/db_resnet50-ac60cadc.pt',
    }
    # Build the model
    model = DBNet(feat_extractor, cfg=cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # state_dict = torch.load(os.getcwd()+'/static/assets/models/pytorch/db_resnet50-ac60cadc.pt', map_location=device)
        state_dict = torch.load(os.getcwd()+'/static/assets/models/pytorch/doctr_detection_model.pt', map_location=device)
        # Load weights
        model.load_state_dict(state_dict)

    return model


def db_resnet50(pretrained: bool = False, **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a ResNet-50 backbone.

    Example::
        >>> import torch
        >>> from doctr.models import db_resnet50
        >>> model = db_resnet50(pretrained=True)
        >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _dbnet(
        'db_resnet50',
        pretrained,
        resnet50,
        ['layer1', 'layer2', 'layer3', 'layer4'],
        None,
        **kwargs,
    )
