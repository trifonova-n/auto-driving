import numpy as np
from auto_driving import _mask


rle_encoding = _mask.rle_encoding


def compute_iou(dt_masks,
                gt_masks,
                gt_crowd=None):
    if dt_masks.dtype != np.bool:
        dt_masks = dt_masks.astype(np.bool)
    if gt_masks.dtype != np.bool:
        gt_masks = gt_masks.astype(np.bool)
    if gt_crowd is not None and gt_crowd.dtype != np.bool:
        gt_crowd = gt_crowd.astype(np.bool)
    return _mask.compute_iou(dt_masks, gt_masks, gt_crowd)

