import numpy as np


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[0], 4], dtype=np.float32)
    for i in range(mask.shape[0]):
        m = mask[i, :, :]

        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1, y1, x2, y2], dtype=np.float32)
    return boxes


def extract_bynary_masks(mask, class_map=None, max_object_count=80):
    vals = np.unique(mask)
    if class_map:
        vals = [a for a in vals if a // 1000 in class_map]
    object_count = len(vals) - 1
    if object_count > max_object_count:
        print(object_count)
    masks = np.zeros((max_object_count, mask.shape[0], mask.shape[1]), dtype=np.float32)
    classes = np.zeros(max_object_count, np.int64)
    for i, val in enumerate(vals[1:min(len(vals), max_object_count)]):
        masks[i, :, :] = (mask == val)
        if class_map:
            classes[i] = class_map[val // 1000]
        else:
            classes[i] = val // 1000

    return masks, classes


def cumm_mask(masks):
    if not masks:
        return np.zeros((masks.shape[1], masks.shape[2]), dtype=np.int64)
    mask = np.tensordot(list(range(1, len(masks) + 1)), masks, axes=1)
    return mask


def image_score(predictions, targets):
    predictions = predictions.astype(np.int32)
    targets = targets.astype(np.int32)
    thresholds = np.linspace(0.5, 0.95, 10)
    target_labels, t_counts = np.unique(targets, return_counts=True)
    target_labels, t_counts = target_labels[:-1], t_counts[:-1]
    predict_labels, p_counts = np.unique(predictions, return_counts=True)
    predict_labels, p_counts = predict_labels[:-1], p_counts[:-1]
    # compute intersections of all predictions with all targets
    un_mask = predictions*1000 + targets

    un_labels, counts = np.unique(un_mask, return_counts=True)
    # areas for intersections of prediction and target
    un_dict = dict(zip(un_labels, counts))
    # areas of targets
    t_dict = dict(zip(target_labels, t_counts))
    # areas of predictions
    p_dict = dict(zip(predict_labels, p_counts))
    TPs = np.zeros_like(thresholds)
    t_nomatch = {}
    p_nomatch = {}
    for tl in target_labels:
        for pl in predict_labels:
            ul = pl*1000 + tl
            intersetion_area = un_dict.get(ul, 0)
            union_area = t_dict[tl] + p_dict[pl] - intersetion_area
            iou = float(intersetion_area) / union_area
            matches = (iou > thresholds)
            TPs += matches
            t_nomatch[tl] = np.invert(matches) & t_nomatch.get(tl, np.ones_like(thresholds, dtype=bool))
            p_nomatch[pl] = np.invert(matches) & p_nomatch.get(pl, np.ones_like(thresholds, dtype=bool))

    FPs = np.sum(list(p_nomatch.values()), axis=0)
    FNs = np.sum(list(t_nomatch.values()), axis=0)
    if len(predict_labels) == 0:
        FNs = len(target_labels)
    #print('predict_labels', len(predict_labels))
    #print('target_labels', len(target_labels))
    #print('TPs', TPs)
    #print('FPs', FPs)
    #print('FNs', FNs)
    scores = TPs/(TPs + FPs + FNs)
    return np.mean(scores)


def score(prediction_masks, prediciton_classes, target_masks, target_classes):

    pass

def cumm_mask(masks):
    if not masks:
        return np.zeros((masks.shape[1], masks.shape[2]), dtype=np.int64)
    mask = np.tensordot(list(range(1, len(masks) + 1)), masks, axes=1)
    return mask

