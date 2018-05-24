import numpy as np


def cumargmax(a):
    m = np.maximum.accumulate(a)
    x = np.repeat(np.arange(a.shape[0])[:, None], a.shape[1], axis=1)
    x[1:] *= m[:-1] < m[1:]
    np.maximum.accumulate(x, axis=0, out=x)
    return x


def compute_iou(predictions, targets, scores):
    """
    Computes iou matrix
    :param predictions: prediction label image with N object labels from 1
    :param targets: target label image with object labels from 1 to 999
    :param scores: N prediction scores produced by model
    :return: iou, idx_2_plabel, idx_2_tlabel
    """
    target_labels, t_counts = np.unique(targets, return_counts=True)
    target_labels, t_counts = target_labels[:-1], t_counts[:-1]
    predict_labels, p_counts = np.unique(predictions, return_counts=True)
    predict_labels, p_counts = predict_labels[:-1], p_counts[:-1]

    p_number = len(predict_labels)
    t_number = len(target_labels)
    # compute intersections of all predictions with all targets
    un_mask = predictions * 1000 + targets

    in_labels, counts = np.unique(un_mask, return_counts=True)
    in_labels = [(l // 1000, l % 1000) for l in in_labels]

    idx_2_plabel = [l for _, l in sorted(zip(scores, predict_labels), reverse=True)]
    plabel_2_idx = dict(zip(idx_2_plabel, range(p_number)))
    idx_2_tlabel = target_labels
    tlabel_2_idx = dict(zip(idx_2_tlabel, range(t_number)))

    intersections = np.zeros((p_number, t_number))
    in_idcs = [(plabel_2_idx[p], tlabel_2_idx[t]) for p, t in in_labels]
    intersections[zip(*in_idcs)] = counts

    # areas of targets
    t_dict = dict(zip(target_labels, t_counts))
    # areas of predictions
    p_dict = dict(zip(predict_labels, p_counts))

    p_counts = [p_dict[idx_2_plabel[i]] for i in range(p_number)]
    t_counts = [t_dict[idx_2_tlabel[i]] for i in range(t_number)]
    unions = ((-intersections + t_counts).T + p_counts).T
    iou = intersections / unions
    return iou, idx_2_plabel, idx_2_tlabel


def image_avrg_prec(predictions, targets, scores):
    predictions = predictions.astype(np.int32)
    targets = targets.astype(np.int32)
    iou_thresholds = np.linspace(0.5, 0.95, 10)

    iou, idx_2_plabel, idx_2_tlabel = compute_iou(predictions, targets, scores)

    match_indices = cumargmax(iou)
    scores = np.sort(scores)[::-1]
    score_vals, indices = np.unique(scores)
    score_vals = score_vals.append(1.0)
    indices = indices.append(-1)

    for i, score_thresh in zip(indices, score_vals):
        matches = match_indices[i, :]
        matche_ious = iou[i, matches]
        matches_per_iou_thresh = matche_ious




    TPs = np.zeros_like(thresholds)
    t_nomatch = {}
    p_nomatch = {}
    for tl in target_labels:
        for pl in predict_labels:
            ul = pl*1000 + tl
            intersetion_area = in_dict.get(ul, 0)
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