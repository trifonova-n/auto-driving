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


class Eval(object):
    def __init__(self, params=None):
        if params is None:
            params = Params()
        self.params = params
        self.eval_res = []
        self.precision = None

    def evaluate_img_cat(self, detect_img, gt_img, scores):
        detect_img = detect_img.astype(np.int32)
        gt_img = gt_img.astype(np.int32)
        iou_thresholds = self.params['iouThrs']

        ious, idx_2_dlabel, idx_2_glabel = compute_iou(detect_img, gt_img, scores)

        scores = np.sort(scores)[::-1]

        T = len(iou_thresholds)
        G = len(idx_2_glabel)
        D = len(idx_2_dlabel)

        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        for tind, t in enumerate(iou_thresholds):
            for dind in range(D):
                iou = t
                m = -1
                for gind in range(G):
                    if gtm[tind, gind] > 0 or ious[dind, gind] < iou:
                        continue
                    m = gind
                    iou = ious[dind, gind]
                gtm[tind, m] = 1
                dtm[tind, dind] = 1
        return dtm, scores, G

    def evaluate_img(self, detect_masks, gt_img, scores, dt_cats):
        matches = []
        for cat in self.params.catIds:
            gt_cat_img = gt_img & (gt_img // 1000 == cat)
            dt_cat_masks = detect_masks[dt_cats == cat]
            dtm, scores, gtN = self.evaluate_img_cat(dt_cat_masks, gt_cat_img, scores[dt_cats == cat])
            matches.append({'dtMatches': dtm, 'dtScores': scores, 'gtN': gtN})
        self.eval_res.append(matches)
        self.precision = None

    def accumulate(self):
        p = self.params
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds)
        maxDet = self.params.maxDet
        precision = -np.ones((T, R, K))  # -1 for the precision of absent categories
        recall = -np.ones((T, K))
        for k in range(K):
            E = [e[k] for e in self.eval_res]
            E = [e for e in E if not e is None]
            if len(E) == 0:
                continue

            dtScores = np.concatenate([e['dtScores'] for e in E])
            inds = np.argsort(-dtScores, kind='mergesort')
            dtScoresSorted = dtScores[inds]
            dtm = np.concatenate([e['dtMatches'] for e in E], axis=1)[:, inds]
            gtN = np.sum(e['gtN'] for e in E)

            tps = dtm
            fps = np.logical_not(dtm)

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

            for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                rc = tp / gtN
                pr = tp / (tp + fp + np.spacing(1))
                q = np.zeros((R,))

                # numpy is slow without cython optimization for accessing elements
                # use python array gets significant speed improvement
                pr = pr.tolist()
                q = q.tolist()

                for i in range(len(tp) - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]

                inds = np.searchsorted(rc, p.recThrs, side='left')

                try:
                    for ri, pi in enumerate(inds):
                        q[ri] = pr[pi]
                except:
                    pass
                precision[t, :, k] = np.array(q)
        self.precision = precision

    def mean_avrg_precision(self, iouThr=None):
        if self.precision is None:
            self.accumulate()
        s = self.precision
        if iouThr is not None:
            t = np.where(iouThr == self.params.iouThrs)[0]
            s = self.precision[t]
        if len(s[s > -1]) == 0:
            return -1

        return np.mean(s[s > -1])

class Params(object):
    def __init__(self):
        self.iouThrs = np.linspace(0.5, 0.95, 10)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .1) + 1, endpoint=True)
        self.catIds = []
        self.maxDet = 100
