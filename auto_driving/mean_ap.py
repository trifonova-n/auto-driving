import numpy as np
from .mask import compute_iou


class Eval(object):
    def __init__(self, config, params=None):
        if params is None:
            params = Params()
            params.catIds = config.catIds
        self.params = params
        self.eval_res = []
        self.precision = None

    def evaluate_img_cat(self, dt_masks, gt_masks, scores):
        iou_thresholds = self.params.iouThrs

        idx = np.argsort(scores)[::-1]
        scores = scores[idx]
        dt_masks = dt_masks[idx]
        ious = compute_iou(dt_masks, gt_masks)

        T = len(iou_thresholds)
        G = gt_masks.shape[0]
        D = dt_masks.shape[0]

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
                if m >= 0:
                    gtm[tind, m] = 1
                    dtm[tind, dind] = 1
        return dtm, scores, G

    def evaluate_img(self, dt_masks, gt_masks, dt_cats, gt_cats, dt_scores):
        matches = []
        for cat in range(1, len(self.params.catIds)):
            gt_cat_masks = gt_masks[gt_cats == cat]
            dt_cat_masks = dt_masks[dt_cats == cat]
            dtm, scores, gtN = self.evaluate_img_cat(dt_cat_masks, gt_cat_masks, dt_scores[dt_cats == cat])
            if gtN == 0:
                matches.append(None)
            else:
                matches.append({'dtMatches': dtm, 'dtScores': scores, 'gtN': gtN})
        self.eval_res.append(matches)
        self.precision = None

    def accumulate(self):
        p = self.params
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) - 1  # exclude background
        precision = -np.ones((T, R, K))  # -1 for the precision of absent categories
        for k in range(K):
            E = [e[k] for e in self.eval_res]
            E = [e for e in E if not e is None]
            if len(E) == 0:
                continue

            dtScores = np.concatenate([e['dtScores'] for e in E])
            inds = np.argsort(-dtScores, kind='mergesort')
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
        self.crowdCats = []
        self.maxDet = 100
