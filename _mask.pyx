cimport cython
cimport numpy as np
import numpy as np
ctypedef np.uint8_t uint8


def compute_iou(uint8[:, :, :] dt_masks not None, uint8[:, :, :] gt_masks not None, uint8[:] gt_crowd=None):
    cdef int D = dt_masks.shape[0]
    cdef int G = gt_masks.shape[0]
    cdef int H = dt_masks.shape[1]
    cdef int W = dt_masks.shape[2]
    cdef int d, g, y, x
    assert H == gt_masks.shape[1] and W == gt_masks.shape[2]
    if gt_crowd is not None:
        assert gt_crowd.shape[0] == G
    output = np.empty((D, G), dtype=np.float32)
    cdef np.float32[:, :] ious = output

    for d in range(D):
        for g in range(G):
            cdef int intersection_area = 0
            cdef int union_area = 0
            for y in range(H):
                for x in range(W):
                    if dt_masks[d, y, x] and gt_masks[g, y, x]:
                        intersection_area += 1
                    # for crowd gt union area is equal to detection area
                    if gt_crowd is not None and gt_crowd[g] and dt_masks[g, y, x]:
                        union_area += 1
                        continue
                    if dt_masks[d, y, x] or gt_masks[g, y, x]:
                        union_area += 1
            ious[d, g] = np.float32(intersection_area) / union_area
    return output


def rle_encoding(mask):
    cdef int H = mask.shape[0]
    cdef int W = mask.shape[1]
    cdef uint8[:, :] M = mask
    cdef int i
    cdef l = 0
    cdef start = 0
    encoding = []

    for i in range(H*W):
        if M[i // W, i % W]:
            l += 1
        elif l > 0:
            start = i - l
            encoding.append((start, l))
            l = 0
    if l > 0:
        start = i - l + 1
        encoding.append((start, l))
    return encoding
