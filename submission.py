import numpy as np


def rle_encoding(x):
    """ Run-length encoding based on
    https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    Modified by Konstantin, https://www.kaggle.com/lopuhin
    """
    assert x.dtype == np.bool
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.append([b, 0])
        run_lengths[-1][1] += 1
        prev = b
    return '|'.join('{} {}'.format(*pair) for pair in run_lengths)


def segs_to_rle_rows(lab_img, **kwargs):
    out_rows = []
    for i in np.unique(lab_img[lab_img>0]):
        c_dict = dict(**kwargs)
        c_dict['LabelId'] = i//1000
        c_dict['PixelCount'] = np.sum(lab_img==i)
        c_dict['Confidence'] = 0.5 # our classifier isnt very good so lets not put the confidence too high
        c_dict['EncodedPixels'] = rle_encoding(lab_img==i)
        out_rows += [c_dict]
    return out_rows

