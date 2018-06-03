import numpy as np
import pandas as pd
import mask


class Submission(object):
    def __init__(self, config):
        self.config = config
        columns = ['ImageId', 'LabelId', 'Confidence', 'PixelCount', 'EncodedPixels']
        self.df = pd.DataFrame(columns=columns)

    def add_img(self, ImageId, dt_masks, dt_classes, dt_scores):
        rdict = {'ImageId': ImageId}
        for x, cl, sc in zip(dt_masks, dt_classes, dt_scores):
            rdict['LabelId'] = self.config.catIds[cl]
            rdict['Confidence'] = sc
            pixel_count, rle = rle_encoding(x)
            rdict['PixelCount'] = pixel_count
            rdict['EncodedPixels'] = rle
        self.df.append(rdict)

    def save(self, filename):
        self.df.to_csv(filename, index=False)


def rle_encoding(x):
    """ Run-length encoding based on
    https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    Modified by Konstantin, https://www.kaggle.com/lopuhin
    """
    assert x.dtype == np.bool
    run_lengths = mask.rle_encoding(x)
    pixel_count = sum([p[1] for p in run_lengths])
    return pixel_count, '|'.join('{} {}'.format(*pair) for pair in run_lengths)


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

