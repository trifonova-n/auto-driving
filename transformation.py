import numpy as np
from skimage.transform import resize


class FrameTransform(object):
    def __init__(self, config):
        self.config = config
        pass

    def resize(self, image, size):
        dtype = image.dtype
        img = resize(image, size, preserve_range=True, mode='constant', cval=0.0)
        return img.astype(dtype)

    def transform(self, sample):
        sample = sample.copy()
        img_height = sample['image'].shape[0]
        img_width = sample['image'].shape[1]
        sample['img_height'] = img_height
        sample['img_width'] = img_width
        img = self.resize(sample['image'], self.config.size)
        sample['image'] = img
        if 'mask' in sample:
            masks = np.zeros((sample['mask'].shape[0], self.config.size[0], self.config.size[1]), dtype=sample['mask'].dtype)
            for i, mask in enumerate(sample['mask']):
                masks[i, :, :] = self.resize(mask, self.config.size)
            sample['mask'] = masks
        return sample

    def reverse_transform(self, sample):
        sample = sample.copy()
        img_height = sample['img_height']
        img_width = sample['img_width']
        img = self.resize(sample['image'], (img_height, img_width))
        sample['image'] = img
        if 'mask' in sample:
            masks = np.zeros((sample['mask'].shape[0], img_height, img_width), dtype=sample['mask'].dtype)
            for i, mask in enumerate(sample['mask']):
                masks[i, :, :] = self.resize(mask, self.config.size)
            sample['mask'] = masks
        if 'probs' in sample:
            probs = np.zeros((sample['probs'].shape[0], img_height, img_width), dtype=sample['probs'].dtype)
            for i, prob in enumerate(sample['probs']):
                probs[i, :, :] = self.resize(prob, self.config.size)
            sample['probs'] = probs
        return sample
