from torch.utils.data import Dataset
import torch
from utils import extract_bboxes, extract_bynary_masks
import numpy as np


class FrameDataset(Dataset):
    def __init__(self, data, config, transform=None, max_object_count=20):
        self.data = data
        self.config = config
        self.transform = transform
        self.ids = []
        self.max_object_count = max_object_count
        for video_id in range(len(self.data.videos)):
            for frame_id in range(len(self.data.videos[video_id])):
                self.ids.append((video_id, frame_id))

    def __len__(self):
        return self.data.image_count

    def __getitem__(self, idx):
        sample = self.data.get_frame(*self.ids[idx])
        if 'mask' in sample:
            masks, classes = extract_bynary_masks(sample['mask'],
                                                  self.config.catId_map,
                                                  max_object_count=self.config.max_object_count)
            if len(classes) == 0:
                return None
            sample['mask'] = masks
            if self.transform:
                sample = self.transform(sample)
            bboxes = extract_bboxes(sample['mask'])
            sample['classes'] = torch.from_numpy(classes)
            sample['mask'] = torch.from_numpy(np.expand_dims(sample['mask'], axis=1))
            sample['bboxes'] = torch.from_numpy(bboxes)
        else:
            if self.transform:
                sample = self.transform(sample)
        sample['image'] = torch.from_numpy(sample['image'].transpose((2, 0, 1)))
        return sample
