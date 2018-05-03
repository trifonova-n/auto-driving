from torch.utils.data import Dataset
import torch
from utils import extract_bboxes, extract_bynary_masks
import numpy as np

from torchvision.transforms import ToPILImage

class FrameDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.ids = []
        for video_id in range(len(self.data.videos)):
            for frame_id in range(len(self.data.videos[video_id])):
                self.ids.append((video_id, frame_id))

    def __len__(self):
        return self.data.image_count

    def __getitem__(self, idx):
        sample = self.data.get_frame(*self.ids[idx])
        if self.transform:
            sample = self.transform(sample)
        if 'mask' in sample:
            masks, classes = extract_bynary_masks(sample['mask'], self.data.class_map)
            bboxes = extract_bboxes(masks)
            sample['classes'] = torch.from_numpy(classes)
            sample['mask'] = torch.from_numpy(np.expand_dims(masks, axis=1))
            sample['bboxes'] = torch.from_numpy(bboxes)
        sample['image'] = torch.from_numpy(sample['image'].transpose((2, 0, 1))).float().div(255)
        return sample
