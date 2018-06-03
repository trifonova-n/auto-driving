from skimage.io import imread
import os
from pathlib import Path
import numbers
import numpy as np


class Data(object):
    def __init__(self, path, mode='train', video_names=None):
        self.data_dir = Path(path)
        self.videos = []
        self.video_masks = []
        self.video_name2id = {}
        self.video_id2name = []
        self.mode = mode
        self.image_count = 0
        if mode == 'train':
            self.video_dir = self.data_dir / 'train_color'
            self.mask_dir = self.data_dir / 'train_label'
            if video_names:
                self.load_train_filenames(video_names)
            else:
                self.load_train_filenames(self.data_dir / 'list_train')
        elif mode == 'test':
            self.video_dir = self.data_dir / 'test'
            self.mask_dir = ''
            self.load_test_filenames(self.data_dir / 'list_test_mapping')
        else:
            raise RuntimeError('Unexpected mode ' + mode)

    def load_train_filenames(self, list_dir):
        if isinstance(list_dir, (list, tuple)):
            list_files = sorted([self.data_dir / 'list_train' / (p + '.txt') for p in list_dir])
        else:
            list_files = sorted(list_dir.glob('*.txt'))
        for i, file_path in enumerate(list_files):
            video_files = []
            video_mask_files = []
            with file_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    fields = line.split('\t')
                    img_file = fields[0].split('\\')[-1]
                    mask_file = fields[1].split('\\')[-1]
                    video_files.append(img_file)
                    video_mask_files.append(mask_file)
                    self.image_count += 1
            self.video_name2id[file_path.stem] = i
            self.video_id2name.append(file_path.stem)
            self.videos.append(video_files)
            self.video_masks.append(video_mask_files)

    def load_test_filenames(self, mapping_dir):
        mapping_files = sorted(mapping_dir.glob('*.txt'))
        for file_path in mapping_files:
            video_files = []
            with file_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    fields = line.split()
                    img_file = fields[0] + '.jpg'
                    video_files.append(img_file)
                    self.image_count += 1
            self.videos.append(video_files)

    def get_frame(self, video_id, frame_id):
        sample = {}
        frame = imread(str(self.video_dir / self.videos[video_id][frame_id]))
        frame = frame.astype(np.float32) / 255.
        sample['ImageId'] = self.videos[video_id][frame_id].split('.')[0]
        sample['image'] = frame
        if self.mode == 'train':
            mask = imread(str(self.mask_dir / self.video_masks[video_id][frame_id]))
            sample['mask'] = mask
        return sample

    def video_iter(self, video_id, start_frame=0, end_frame=None):
        if end_frame is None:
            end_frame = len(self.videos[video_id]) - 1
        for frame_id in range(start_frame, end_frame + 1):
            yield self.get_frame(video_id, frame_id)

    def train_test_split(self, test_size=0.25, random_state=None):
        if not random_state:
            random_state = np.random.RandomState()
        if not isinstance(test_size, numbers.Integral):
            test_size = int(self.image_count * test_size)
        test_ids = []
        test_img_count = 0
        while test_img_count < test_size:
            while True:
                i = random_state.randint(0, len(self.videos))
                if not i in test_ids:
                    break
            fields = self.video_id2name[i].split('_')
            for cam in ['5', '6']:
                fields[2] = cam
                cam_vid_name = '_'.join(fields)
                if cam_vid_name in self.video_name2id:
                    id = self.video_name2id[cam_vid_name]
                    test_ids.append(id)
                    test_img_count += len(self.videos[id])
        set_test_ids = set(test_ids)
        train_ids = [i for i in range(len(self.videos)) if i not in set_test_ids]
        train_names = [self.video_id2name[i] for i in train_ids]
        test_names = [self.video_id2name[i] for i in test_ids]
        return Data(self.data_dir, self.mode, train_names), Data(self.data_dir, self.mode, test_names)


    def __len__(self):
        return len(self.videos)
