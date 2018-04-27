from skimage.io import imread
import os
from pathlib import Path


class Data(object):
    def __init__(self, path, mode='train'):
        self.data_dir = Path(path)
        self.videos = []
        self.video_masks = []
        self.mode = mode
        if mode == 'train':
            self.video_dir = self.data_dir / 'train_color'
            self.mask_dir = self.data_dir / 'train_label'
            self.load_train_filenames(self.data_dir / 'list_train')
        elif mode == 'test':
            list_dir = self.data_dir / 'list_test'
            self.video_dir = self.data_dir / 'test'
            self.mask_dir = ''
            self.load_test_filenames(self.data_dir / 'list_test_mapping')
        else:
            raise RuntimeError('Unexpected mode ' + mode)

    def load_train_filenames(self, list_dir):
        list_files = sorted(list_dir.glob('*.txt'))
        for file_path in list_files:
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
            self.videos.append(video_files)

    def get_frame(self, video_id, frame_id):
        frame = imread(str(self.video_dir / self.videos[video_id][frame_id]))
        if self.mode == 'train':
            mask = imread(str(self.mask_dir / self.video_masks[video_id][frame_id]))
            return frame, mask
        return (frame,)

    def video_iter(self, video_id, start_frame=0, end_frame=None):
        if end_frame is None:
            end_frame = len(self.videos[video_id]) - 1
        for frame_id in range(start_frame, end_frame + 1):
            yield self.get_frame(video_id, frame_id)

    def __len__(self):
        return len(self.videos)
