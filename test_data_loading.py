import pytest
from load_data import Data
import numpy as np
from dataset import FrameDataset
from transformation import FrameTransform
import torch


def test_train_data():
    data_path = 'test_data'
    data = Data(data_path, mode='train')
    assert len(data) == 3
    assert len(data.videos[0]) == 4
    assert len(data.video_masks[0]) == 4
    assert data.videos[0][0] == '170908_061502408_Camera_5.jpg'
    assert data.videos[0][3] == '170908_061502825_Camera_5.jpg'
    assert data.video_masks[0][0] == '170908_061502408_Camera_5_instanceIds.png'
    assert data.video_masks[0][2] == '170908_061502686_Camera_5_instanceIds.png'

    assert len(data.videos[1]) == 3
    assert len(data.video_masks[1]) == 3
    assert data.videos[1][0] == '170908_061955478_Camera_5.jpg'
    assert data.videos[1][2] == '170908_061955778_Camera_5.jpg'
    assert data.video_masks[1][0] == '170908_061955478_Camera_5_instanceIds.png'
    assert data.video_masks[1][2] == '170908_061955778_Camera_5_instanceIds.png'

    sample = data.get_frame(0, 0)
    assert 'image' in sample
    assert 'mask' in sample
    assert sample['image'].shape == (2710, 3384, 3)
    assert sample['mask'].shape == (2710, 3384)

    for video_id in range(len(data)):
        img_count = 0
        for sample in data.video_iter(video_id):
            assert sample['image'].shape == (2710, 3384, 3)
            assert sample['mask'].shape == (2710, 3384)
            img_count += 1
        assert img_count == len(data.videos[video_id])


def test_train_data_files():
    data_path = 'test_data'
    names = ['road01_cam_5_video_2_image_list_train']
    data = Data(data_path, mode='train', video_names=names)
    assert len(data) == 1
    assert len(data.videos) == 1
    assert len(data.videos[0]) == 4
    assert len(data.video_masks[0]) == 4


def test_train_test_split():
    data_path = 'test_data'
    data = Data(data_path, mode='train')
    random_state = np.random.RandomState(seed=1)
    data_train, data_test = data.train_test_split(test_size=1, random_state=random_state)
    assert len(data) == len(data_train) + len(data_test)


def test_test_data():
    data_path = 'test_data'
    data = Data(data_path, mode='test')
    assert len(data) == 2
    assert len(data.videos[0]) == 2
    assert data.videos[0][0] == '56dbd8514bd2b8d1566f8977cfeb0406.jpg'
    assert data.videos[0][1] == '632c2285b6c98a9973b87ee22bb3269c.jpg'

    assert len(data.videos[1]) == 3
    assert data.videos[1][0] == 'd5f147ba4d1c8056c7b4525520b1682f.jpg'
    assert data.videos[1][2] == 'fd14c90019ff78722643c89ca2528814.jpg'

    sample = data.get_frame(0, 0)
    assert sample['image'].shape == (2710, 3384, 3)

    for video_id in range(len(data)):
        img_count = 0
        for sample in data.video_iter(video_id):
            img = sample['image']
            assert img.shape == (2710, 3384, 3)
            img_count += 1
        assert img_count == len(data.videos[video_id])


def test_transform():
    size = (400, 512)
    transform = FrameTransform(size=size)
    dtypes = [np.int8, np.int32, np.float32, np.float64]
    for dtype in dtypes:
        sample = {}
        sample['image'] = np.ones((2710, 3384, 3), dtype=dtype)
        sample['mask'] = np.ones((10, 2710, 3384), dtype=dtype)
        newsample = transform.transform(sample)
        assert sample != newsample
        assert sample['image'] != newsample['image']
        assert sample['mask'] != newsample['mask']
        assert newsample['image'].shape == (size[0], size[1], 3)
        assert newsample['mask'].shape == (sample['mask'].shape[0], size[0], size[1])
        assert newsample['image'].dtype == dtype
        assert newsample['mask'].dtype == dtype
        assert newsample['image'].max() == 1.0
        assert newsample['mask'].max() == 1.0


def test_dataset():
    data_path = 'test_data'
    data = Data(data_path, mode='train')
    size = (400, 512)
    sizes = [(2710, 3384), size]
    scale = FrameTransform(size=size)
    transforms = [None, scale.transform]
    for transform, expected_size in zip(transforms, sizes):
        dataset = FrameDataset(data, transform=transform)
        sample = dataset[0]
        assert 'image' in sample
        assert 'mask' in sample
        assert 'bboxes' in sample
        assert 'classes' in sample
        assert sample['image'].type() == torch.FloatTensor(0).type()
        assert sample['mask'].type() == torch.FloatTensor(0).type()
        assert sample['bboxes'].type() == torch.FloatTensor(0).type()
        assert sample['classes'].type() == torch.LongTensor(0).type()
        assert sample['image'].shape[0] == 3
        assert sample['image'].shape[1:] == expected_size
        assert sample['mask'].shape[1] == 1
        assert sample['mask'].shape[2:] == expected_size
