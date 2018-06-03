import pytest
from auto_driving.load_data import Data
import numpy as np
from auto_driving.dataset import FrameDataset
from auto_driving.transformation import FrameTransform


@pytest.fixture
def dataset_t():
    data_path = 'test_data'
    data = Data(data_path, mode='train')
    from auto_driving.config import Config
    config = Config()
    config.size = (400, 512)
    scale = FrameTransform(config)
    dataset = FrameDataset(data, config, transform=scale.transform)
    return dataset


def test_iou(dataset_t):
    gt_mask = np.array([[1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 0],
                        [2, 2, 3, 3, 3],
                        [2, 2, 3, 3, 3]])
    pass