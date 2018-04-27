import pytest
from load_data import Data


def test_train_data():
    data_path = 'test_data'
    data = Data(data_path, mode='train')
    assert len(data) == 2
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

    img, mask = data.get_frame(0, 0)
    assert img.shape == (2710, 3384, 3)
    assert mask.shape == (2710, 3384)

    for video_id in range(len(data)):
        img_count = 0
        for img, mask in data.video_iter(video_id):
            assert img.shape == (2710, 3384, 3)
            assert mask.shape == (2710, 3384)
            img_count += 1
        assert img_count == len(data.videos[video_id])


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

    img, = data.get_frame(0, 0)
    assert img.shape == (2710, 3384, 3)

    for video_id in range(len(data)):
        img_count = 0
        for img, in data.video_iter(video_id):
            assert img.shape == (2710, 3384, 3)
            img_count += 1
        assert img_count == len(data.videos[video_id])
