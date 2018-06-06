import pytest
from auto_driving.load_data import Data
import numpy as np
from auto_driving.dataset import FrameDataset
from auto_driving.transformation import FrameTransform
from auto_driving.utils import extract_binary_masks
from auto_driving.mask import compute_iou, rle_encoding
from auto_driving.mean_ap import Eval
from auto_driving.config import Config


@pytest.fixture
def dataset_t():
    data_path = 'test_data'
    data = Data(data_path, mode='train')
    config = Config()
    config.size = (400, 512)
    scale = FrameTransform(config)
    dataset = FrameDataset(data, config, transform=scale.transform)
    return dataset


@pytest.fixture
def masks1():
    mask = np.array([[1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [2, 4, 3, 3, 3],
                     [2, 2, 3, 3, 3]])
    mask = mask + (mask > 0) * 1000 + (mask > 1) * 1000
    mask, cats = extract_binary_masks(mask, max_object_count=10)
    return mask, cats


@pytest.fixture
def masks2():
    mask = np.array([[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 3, 3, 4],
                     [2, 2, 3, 3, 3]])
    mask = mask + (mask > 0) * 1000 + (mask > 1) * 1000
    mask, cats = extract_binary_masks(mask, max_object_count=10)
    return mask, cats


def test_extract_binary_masks():
    mask = np.array([
        [1000, 1001],
        [2000, 0]
    ])
    max_object_count = 10
    bin_mask, cats = extract_binary_masks(mask, max_object_count=max_object_count)
    assert cats[0] == 1
    assert cats[1] == 1
    assert cats[2] == 2
    for c in cats[3:]:
        assert c == 0
    m1 = bin_mask[cats == 1]
    assert m1.shape[0] == 2
    assert m1.shape[1] == 2
    assert m1.shape[2] == 2
    m1_expected = np.array([
        [
            [1, 0],
            [0, 0]
        ],
        [
            [0, 1],
            [0, 0]
        ]
    ], dtype=np.bool)
    assert (m1 == m1_expected).all()
    m2 = bin_mask[cats == 2]
    m2_expected = np.array([
        [
            [0, 0],
            [1, 0]
        ]
    ], dtype=np.bool)
    assert (m2 == m2_expected).all()


def test_iou(masks1, masks2):
    config = Config()
    dt_masks, dt_cats = masks1
    gt_masks, gt_cats = masks2
    ious = compute_iou(dt_masks, gt_masks)
    #print(ious)
    scores = np.array([0.8, 0.6, 0.5] + [0.0] * (len(dt_cats) - 3))
    evaluator = Eval(config)
    evaluator.params.catIds = [0, 1, 2, 3]
    evaluator.evaluate_img(dt_masks=dt_masks, gt_masks=gt_masks, dt_cats=dt_cats, gt_cats=gt_cats, dt_scores=scores)
    print(evaluator.eval_res)
    assert 1 == 2
    pass


def test_evaluate_img_cat(masks1, masks2):
    config = Config()
    dt_masks, dt_cats = masks1
    gt_masks, gt_cats = masks2
    scores = np.array([0.8, 0.7, 0.6, 0.5, 0.4] + [0.0] * (len(dt_cats) - 5))
    evaluator = Eval(config)
    evaluator.params.catIds = [0, 1, 2, 3]
    dtm_1, scores_1, G_1 = evaluator.evaluate_img_cat(dt_masks[dt_cats == 1], gt_masks[gt_cats == 1], scores[dt_cats == 1])
    assert dtm_1.shape[1] == 1
    assert dtm_1[0, 0] == 1
    for i in range(1, dtm_1.shape[0]):
        assert dtm_1[i, 0] == 0
    assert G_1 == 1

    dtm_2, scores_2, G_2 = evaluator.evaluate_img_cat(dt_masks[dt_cats == 2], gt_masks[gt_cats == 2], scores[dt_cats == 2])
    print(dtm_2)
    print(G_2)

    dtm_3, scores_3, G_3 = evaluator.evaluate_img_cat(dt_masks[dt_cats == 3], gt_masks[gt_cats == 3],
                                                      scores[dt_cats == 3])
    print(dtm_3)
    print(G_3)


def test_evaluate_img(masks1, masks2):
    config = Config()
    dt_masks, dt_cats = masks1
    gt_masks, gt_cats = masks2
    scores = np.array([0.8, 0.7, 0.6, 0.5, 0.4] + [0.0] * (len(dt_cats) - 5))
    evaluator = Eval(config)
    evaluator.params.catIds = [0, 1, 2, 3]

    evaluator.evaluate_img(dt_masks, gt_masks, dt_cats, gt_cats, scores)
    print(evaluator.eval_res)
    pass


def test_mean_avrg_precision(masks1, masks2):
    config = Config()
    dt_masks, dt_cats = masks1
    gt_masks, gt_cats = masks2
    scores = np.array([0.8, 0.7, 0.6, 0.5, 0.4] + [0.0] * (len(dt_cats) - 5))
    evaluator = Eval(config)
    evaluator.params.catIds = [0, 1, 2, 3]

    evaluator.evaluate_img(dt_masks, gt_masks, dt_cats, gt_cats, scores)
    evaluator.accumulate()
    print(evaluator.precision[0])
    print(evaluator.mean_avrg_precision())
