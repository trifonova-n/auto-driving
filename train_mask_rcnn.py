import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
tensor2pil = ToPILImage()
from load_data import Data
from dataset import FrameDataset
from transformation import FrameTransform
from visualize import visualize_video
sys.path.append('mask-rcnn')
from maskrcnn import MaskRCNN


data_path = 'data'
data = Data(data_path)
random_state = np.random.RandomState(seed=1)
# reduce data size for test purposes
#data, _ = data.train_test_split(test_size=0.90)
train_data, valid_data = data.train_test_split(random_state=random_state)
print('full data', len(data))
print('train data', len(train_data))
print('valid data', len(valid_data))


BATCH_SIZE = 1
NUM_WORKERS = 4
GPU_NUM = 0
TRAIN_RPN_ONLY = False
MODEL_SAVE_PATH = 'model'
max_object_count = 80


transform = FrameTransform(size=(400, 512))
train_dataset = FrameDataset(train_data, transform=transform.transform, max_object_count=max_object_count)
val_dataset = FrameDataset(valid_data, transform=transform.transform, max_object_count=max_object_count)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# ==================================
# define train, val, test function
# ==================================

from torch.autograd import Variable
from tqdm import tqdm
import time


def train_epoch(model, optimizer, epoch, loader):
    model.train()
    start_time = time.time()
    train_loss = 0

    for sample in tqdm(loader):
        image = sample['image'].cuda()
        gt_cls = sample['classes'].cuda()
        gt_bbox = sample['bboxes'].cuda()
        gt_mask = sample['mask'].cuda()

        optimizer.zero_grad()
        result, loss = model(image, gt_cls, gt_bbox, gt_mask)
        train_loss += loss.data
        loss.backward()
        optimizer.step()

    output_str = "Train epoch: {} \t Train loss: {} \t Time elapse: {}s".format(
        epoch, round(train_loss[0].item() / len(loader), 4), int(time.time() - start_time))
    print(output_str)


def val_epoch(model, epoch, loader):
    model.eval()
    results = []
    val_loss = 0

    for sample in tqdm(loader):
        image = sample['image'].cuda()
        gt_cls = sample['classes'].cuda()
        gt_bbox = sample['bboxes'].cuda()
        gt_mask = sample['mask'].cuda()

        result, loss = model(image, gt_cls, gt_bbox, gt_mask)

        assert BATCH_SIZE == 1
        results.extend(result)
        val_loss += loss.data

    output_str = "Val loss: {}".format(round(val_loss[0].item() / len(loader), 4))
    print(output_str)


def train(model, optimizer, epochs):
    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, epoch, train_dataloader)
        val_epoch(model, epoch, val_dataloader)
        model_path = os.path.join(MODEL_SAVE_PATH, "./maskrcnn_%d.state" % epoch)
        torch.save(model.state_dict(), model_path)


#==========
# training
#==========


with torch.cuda.device(GPU_NUM):
    model = MaskRCNN(num_classes=data.num_classes, pretrained="imagenet").cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=0.0001, weight_decay=0.0001)
    train(model, optimizer, 10)

