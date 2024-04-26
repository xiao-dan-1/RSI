import sys

import numpy as np
import torch
from tqdm import tqdm

from model.unet_model import UNet
from train import one_hot2mask
from utils.dataset import Data_Loader
from utils.metric import SegMetric

batch_size = 4
data_path = "data/train/"
# 加载训练集
dataset = Data_Loader(data_path)
train_sampler, valid_sampler = dataset.split(dataset, 0.1)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                           sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                           sampler=valid_sampler)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(n_channels=3, n_classes=17)
net.to(device)
net.load_state_dict(torch.load('300_model.pth', map_location=device))

# 定义评估
Metric = SegMetric(17)
net.eval()
CMatrix = np.zeros((17,17))
with torch.no_grad():
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images,labels = data
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        preds = net(images)
        # 计算acc
        preds_mask, lable_mask = one_hot2mask(preds, labels)
        CMatrix += Metric.genCMatrix(preds_mask, lable_mask)

    # print(CMatrix)
    # print(np.diag(CMatrix))
    Metric.CMatrix = CMatrix
    accurcy = Metric.pixelAccuracy()
    pa = np.round(Metric.pixelAccuracy(), 3)
    cpa = np.round(Metric.classPixelAccuracy(), 3)
    iou = np.round(Metric.Intersection_Over_Union(), 3)
    precision = np.round(Metric.Precision(), 3)
    recall = np.round(Metric.Recall(), 3)
    specificity = np.round(Metric.Specificity(), 3)
    f1 = np.round(Metric.F1_Score(), 3)

    print(f"PA          :{pa}\n"
          f"CPA         :{cpa}\n"
          f"IOU         :{iou}\n"
          f"precision   :{precision}\n"
          f"recall      :{recall}\n"
          f"Specificity :{specificity}\n"
          f"F1_Score    :{f1}\n")
