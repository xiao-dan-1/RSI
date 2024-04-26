import sys

import numpy as np
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from model.unet_model import UNet
from utils.dataset import Data_Loader
from utils.metric import SegMetric
from torch import optim
import torch.nn as nn
import torch
from collections import defaultdict
import torch.nn.functional as F

from tensorboardX import SummaryWriter

writer = SummaryWriter('./runs')


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def calc_loss(pred, target, bce_weight=0.8):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

def one_hot2mask(pred, target):
    pred = pred.data.cpu().permute(0, 2, 3, 1).numpy()
    target = target.data.cpu().permute(0, 2, 3, 1).numpy()
    preds_decoded = np.argmax(pred, axis=3)
    labels_decoded = np.argmax(target, axis=3)
    return preds_decoded, labels_decoded


def Accuracy(pred, target):
    pred_mask, target_mask = one_hot2mask(pred, target)
    batch, c, w, h = pred.shape
    accuracy = (pred_mask == target_mask).sum() / (batch * w * h)
    return accuracy


def train_net(net, device, data_path, epochs=300, batch_size=8, lr=1e-4):
    # 加载训练集
    dataset = Data_Loader(data_path)
    train_sampler, valid_sampler = dataset.split(dataset, 0.2)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                               sampler=valid_sampler)
    # 优化器
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-6)
    # 定义Loss算法
    criterion = calc_loss  # nn.BCEWithLogitsLoss() 、nn.CrossEntropyLoss()
    # 定义评估
    Metric = SegMetric(17)
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        print(f"-------第 {epoch + 1} 轮训练开始-------")
        train_bar = tqdm(train_loader, file=sys.stdout)
        net.train()  # 训练模式
        # 按照batch_size开始训练
        running_loss = 0.0
        total_accuracy = 0.0
        for step, data in enumerate(train_bar):
            image, label = data
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), f'{epoch}_model.pth')
            # 计算acc
            pred_mask, label_mask = one_hot2mask(pred, label)
            Metric.addBatch(pred_mask, label_mask)
            accuracy = Metric.pixelAccuracy()

            total_accuracy += accuracy
            # 优化器优化模型# 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(epoch + 1, epochs, loss,
                                                                                       accuracy)
        train_loss = running_loss / len(train_loader)
        train_accuracy = total_accuracy / len(train_loader)

        # 测试
        net.eval()
        total_eval_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)
                preds = net(images)
                loss = criterion(preds, labels)
                total_eval_loss += loss.item()
                # 计算acc
                preds_mask, lable_mask = one_hot2mask(preds, labels)
                Metric.addBatch(preds_mask, lable_mask)
                accurcy = Metric.pixelAccuracy()
                PA = np.round(Metric.pixelAccuracy(), 3)
                CPA = np.round(Metric.classPixelAccuracy(), 3)
                IOU = np.round(Metric.Intersection_Over_Union(), 3)
                precision = np.round(Metric.Precision(), 3)
                recall = np.round(Metric.Recall(), 3)
                Specificity = np.round(Metric.Specificity(), 3)
                F1 = np.round(Metric.F1_Score(), 3)
                total_accuracy = total_accuracy + accurcy
        val_loss = total_eval_loss / len(valid_loader)
        val_accuracy = total_accuracy / len(valid_loader)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, train_loss, val_accuracy))
        print(f"PA          :{PA}\n"
              f"CPA         :{CPA}\n"
              f"IOU         :{IOU}\n"
              f"precision   :{precision}\n"
              f"recall      :{recall}\n"
              f"Specificity :{Specificity}\n"
              f"F1_Score    :{F1}\n")

        writer.add_scalar('data/train_accuracy', train_accuracy, epoch)
        writer.add_scalar('data/train_loss', train_loss, epoch)
        writer.add_scalar('data/val_loss', val_loss, epoch)
        writer.add_scalar('data/val_accuracy', val_accuracy, epoch)
    writer.close()


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=3, n_classes=17)

    input = torch.randn((2, 3, 256, 256))
    writer.add_graph(net, input_to_model=input)
    # # 将网络拷贝到deivce中
    net.to(device=device)
    # # 指定训练集地址，开始训练
    data_path = "data/train/"
    train_net(net, device, data_path)
