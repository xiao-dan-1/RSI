import numpy as np
np.set_printoptions(linewidth=100000)
"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，
P\L     P    N
P      TP    FP
N      FN    TN

TP = Metric[i,i]
FP = Metric[i,：].sum() - TP
FN = Metric[：,i].sum() - TP
TN = Metric.sum() - TP - FP - FN
"""


class SegMetric(object):
    def __init__(self, n_class):
        self.n_class = n_class
        self.CMatrix = np.zeros((n_class, n_class))

    def genCMatrix(self, a, b):
        n = self.n_class
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  # 核心代码

    def pixelAccuracy(self):
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.CMatrix).sum() / self.CMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # acc = (TP) / TP + FP
        TP_FP = self.CMatrix.sum(axis=1)
        # TP_FP[TP_FP == 0] = 65535
        classAcc = np.diag(self.CMatrix) / TP_FP  # axis=1 行
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def Intersection_Over_Union(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        # S(AUB) = SA + SB - SA∩B
        intersection = np.diag(self.CMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.CMatrix, axis=1) + np.sum(self.CMatrix, axis=0) - np.diag(
            self.CMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        # union[union == 0] = 65535
        IOU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IOU)  # 求各类别IoU的平均
        return IOU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.CMatrix, axis=1) / np.sum(self.CMatrix)
        iu = np.diag(self.CMatrix) / (
                np.sum(self.CMatrix, axis=1) + np.sum(self.CMatrix, axis=0) -
                np.diag(self.CMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Precision(self):
        # Precision = TP /(TP +FP)
        TP = np.diag(self.CMatrix)
        TP_FP = self.CMatrix.sum(axis=0)
        # TP_FP[TP_FP == 0] = 65535
        precision = np.diag(self.CMatrix) / TP_FP
        precision[precision == 0 ] = 1
        return precision

    def Recall(self):
        # TPR=TP / TP + FN
        TP_FN = self.CMatrix.sum(axis=1)
        # TP_FN[TP_FN ==0 ] = 65535
        recall = np.diag(self.CMatrix) / TP_FN
        recall[recall == 0] = 1
        return recall

    def Specificity(self):
        # TNR = TN / (TN + FP)
        specificity = np.zeros((self.n_class))
        for i in range(self.n_class):  # 精确度、召回率、特异度的计算
            TP = self.CMatrix[i, i]
            FP = np.sum(self.CMatrix[i, :]) - TP
            FN = np.sum(self.CMatrix[:, i]) - TP
            TN = np.sum(self.CMatrix) - TP - FP - FN
            specificity[i] = TN / (TN + FP)
        return specificity

    def F1_Score(self):
        # F1 = 2 * (Precision * Recall) /(Precision + Recall)
        Precision_Recall = self.Precision() + self.Recall()
        Precision_Recall[Precision_Recall == 0] = 65535
        f1 = 2 * (self.Precision() * self.Recall()) / Precision_Recall
        return f1

    def addBatch(self, Predict, Label):
        assert Predict.shape == Label.shape
        self.CMatrix = np.zeros((self.n_class, self.n_class))
        self.CMatrix += self.genCMatrix(Predict, Label)


def demo():
    pre = np.array([[0, 1, 0],
                    [2, 1, 0],
                    [2, 2, 1]])

    lab = np.array([[0, 2, 0],
                    [2, 1, 0],
                    [2, 2, 1]])

    Metric = SegMetric(3)
    Metric.addBatch(pre, lab)
    PA = np.round(Metric.pixelAccuracy(), 3)
    CPA = np.round(Metric.classPixelAccuracy(), 3)
    IOU = np.round(Metric.Intersection_Over_Union(), 3)
    precision = np.round(Metric.Precision(), 3)
    recall = np.round(Metric.Recall(), 3)
    Specificity = np.round(Metric.Specificity(), 3)
    F1 = np.round(Metric.F1_Score(), 3)
    print("Matrix:\n", Metric.CMatrix)

    print(f"PA          :{PA}\n"
          f"CPA         :{CPA}\n"
          f"IOU         :{IOU}\n"
          f"precision   :{precision}\n"
          f"recall      :{recall}\n"
          f"Specificity :{Specificity}\n"
          f"F1_Score    :{F1}\n")

if __name__ == "__main__":
    from utils.dataset import Data_Loader
    import torch
    from tqdm import tqdm
    import sys
    from model.unet_model import UNet
    from train import one_hot2mask


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 1
    data_path = "../data/train/"
    # 加载训练集
    dataset = Data_Loader(data_path)
    train_sampler, valid_sampler = dataset.split(dataset, 0.4)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                               sampler=valid_sampler)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=17)
    net.to(device)
    net.load_state_dict(torch.load('../2100_100_model.pth', map_location=device))
    # 定义评估
    Metric = SegMetric(17)

    net.eval()
    CMatrix = np.zeros((17,17))

    with torch.no_grad():
        train_bar = tqdm(valid_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            preds = net(images)
            # 计算acc
            preds_mask, lable_mask = one_hot2mask(preds, labels)
            CMatrix += Metric.genCMatrix(preds_mask,lable_mask)
            print(np.diag(CMatrix))

    print('')
    print(CMatrix)
    print(np.diag(CMatrix))
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





