import numpy as np
import torch
import cv2
import os
import glob

from matplotlib import pyplot as plt
from torch.utils.data import Dataset, SubsetRandomSampler
from torch.utils.data import random_split
import torch.nn.functional as F
from torchvision import transforms


class Data_Loader(Dataset):
    def __init__(self, data_path, transform=None):
        self.class_item = ["airplane", "brea_soil", "buildings", "cars", "chaparral", "count",
                           "dock", "filed", "grass", "mobile_home", "pavement", "sand",
                           "sea", "ship", "tanks", "trees", "water"]
        self.rgb_values = [[166, 202, 240], [128, 128, 0], [0, 0, 128], [255, 0, 0], [0, 128, 0], [128, 0, 0],
                           [255, 233, 233], [160, 160, 164], [0, 128, 128], [90, 87, 255], [255, 255, 0], [255, 192, 0],
                           [0, 0, 255], [255, 0, 192], [128, 0, 128], [0, 255, 0], [0, 255, 255]]
        self.transform = transform
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = sorted(glob.glob(os.path.join(data_path, 'image/*.tif')))
        self.labels_path = sorted(glob.glob(os.path.join(data_path, 'label/*.png')))

    def split(self, dataset, validation_split, shuffle_dataset=True, random_seed=42):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        return train_sampler, valid_sampler

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def grb_to_label(self, grb_label):
        label = np.zeros(grb_label.shape, dtype=np.uint8)
        for i, rgb in enumerate(self.rgb_values):
            label[np.all(grb_label == rgb, axis=-1)] = i
        label = label[:, :, 0]
        return label

    def mask2one_hot(self, label, num_classes):
        """
       label: 标签图像 # （batch_size, 1, h, w)
       out: 网络的输出
       """
        current_label = label  # （batch_size, 1, h, w) ---> （batch_size, h, w)
        h, w = current_label.shape[0], current_label.shape[1]
        # print(h, w, batch_size)
        one_hots = []
        for i in range(num_classes):
            tmplate = torch.ones(h, w)  # （batch_size, h, w)
            tmplate[current_label != i] = 0
            # tmplate = tmplate.view(17, h, w)  # （batch_size, h, w) --> （batch_size, 1, h, w)
            one_hots.append(tmplate.numpy())
        onehot = torch.tensor(np.array(one_hots))
        return onehot

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = self.labels_path[index]
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # # 将数据转为RGB的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = self.grb_to_label(label)
        label = self.mask2one_hot(label, 17)
        # label = np.expand_dims(label, axis=0)
        # whc -> cwh
        image = np.transpose(image, (2, 0, 1))
        # # 处理标签，将像素值为255的改为1
        # if label.max() > 1:
        #     label = label / 255
        # 随机进行数据增强，为2时不做处理
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


def one_hot2mask(pred):
    pred = pred.data.cpu().permute(0, 2, 3, 1).numpy()
    preds_decoded = np.argmax(pred, axis=3)
    return preds_decoded


def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.imshow(npimg)


if __name__ == "__main__":
    dataset = Data_Loader("../data/train/")
    print("数据个数：", len(dataset))

    train_sampler, valid_sampler = dataset.split(dataset, 0.2)

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4,
                                               sampler=valid_sampler)

    print(f"train len:{len(train_loader)}")
    print(f"valid len:{len(valid_loader)}")

    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    label = labels
    print("images.shape:", images.shape)
    print("label.shape", label.shape)
    label = one_hot2mask(label)
    plt.subplot(121), plt.imshow(np.transpose(images[0], (1, 2, 0)))
    plt.subplot(122), plt.imshow(label[0])
    plt.show()
    #
    # for image, label in train_loader:
    #     print("image.shape:", image.shape)
    #     print("label.shape:", label.shape)
