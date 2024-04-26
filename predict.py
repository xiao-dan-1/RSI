import glob
import numpy as np
import torch
import os
import cv2
from matplotlib import pyplot as plt

from model.unet_model import UNet
from torch.utils.data import Dataset

class_item = ["airplane", "brea_soil", "buildings", "cars", "chaparral", "count",
              "dock", "filed", "grass", "mobile_home", "pavement", "sand",
              "sea", "ship", "tanks", "trees", "water"]
rgb_values = [[166, 202, 240], [128, 128, 0], [0, 0, 128], [255, 0, 0], [0, 128, 0], [128, 0, 0],
              [255, 233, 233], [160, 160, 164], [0, 128, 128], [90, 87, 255], [255, 255, 0], [255, 192, 0],
              [0, 0, 255], [255, 0, 192], [128, 0, 128], [0, 255, 0], [0, 255, 255]]

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=3, n_classes=17)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('300_model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('data/test/*.tif')
    # 遍历素有图片
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取图片
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))

        # 转为tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        print("img_tensor.shape:", img_tensor.shape)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        # pred = np.array(pred.data.cpu()[0])
        pred = pred.data.cpu()[0].permute(1, 2, 0).numpy()
        decoded_classes = np.argmax(pred, axis=2)
        mask_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        for r in range(256):
            for c in range(256):
                mask_rgb[r,c,:] = rgb_values[decoded_classes[r,c]]

        print('pred.shape:', pred.shape)
        print('pred.shape:', mask_rgb.shape)
        # # 处理结果
        # pred[pred >= 0.5] = 255
        # pred[pred < 0.5] = 0
        # 保存图片
        # plt.subplot(121), plt.imshow(np.transpose(img, (1, 2, 0)))
        # plt.subplot(122), plt.imshow(mask_rgb)
        # plt.show()
        mask_rgb = cv2.cvtColor(mask_rgb,cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_res_path, mask_rgb)
