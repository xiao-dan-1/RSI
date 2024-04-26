import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
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

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation")

        # 创建选择图像按钮
        self.select_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        # 显示标签颜色图的标签
        self.legend_image = Image.open("legend.png")
        self.legend_image = ImageTk.PhotoImage(self.legend_image)
        self.legend_label = tk.Label(self.root, image=self.legend_image)
        self.legend_label.pack(side=tk.RIGHT, pady=10)
        
        # 显示原始图像的标签
        self.original_image_label = tk.Label(self.root)
        self.original_image_label.pack(side=tk.LEFT,padx=10)

        # 显示分割后图像的标签
        self.segmented_image_label = tk.Label(self.root)
        self.segmented_image_label.pack(side=tk.RIGHT,padx=10)



    def select_image(self):
        # 打开文件对话框以选择图像文件
        file_path = filedialog.askopenfilename()
        if file_path:
            # 使用OpenCV加载图像
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 将图像转换为适合在Tkinter中显示的格式
            original_image = Image.fromarray(image_rgb)
            original_image = ImageTk.PhotoImage(original_image)

            # 显示原始图像
            self.original_image_label.configure(image=original_image)
            self.original_image_label.image = original_image

            # 调用图像分割函数
            segmented_image = self.segment_image(image)
            #
            # # 将分割后的图像转换为适合在Tkinter中显示的格式
            segmented_image = Image.fromarray(segmented_image)
            segmented_image = ImageTk.PhotoImage(segmented_image)
            #
            # # 显示分割后的图像
            self.segmented_image_label.configure(image=segmented_image)
            self.segmented_image_label.image = segmented_image

    def segment_image(self, image):
        # 此处为示例，你可以使用你喜欢的图像分割模型
        # 在这里使用简单的颜色阈值分割
        # 这里是一个简单的例子，具体的分割算法取决于你要解决的问题和你的数据
        # 实际项目中可能需要更复杂的模型
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                mask_rgb[r, c, :] = rgb_values[decoded_classes[r, c]]


        return mask_rgb


root = tk.Tk()
app = ImageSegmentationApp(root)
root.mainloop()
