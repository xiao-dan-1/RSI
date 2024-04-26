import cv2
import gradio as gr
import torch
import numpy as np
import os.path
import matplotlib.pyplot as plt
from model.unet_model import UNet

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
net.load_state_dict(torch.load('2100_100_model.pth', map_location=device))
# 测试模式
net.eval()

def predict(image):
    img = np.transpose(image, (2, 0, 1))
    # 转为tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0)
    print("img_tensor.shape:", img_tensor.shape)
    # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    # 预测
    pred = net(img_tensor)
    pred = pred.data.cpu()[0].permute(1, 2, 0).numpy()
    decoded_classes = np.argmax(pred, axis=2)
    mask_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    for r in range(256):
        for c in range(256):
            mask_rgb[r, c, :] = rgb_values[decoded_classes[r, c]]

    print('pred.shape:', pred.shape)
    print('pred.shape:', mask_rgb.shape)

    return mask_rgb


# 0.95687395
# 255
if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.HTML(
            """
            <h1 align="center">RSI</h1>
            <p align="center">这是一个基于深度学习的遥感图像地物解译系统</p>
            """)
        with gr.Row():
            image_input = gr.Image(height=256,width=256)
            image_output = gr.Image(height=256,width=256)
        image_button = gr.Button("解译")


        image_button.click(predict, inputs=image_input, outputs=image_output)
    demo.launch()