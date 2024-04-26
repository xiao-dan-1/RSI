import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
# 示例预测结果张量
pred = torch.randn(17, 256, 256)  # 替换为实际的预测张量，注意这里只有一个样本

# 使用 torch.argmax 获取每个像素点预测概率最大的类别索引
pred_classes = torch.argmax(pred, dim=0)

# 将类别索引转换为彩色图像
colored_pred = np.array(pred_classes, dtype=np.uint8)

# 显示彩色图像
plt.imshow(colored_pred)
plt.axis('off')  # 关闭坐标轴
plt.show()