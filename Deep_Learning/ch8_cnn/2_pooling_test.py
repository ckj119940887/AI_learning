import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# read picture
img = plt.imread("../data/duck.jpg")
# H * W * C
print(img.shape)

# convert picture to input map
# C * H * W
input = torch.tensor(img).permute(2, 0, 1).float()
print(input.shape)

# define model
# 输入和输出都为3，卷积核为9*9，步幅为3
conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=9, stride=3, padding=0)

# forward
output1 = conv(input)
print(output1.shape)

# define pooling layer
pool = nn.MaxPool2d(kernel_size=6, stride=6, padding=1)
output2 = pool(conv(output1))
print(output2.shape)

# 将浮点数转换为限定范围内的整数
output1 = ((output1 - torch.min(output1)) / (torch.max(output1) - torch.min(output1))) * 255
output1_img = output1.int().permute(1, 2, 0).detach().numpy()

output2 = ((output2 - torch.min(output2)) / (torch.max(output2) - torch.min(output2))) * 255
output2_img = output2.int().permute(1, 2, 0).detach().numpy()

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(img)
ax[1].imshow(output1_img)
ax[2].imshow(output2_img)
plt.show()
