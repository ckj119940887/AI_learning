import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

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
output = conv(input)
print(output.shape)

# 将浮点数转换为限定范围内的整数
output = torch.clamp(output.int(), min=0, max=255)
output_img = output.permute(1, 2, 0).detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img)
ax[1].imshow(output_img)
plt.show()