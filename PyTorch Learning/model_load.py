import torch

# 方式1：保存方式1：加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方法2：加载模型
import torchvision.models
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
model = torch.load("vgg16_method2.pth")
print(model)
