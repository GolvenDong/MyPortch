import torch
import torchvision

# train_data = torchvision.datasets.ImageNet("./dateset",split='train',download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 模型的保存方式1：保存模型结构+模型参数
torch.save(vgg16,"vgg16_method1.pth")

# 模型的保存方式2：保存模型参数（官方推荐）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")



# 方式1的陷阱：加载的时候需要将class重新定义，否则报错
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui,"tuidui_method1.pth")