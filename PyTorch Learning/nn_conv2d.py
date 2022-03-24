import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=4)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        # 定义一个卷积层
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

tudui = Tudui()
print(tudui)

writer = SummaryWriter("nn")

step = 0
# 将每一张图像方式该卷积层
for data in dataloader:
    imgs,targets = data
    output = tudui(imgs)
    print(output.shape)
    writer.add_images("input",imgs,step)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step +=1

writer.close()
