import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 最大池化层:保留输入的特征，减少数据量
dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=4)
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]
                      ],dtype=torch.float32)
# -1表示自动计算batchsize大小
input = torch.reshape(input,(-1,1,5,5))
print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        # ceil_Mode表示向上取
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=False)
    def forward(self,input):
        output = self.maxpool1(input)
        return output
writer = SummaryWriter("logs")
step = 1
tudui = Tudui()
for data in dataloader:
    imgs,targets = data
    writer.add_images("max_pool_input",imgs,step)

    output = tudui(imgs)
    writer.add_images("max_pool_output",output,step)
    step += 1

writer.close()