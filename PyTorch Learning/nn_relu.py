import torchvision
import torch
from torch import nn
from torch.nn import ReLU
from torch.utils.tensorboard import SummaryWriter

# ReLU：小于0全为0，大于0还为本身
from torch.utils.data import DataLoader

input = torch.tensor([[1,-0.5],
                      [-1,3]
                      ])
output = torch.reshape(input,(-1,1,2,2))
print(output.shape)
dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=4)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        # inplace = True 表示在原来的地方进行结果替换，False表示不进行替换
        self.relu1 = ReLU()
    def forward(self,input):
        output = self.relu1(input)
        return output
writer = SummaryWriter("logs")
tudui = Tudui()
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("relu_input",imgs,step)
    output = tudui.relu1(imgs)
    writer.add_images("relu_output",output,step)
    step += 1

writer.close()

print(output)
