import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

# shuffle表示图片的选取顺序，一般为True表示选取顺序不同
test_loader = DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

# 测试数据集中第一张图片及target
img,target = test_set[0]
print(img)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,target = data
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step+=1

writer.close()