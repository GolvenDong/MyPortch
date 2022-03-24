import torch
import torch.nn.functional as F
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]
                      ])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]
                       ])
# 将inputreshape为通道数为1，batch的大小为1，数据维度为5*5
input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
print(input.shape)
print(kernel.shape)

output = F.conv2d(input,kernel,stride=1)
print(output)

output2 = F.conv2d(input,kernel,stride=2)
print(output2)

# padding=1表示将input的H和W各拓宽一列或一行
output_padding_1 = F.conv2d(input,kernel,padding=1,stride=1)
print(output_padding_1)

