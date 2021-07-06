import torch
import torch.nn as nn
from torch.nn import functional as F

"""
主要功能：
创建VGG19模型

调用格式：
    model = VGG19(num_classes=1000).cuda().train()
"""
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding#padding为空时，默认为kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=True)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, num_middleconvs):
        super(Conv_block,self).__init__()
        self.First_Conv = Conv(in_channels, out_channels, kernel_size=3, stride=1)
        self.num_blocks=num_middleconvs
        self.Middle_Conv=Conv(out_channels, out_channels, kernel_size=3, stride=1)
        self.maxpooling=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block=nn.Sequential(self.First_Conv)
        for i in range(self.num_blocks):
            self.block.add_module('Middle_Conv'+str(i+1),self.Middle_Conv)
        self.block.add_module('maxpooling',self.maxpooling)
    def forward(self,x):
        x=self.block(x)
        return x

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.conv_block1=Conv_block(3, 64, num_middleconvs=1)
        self.conv_block2=Conv_block(64, 128, num_middleconvs=1)
        self.conv_block3=Conv_block(128, 256, num_middleconvs=3)
        self.conv_block4=Conv_block(256, 512, num_middleconvs=3)
        self.conv_block5=Conv_block(512, 512, num_middleconvs=3)

        self.FL1=nn.Flatten(start_dim=1, end_dim=-1)
        self.Linear1=nn.Linear(512 * 7 * 7, 4096)
        self.ReLu1=nn.ReLU(inplace=True)
        self.Linear2=nn.Linear(4096, 4096)
        self.ReLu2=nn.ReLU(inplace=True)
        self.Linear3=nn.Linear(4096, num_classes)

    def forward(self, x):
        x1=self.conv_block1(x)
        x2 = self.conv_block2(x1)
        x3 = self.conv_block3(x2)
        x4 = self.conv_block4(x3)
        x5=self.conv_block5(x4)

        f1=self.FL1(x5)
        l1=self.Linear1(f1)
        l1=self.ReLu1(l1)
        l2=self.Linear2(l1)
        l2=self.ReLu2(l2)
        output=self.Linear3(l2)
        return output

if __name__ == "__main__":
    inputs = torch.rand((8, 3, 224, 224)).cuda()
    model = VGG19(num_classes=1000).cuda().train()
    print(model)
    outputs = model(inputs)
    print(outputs.shape)