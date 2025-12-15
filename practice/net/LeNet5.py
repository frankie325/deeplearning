import torch as torch
import torch.nn as nn
from torchsummary import summary

"""
LeNet5模型介绍：
LeNet5是一种用于图像分类的卷积神经网络模型，由Yann LeCun等人在1998年提出。
它由两个卷积层和三个全连接层组成，用于识别手写数字。
LeNet5模型的架构如下：
    输入层：接受32x32的灰度图像（1通道）
    卷积层1：6个5x5的卷积核，步长为1，padding为0，输出特征图为6x28x28
    池化层1：2x2的平均池化核，步长为2，输出特征图为6x14x14
    卷积层2：16个5x5的卷积核，步长为1，padding为0，输出特征图为16x10x10
    池化层2：2x2的平均池化核，步长为2，输出特征图为16x5x5
    全连接层1：120个神经元
    全连接层2：84个神经元
    输出层：10个神经元（对应0-9的数字）
"""


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积层1：输入通道为1（灰度图像），卷积核大小为5x5，输出特征图为6x28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # 池化层1：2x2的平均池化核，步长为2，输出特征图为6x14x14
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层2：输入通道为6（来自池化层1），卷积核大小为5x5，输出特征图为16x10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 池化层2：2x2的平均池化核，步长为2，输出特征图为16x5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 全连接层
        num = 16 * 5 * 5
        self.fc1 = nn.Linear(num, 120)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        # LeNet以前是用的sigmoid函数，那个时候还没有ReLU函数
        x = self.pool1(torch.sigmoid(self.conv1(x)))
        x = self.pool2(torch.sigmoid(self.conv2(x)))
        # x的形状为（batch_size, 16, 5, 5）
        x = torch.flatten(x, 1)  # 从第二个维度开始展平，batch_size的维度不展平

        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.output(x)
        return x


if __name__ == "__main__":
    # 创建模型，部署gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    summary(model, (1, 32, 32), batch_size=8)
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [8, 6, 28, 28]             156
         AvgPool2d-2            [8, 6, 14, 14]               0
            Conv2d-3           [8, 16, 10, 10]           2,416
         AvgPool2d-4             [8, 16, 5, 5]               0
            Linear-5                  [8, 120]          48,120
            Linear-6                   [8, 84]          10,164
            Linear-7                   [8, 10]             850
================================================================
Total params: 61,706
Trainable params: 61,706
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 0.24
Estimated Total Size (MB): 0.30
----------------------------------------------------------------
"""
