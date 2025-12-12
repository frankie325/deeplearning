import torch
import torch.nn as nn

"""
LeNet模型介绍：
LeNet是一种用于图像分类的卷积神经网络模型，由Yann LeCun等人在1998年提出。
它由两个卷积层和三个全连接层组成，用于识别手写数字。
LeNet模型的架构如下：
- 输入层：接受28x28的灰度图像
- 卷积层1：6个5x5的卷积核，步长为1，padding为0，输出特征图为6x24x24
- 池化层1：2x2的最大池化核，步长为2，输出特征图为6x12x12
- 卷积层2：16个5x5的卷积核，步长为1，padding为0，输出特征图为16x8x8
- 池化层2：2x2的最大池化核，步长为2，输出特征图为16x4x4
- 全连接层1：120个神经元
- 全连接层2：84个神经元
- 输出层：10个神经元（对应0-9的数字）
"""


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
