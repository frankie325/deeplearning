import torch as torch
import torch.nn as nn
from torchsummary import summary

"""
AlexNet模型：由八层网络层组成的，包括5层卷积层和3层全连接层

输入层：输入层是一个3通道的227x227的图像
卷积层1：卷积层1是一个96通道的11x11的卷积核，步长为4，填充为0，输出特征图为96x55x55
池化层1：池化层1是一个3x3的最大池化核，步长为2，填充为0，输出特征图为96x27x27
卷积层2：卷积层2是一个256通道的5x5的卷积核，步长为1，填充为2，输出特征图为256x27x27
池化层2：池化层2是一个3x3的最大池化核，步长为2，填充为0，输出特征图为256x13x13
卷积层3：卷积层3是一个384通道的3x3的卷积核，步长为1，填充为1，输出特征图为384x13x13
卷积层4：卷积层4是一个384通道的3x3的卷积核，步长为1，填充为1，输出特征图为384x13x13
卷积层5：卷积层5是一个256通道的3x3的卷积核，步长为1，填充为1，输出特征图为256x13x13
池化层3：池化层3是一个3x3的最大池化核，步长为2，填充为0，输出特征图为256x6x6
全连接层1：全连接层1是一个4096个神经元的全连接层，使用ReLU激活函数
全连接层2：全连接层2是一个4096个神经元的全连接层，使用ReLU激活函数
输出层：输出层是一个1000个神经元的全连接层，对应1000个类别

AlexNet与LeNet5的区别：
1. AlexNet使用了更多的卷积层和池化层，从而提高了模型的复杂度和表达能力
2. AlexNet使用了ReLU激活函数，而LeNet5使用了Sigmoid激活函数
3. AlexNet在全连接层中使用了Dropout正则化技术，而LeNet5没有使用
"""


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # 全连接层
        num = 6 * 6 * 256
        # self.fc1 = nn.Linear(num, 4096)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 1000)

        self.fc1 = nn.Linear(num, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.5)

        # 初始化参数，防止梯度消失或爆炸
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))  # 使用Relu函数
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.pool3(x)

        # x的形状为（batch_size, 256, 6, 6）
        x = torch.flatten(
            x, start_dim=1
        )  # 从第二个维度开始展平，batch_size的维度不展平

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    summary(model, (3, 227, 227), batch_size=8)

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [8, 96, 55, 55]          34,944
         MaxPool2d-2            [8, 96, 27, 27]               0
            Conv2d-3           [8, 256, 27, 27]         614,656
         MaxPool2d-4           [8, 256, 13, 13]               0
            Conv2d-5           [8, 384, 13, 13]         885,120
            Conv2d-6           [8, 384, 13, 13]       1,327,488
            Conv2d-7           [8, 256, 13, 13]         884,992
         MaxPool2d-8             [8, 256, 6, 6]               0
            Linear-9                  [8, 4096]      37,752,832
           Linear-10                  [8, 4096]      16,781,312
           Linear-11                  [8, 1000]       4,097,000
================================================================
Total params: 62,378,344
Trainable params: 62,378,344
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.72
Forward/backward pass size (MB): 47.71
Params size (MB): 237.95
Estimated Total Size (MB): 290.39
----------------------------------------------------------------
"""
