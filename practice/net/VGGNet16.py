import torch as torch
import torch.nn as nn
from torchsummary import summary

"""
VGGNet16模型：由16层网络层组成的，包括13层卷积层和3层全连接层

输入层：输入层是一个3通道的224x224的图像

第一个VGG block层：
    包含2个卷积层，每个卷积层后面跟着一个ReLU激活函数，然后是一个最大池化层。
    第一个卷积层的核大小为3x3x3，填充为1，步长为1，共64个卷积核，输出特征图为64x224x224
    第二个卷积层的核大小为3x3x3，填充为1，步长为1，共64个卷积核，输出特征图为64x224x224
    最大池化层的核大小为2x2，填充为0，步长为2。输出特征图为64x112x112

第二个VGG block层：
    包含2个卷积层，每个卷积层后面跟着一个ReLU激活函数，然后是一个最大池化层。
    第一个卷积层的核大小为64x3x3，填充为1，步长为1，共128个卷积核，输出特征图为128x112x112
    第二个卷积层的核大小为128x3x3，填充为1，步长为1，共128个卷积核，输出特征图为128x112x112
    最大池化层的核大小为2x2，填充为0，步长为2。输出特征图为128x56x56

第三个VGG block层：
    包含2个卷积层，每个卷积层后面跟着一个ReLU激活函数，然后是一个最大池化层。
    第一个卷积层的核大小为128x3x3，填充为1，步长为1，共256个卷积核，输出特征图为256x56x56
    第二个卷积层的核大小为256x3x3，填充为1，步长为1，共256个卷积核，输出特征图为256x56x56
    最大池化层的核大小为2x2，填充为0，步长为2。输出特征图为256x28x28

第四个VGG block层：
    包含2个卷积层，每个卷积层后面跟着一个ReLU激活函数，然后是一个最大池化层。
    第一个卷积层的核大小为256x3x3，填充为1，步长为1，共512个卷积核，输出特征图为512x28x28
    第二个卷积层的核大小为512x3x3，填充为1，步长为1，共512个卷积核，输出特征图为512x28x28
    最大池化层的核大小为2x2，填充为0，步长为2。输出特征图为512x14x14

第五个VGG block层：
    包含2个卷积层，每个卷积层后面跟着一个ReLU激活函数，然后是一个最大池化层。
    第一个卷积层的核大小为512x3x3，填充为1，步长为1，共512个卷积核，输出特征图为512x14x14
    第二个卷积层的核大小为512x3x3，填充为1，步长为1，共512个卷积核，输出特征图为512x14x14
    最大池化层的核大小为2x2，填充为0，步长为2。输出特征图为512x7x7

全连接层：在全连接层中使用了Dropout正则化技术
    包含3个全连接层，每个全连接层后面跟着一个ReLU激活函数，最后一个全连接层输出1000个类别的概率分布。
    第一个全连接层的输入特征数为512x7x7，输出特征数为4096
    第二个全连接层的输入特征数为4096，输出特征数为4096
    第三个全连接层的输入特征数为4096，输出特征数为1000
"""


class VGGNet16(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1000),
        )

        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features=512 * 7 * 7, out_features=256),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=256, out_features=128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=128, out_features=10),
        # )

        # 初始化参数，防止梯度消失或爆炸
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGNet16().to(device)
    summary(model, (3, 224, 224), batch_size=8)


"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [8, 64, 224, 224]           1,792
              ReLU-2          [8, 64, 224, 224]               0
            Conv2d-3          [8, 64, 224, 224]          36,928
              ReLU-4          [8, 64, 224, 224]               0
         MaxPool2d-5          [8, 64, 112, 112]               0
            Conv2d-6         [8, 128, 112, 112]          73,856
              ReLU-7         [8, 128, 112, 112]               0
            Conv2d-8         [8, 128, 112, 112]         147,584
              ReLU-9         [8, 128, 112, 112]               0
        MaxPool2d-10           [8, 128, 56, 56]               0
           Conv2d-11           [8, 256, 56, 56]         295,168
             ReLU-12           [8, 256, 56, 56]               0
           Conv2d-13           [8, 256, 56, 56]         590,080
             ReLU-14           [8, 256, 56, 56]               0
        MaxPool2d-15           [8, 256, 28, 28]               0
           Conv2d-16           [8, 512, 28, 28]       1,180,160
             ReLU-17           [8, 512, 28, 28]               0
           Conv2d-18           [8, 512, 28, 28]       2,359,808
             ReLU-19           [8, 512, 28, 28]               0
        MaxPool2d-20           [8, 512, 14, 14]               0
           Conv2d-21           [8, 512, 14, 14]       2,359,808
             ReLU-22           [8, 512, 14, 14]               0
           Conv2d-23           [8, 512, 14, 14]       2,359,808
             ReLU-24           [8, 512, 14, 14]               0
        MaxPool2d-25             [8, 512, 7, 7]               0
          Flatten-26                 [8, 25088]               0
           Linear-27                  [8, 4096]     102,764,544
             ReLU-28                  [8, 4096]               0
          Dropout-29                  [8, 4096]               0
           Linear-30                  [8, 4096]      16,781,312
             ReLU-31                  [8, 4096]               0
          Dropout-32                  [8, 4096]               0
           Linear-33                  [8, 1000]       4,097,000
================================================================
Total params: 133,047,848
Trainable params: 133,047,848
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.59
Forward/backward pass size (MB): 1591.00
Params size (MB): 507.54
Estimated Total Size (MB): 2103.13
----------------------------------------------------------------
"""
