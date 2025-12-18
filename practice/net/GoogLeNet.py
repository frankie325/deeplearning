import torch as torch
import torch.nn as nn
from torchsummary import summary

"""
GoogLeNet模型：由多个Inception模块组成，每个Inception模块包含多个卷积层和池化层，最后通过全局平均池化层和全连接层进行分类。

输入层：输入层是一个3通道的224x224的图像

块1
    卷积层：64个3x7x7的卷积核，步长为2，填充为3，输出特征图为64x112x112
    池化层：3*3的最大池化层，步长为2，填充为1，输出特征图为64x56x56
块2
    卷积层：64个64x1x1的卷积核，步长为1，填充为0，输出特征图为64x56x56
    卷积层：192个64x3x3的卷积核，步长为1，填充为1，输出特征图为192x56x56
    池化层：3*3的最大池化层，步长为2，填充为1，输出特征图为192x28x28

Inception块1：输入为192x28x28
    路径1：
        卷积层：64个192x1x1的卷积核，步长为1，填充为0，输出特征图为64x28x28
    路径2：
        卷积层：96个192x1x1的卷积核，步长为1，填充为0，输出特征图为96x28x28
        卷积层：128个96x3x3的卷积核，步长为1，填充为1，输出特征图为128x28x28
    路径3：
        卷积层：16个192x1x1的卷积核，步长为1，填充为0，输出特征图为16x28x28
        卷积层：32个16x5x5的卷积核，步长为1，填充为2，输出特征图为32x28x28
    路径4：
        池化层：3*3的最大池化层，步长为1，填充为1，输出特征图为192x28x28
        卷积层：32个192x1x1的卷积核，步长为1，填充为0，输出特征图为32x28x28
    合并：
        将路径1、路径2、路径3、路径4的特征图在通道维度上拼接起来
        输出特征图为（64+128+32+32）=256x28x28

Inception块2：输入为256x28x28
    路径1：
        卷积层：128个256x1x1的卷积核，步长为1，填充为0，输出特征图为128x28x28
    路径2：
        卷积层：128个256x1x1的卷积核，步长为1，填充为0，输出特征图为128x28x28
        卷积层：192个128x3x3的卷积核，步长为1，填充为1，输出特征图为192x28x28
    路径3：
        卷积层：32个256x1x1的卷积核，步长为1，填充为0，输出特征图为32x28x28
        卷积层：96个32x5x5的卷积核，步长为1，填充为2，输出特征图为96x28x28
    路径4：
        池化层：3*3的最大池化层，步长为1，填充为1，输出特征图为256x28x28
        卷积层：64个256x1x1的卷积核，步长为1，填充为0，输出特征图为64x28x28
    合并：
        将路径1、路径2、路径3、路径4的特征图在通道维度上拼接起来
        输出特征图为（128+192+96+64）=480x28x28

最大池化层：3*3的最大池化层，步长为2，填充为1，输出特征图为480x14x14

Inception块3：输入为480x14x14 ... 输出为：512x14x14
Inception块4：输入为512x14x14 ... 输出为：512x14x14
Inception块5：输入为512x14x14 ... 输出为：512x14x14
Inception块6：输入为512x14x14 ... 输出为：528x14x14
Inception块7：输入为528x14x14 ... 输出为：832x14x14

最大池化层：3*3的最大池化层，步长为2，填充为1，输出特征图为832x7x7

Inception块8：输入为832x7x7 ... 输出为：832x7x7
Inception块9：输入为832x7x7 ... 输出为：1024x7x7

全局平均池化层：7*7的全局平均池化层，步长为1，填充为0，输出特征图为1024x1x1

全连接层：
    输入特征数1024x1x1，输出特征数1000


GoogLeNet 模型亮点：
Inception 模块：
    多尺度特征提取：Inception 模块设计用于并行处理信息，通过在同一层中使用不同大小的卷积核（如 1x1、3x3、5x5）和最大池化，能够捕捉到不同尺度的特征。
    降维机制：每个卷积层之前会有一个1x1的卷积层，用于降维，减少计算成本，同时学习通道之间的关系。
    
全局平均池化：
    GoogLeNet 使用全局平均池化层来取代传统的全连接层，这不仅减少了参数量，还避免了过拟合，同时加快了训练速度。
    模块化设计：
    整个网络采用模块化设计，便于层的添加与修改，增加了网络的灵活性和可扩展性。

辅助分类器：
    在网络的中间层加入辅助分类器，为深层网络提供了额外的监督信号，有助于缓解梯度消失问题，使深层网络更容易训练。

深度与参数控制：
    尽管 GoogLeNet 有 22 层深，但由于 Inception 模块的高效设计，整体参数量并没有显著增加，相比于其他深度网络，如 VGGNet，GoogLeNet 的模型大小更小。

1x1 卷积核的应用：
    1x1 的卷积核不仅用于降维，还能在不改变特征图尺寸的情况下增加网络的非线性表达能力，从而改善分类性能。

优化的训练策略：
    包括使用 Batch Normalization（批量归一化）等技巧，进一步提高了网络的训练稳定性和收敛速度。
创新的网络结构：
    GoogLeNet 的结构创新在于它解决了深度网络中的几个关键问题，如参数过多导致的过拟合、计算复杂度过大以及梯度消失现象。
"""


# 定义通用的Inception模块
class Inception(nn.Module):

    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()

        # 路径1：c1x1x1的卷积核，输出通道数为c1
        self.path1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1, stride=1, padding=0)

        # 路径2：c2[0]x1x1卷积核，c2[1]x3x3卷积核。c2表示元组，第一个元素为1x1卷积核的输出通道数，第二个元素为3x3卷积核的输出通道数
        self.path2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1, stride=1, padding=0)
        self.path2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, stride=1, padding=1)

        # 路径3：c3[0]x1x1卷积核，c3[1]x5x5卷积核。c3表示元组，第一个元素为1x1卷积核的输出通道数，第二个元素为5x5卷积核的输出通道数
        self.path3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1, stride=1, padding=0)
        self.path3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, stride=1, padding=2)

        # 路径4:3x3的最大池化，1x1的卷积核。c4为1x1卷积核的输出通道数
        self.path4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.path4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        p1 = torch.relu(self.path1_1(x))
        p2 = torch.relu(self.path2_2(torch.relu(self.path2_1(x))))
        p3 = torch.relu(self.path3_2(torch.relu(self.path3_1(x))))
        p4 = torch.relu(self.path4_2(self.path4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 块1：输出特征图为64x14x14
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 块2：输出特征图为192x28x28
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 块3：输出特征图为480x28x28
        self.b3 = nn.Sequential(
            Inception(in_channels=192, c1=64, c2=(96, 128), c3=(16, 32), c4=32), # 第一个Inception块的输出特征图为256x28x28
            Inception(in_channels=256, c1=128, c2=(128, 192), c3=(32, 96), c4=64), # 第二个Inception块的输出特征图为480x28x28
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 块4：输出特征图为832x14x14
        self.b4 = nn.Sequential(
            Inception(in_channels=480, c1=192, c2=(96, 208), c3=(16, 48), c4=64), # 第三个Inception块的输出特征图为512x14x14
            Inception(in_channels=512, c1=160, c2=(112, 224), c3=(24, 64), c4=64), # 第四个Inception块的输出特征图为512x14x14
            Inception(in_channels=512, c1=128, c2=(128, 256), c3=(24, 64), c4=64), # 第五个Inception块的输出特征图为512x14x14
            Inception(in_channels=512, c1=112, c2=(128, 288), c3=(32, 64), c4=64), # 第六个Inception块的输出特征图为528x14x14
            Inception(in_channels=528, c1=256, c2=(160, 320), c3=(32, 128), c4=128), # 第七个Inception块的输出特征图为832x14x14
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 块5：输出特征图为1024x7x7
        self.b5 = nn.Sequential(
            Inception(in_channels=832, c1=256, c2=(160, 320), c3=(32, 128), c4=128), # 第八个Inception块的输出特征图为832x7x7
            Inception(in_channels=832, c1=384, c2=(192, 384), c3=(48, 128), c4=128), # 第九个Inception块的输出特征图为1024x7x7
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=1000),
        )

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet().to(device)
    summary(model, (3, 224, 224), batch_size=8)

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [8, 64, 112, 112]           9,472
              ReLU-2          [8, 64, 112, 112]               0
         MaxPool2d-3            [8, 64, 56, 56]               0
            Conv2d-4            [8, 64, 56, 56]           4,160
              ReLU-5            [8, 64, 56, 56]               0
            Conv2d-6           [8, 192, 56, 56]         110,784
              ReLU-7           [8, 192, 56, 56]               0
         MaxPool2d-8           [8, 192, 28, 28]               0
            Conv2d-9            [8, 64, 28, 28]          12,352
           Conv2d-10            [8, 96, 28, 28]          18,528
           Conv2d-11           [8, 128, 28, 28]         110,720
           Conv2d-12            [8, 16, 28, 28]           3,088
           Conv2d-13            [8, 32, 28, 28]          12,832
        MaxPool2d-14           [8, 192, 28, 28]               0
           Conv2d-15            [8, 32, 28, 28]           6,176
        Inception-16           [8, 256, 28, 28]               0
           Conv2d-17           [8, 128, 28, 28]          32,896
           Conv2d-18           [8, 128, 28, 28]          32,896
           Conv2d-19           [8, 192, 28, 28]         221,376
           Conv2d-20            [8, 32, 28, 28]           8,224
           Conv2d-21            [8, 96, 28, 28]          76,896
        MaxPool2d-22           [8, 256, 28, 28]               0
           Conv2d-23            [8, 64, 28, 28]          16,448
        Inception-24           [8, 480, 28, 28]               0
        MaxPool2d-25           [8, 480, 14, 14]               0
           Conv2d-26           [8, 192, 14, 14]          92,352
           Conv2d-27            [8, 96, 14, 14]          46,176
           Conv2d-28           [8, 208, 14, 14]         179,920
           Conv2d-29            [8, 16, 14, 14]           7,696
           Conv2d-30            [8, 48, 14, 14]          19,248
        MaxPool2d-31           [8, 480, 14, 14]               0
           Conv2d-32            [8, 64, 14, 14]          30,784
        Inception-33           [8, 512, 14, 14]               0
           Conv2d-34           [8, 160, 14, 14]          82,080
           Conv2d-35           [8, 112, 14, 14]          57,456
           Conv2d-36           [8, 224, 14, 14]         226,016
           Conv2d-37            [8, 24, 14, 14]          12,312
           Conv2d-38            [8, 64, 14, 14]          38,464
        MaxPool2d-39           [8, 512, 14, 14]               0
           Conv2d-40            [8, 64, 14, 14]          32,832
        Inception-41           [8, 512, 14, 14]               0
           Conv2d-42           [8, 128, 14, 14]          65,664
           Conv2d-43           [8, 128, 14, 14]          65,664
           Conv2d-44           [8, 256, 14, 14]         295,168
           Conv2d-45            [8, 24, 14, 14]          12,312
           Conv2d-46            [8, 64, 14, 14]          38,464
        MaxPool2d-47           [8, 512, 14, 14]               0
           Conv2d-48            [8, 64, 14, 14]          32,832
        Inception-49           [8, 512, 14, 14]               0
           Conv2d-50           [8, 112, 14, 14]          57,456
           Conv2d-51           [8, 128, 14, 14]          65,664
           Conv2d-52           [8, 288, 14, 14]         332,064
           Conv2d-53            [8, 32, 14, 14]          16,416
           Conv2d-54            [8, 64, 14, 14]          51,264
        MaxPool2d-55           [8, 512, 14, 14]               0
           Conv2d-56            [8, 64, 14, 14]          32,832
        Inception-57           [8, 528, 14, 14]               0
           Conv2d-58           [8, 256, 14, 14]         135,424
           Conv2d-59           [8, 160, 14, 14]          84,640
           Conv2d-60           [8, 320, 14, 14]         461,120
           Conv2d-61            [8, 32, 14, 14]          16,928
           Conv2d-62           [8, 128, 14, 14]         102,528
        MaxPool2d-63           [8, 528, 14, 14]               0
           Conv2d-64           [8, 128, 14, 14]          67,712
        Inception-65           [8, 832, 14, 14]               0
        MaxPool2d-66             [8, 832, 7, 7]               0
           Conv2d-67             [8, 256, 7, 7]         213,248
           Conv2d-68             [8, 160, 7, 7]         133,280
           Conv2d-69             [8, 320, 7, 7]         461,120
           Conv2d-70              [8, 32, 7, 7]          26,656
           Conv2d-71             [8, 128, 7, 7]         102,528
        MaxPool2d-72             [8, 832, 7, 7]               0
           Conv2d-73             [8, 128, 7, 7]         106,624
        Inception-74             [8, 832, 7, 7]               0
           Conv2d-75             [8, 384, 7, 7]         319,872
           Conv2d-76             [8, 192, 7, 7]         159,936
           Conv2d-77             [8, 384, 7, 7]         663,936
           Conv2d-78              [8, 48, 7, 7]          39,984
           Conv2d-79             [8, 128, 7, 7]         153,728
        MaxPool2d-80             [8, 832, 7, 7]               0
           Conv2d-81             [8, 128, 7, 7]         106,624
        Inception-82            [8, 1024, 7, 7]               0
        AvgPool2d-83            [8, 1024, 1, 1]               0
          Flatten-84                  [8, 1024]               0
           Linear-85                  [8, 1000]       1,025,000
================================================================
Total params: 6,948,872
Trainable params: 6,948,872
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.59
Forward/backward pass size (MB): 456.83
Params size (MB): 26.51
Estimated Total Size (MB): 487.94
----------------------------------------------------------------
"""
