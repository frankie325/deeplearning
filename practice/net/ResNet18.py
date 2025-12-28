import torch as torch
import torch.nn as nn
from torchsummary import summary

"""
ResNet模型：
输入层：输入图像为3x224x224

块1：
    卷积层1：64个3x7x7的卷积核，步长为2，填充为3，输出特征图为64x112x112
    批量归一化层：对64个特征图进行归一化
    ReLU激活函数：对64个特征图进行激活
    最大池化层：3x3的最大池化核，步长为2，填充为1，输出特征图为64x56x56

残差块1：输入为64x56x56, 输入通道数与输出通道数相同，不改变特征图的尺寸。
    
    卷积层：64个64x3x3的卷积核，步长为1，填充为1，输出特征图为64x56x56
    批量归一化层：对64个特征图进行归一化
    ReLU激活函数：对64个特征图进行激活
    卷积层：64个64x3x3的卷积核，步长为1，填充为1，输出特征图为64x56x56
    批量归一化层
 
    残差连接：输入与输出通道数相同，输入直接与输出相加
    ReLU激活函数

残差块2：同残差块1

残差块3：输入为64x56x56
    卷积层：128个64x3x3的卷积核，步长为2，填充为1，输出特征图为128x28x28
    批量归一化层
    ReLU激活函数
    卷积层：128个128x3x3的卷积核，步长为1，填充为1，输出特征图为128x28x28
    批量归一化层
   
    残差连接：输入通道数为64，输出通道数为128，需要进行1x1卷积核进行通道数匹配，步长为2，填充为0，输出特征图为128x28x28
    ReLU激活函数

残差块4：输入为128x28x28
    卷积层：128个128x3x3的卷积核，步长为1，填充为1，输出特征图为128x28x28
    批量归一化层
    ReLU激活函数

    卷积层：128个128x3x3的卷积核，步长为1，填充为1，输出特征图为128x28x28
    批量归一化层

    残差连接：输入与输出通道数相同，输入直接与输出相加
    ReLU激活函数

残差块5：输入为128x28x28
    卷积层：256个128x3x3的卷积核，步长为2，填充为1，输出特征图为256x14x14
    批量归一化层
    ReLU激活函数
    卷积层：256个256x3x3的卷积核，步长为1，填充为1，输出特征图为256x14x14
    批量归一化层
    
    残差连接：输入通道数为128，输出通道数为256，需要进行1x1卷积核进行通道数匹配，步长为2，填充为0，输出特征图为256x14x14
    ReLU激活函数

残差块6：输入为256x14x14
    卷积层：256个256x3x3的卷积核，步长为1，填充为1，输出特征图为256x14x14
    批量归一化层
    ReLU激活函数
    卷积层：256个256x3x3的卷积核，步长为1，填充为1，输出特征图为256x14x14
    批量归一化层
    
    残差连接：输入与输出通道数相同，输入直接与输出相加
    ReLU激活函数

残差块7：输入为256x14x14
    卷积层：512个256x3x3的卷积核，步长为2，填充为1，输出特征图为512x7x7
    批量归一化层
    ReLU激活函数
    卷积层：512个512x3x3的卷积核，步长为1，填充为1，输出特征图为512x7x7
    批量归一化层
    
    残差连接：输入通道数为256，输出通道数为512，需要进行1x1卷积核进行通道数匹配，步长为2，填充为0，输出特征图为512x7x7
    ReLU激活函数

残差块8：输入为512x7x7
    卷积层：512个512x3x3的卷积核，步长为1，填充为1，输出特征图为512x7x7
    批量归一化层
    ReLU激活函数
    卷积层：512个512x3x3的卷积核，步长为1，填充为1，输出特征图为512x7x7
    批量归一化层
    
    残差连接：输入与输出通道数相同，输入直接与输出相加
    ReLU激活函数

自适应平均池化：将特征图的尺寸从7x7压缩到1x1，输出通道数为512。


"""


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1conv=False, stride=1):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果残差块的输入通道数与批量归一化后的输出通道数不同，需要进行1x1卷积核进行通道数匹配
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        # 残差连接
        if self.conv3:
            x = self.conv3(x)

        y = self.ReLU(y + x)
        return y


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 残差块1、2
        self.b2 = nn.Sequential(
            Residual(in_channels=64, out_channels=64,use_1conv=False, stride=1),
            Residual(in_channels=64, out_channels=64,use_1conv=False, stride=1),
        )

        # 残差块3、4
        self.b3 = nn.Sequential(
            Residual(in_channels=64, out_channels=128,use_1conv=True, stride=2),
            Residual(in_channels=128, out_channels=128,use_1conv=False, stride=1),
        )

        # 残差块5、6
        self.b4 = nn.Sequential(
            Residual(in_channels=128, out_channels=256,use_1conv=True, stride=2),
            Residual(in_channels=256, out_channels=256,use_1conv=False, stride=1),
        )

        # 残差块7、8
        self.b5 = nn.Sequential(
            Residual(in_channels=256, out_channels=512, use_1conv=True, stride=2),
            Residual(in_channels=512, out_channels=512, use_1conv=False, stride=1),
        )   

        # 全连接层
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 输出特征图为512x1x1
            nn.Flatten(),
            # nn.Linear(512, 1000),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    summary(model, (3, 224, 224), batch_size=8)


"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [8, 64, 114, 114]           1,792
       BatchNorm2d-2          [8, 64, 114, 114]             128
              ReLU-3          [8, 64, 114, 114]               0
         MaxPool2d-4            [8, 64, 57, 57]               0
            Conv2d-5            [8, 64, 57, 57]          36,928
       BatchNorm2d-6            [8, 64, 57, 57]             128
              ReLU-7            [8, 64, 57, 57]               0
            Conv2d-8            [8, 64, 57, 57]          36,928
       BatchNorm2d-9            [8, 64, 57, 57]             128
             ReLU-10            [8, 64, 57, 57]               0
         Residual-11            [8, 64, 57, 57]               0
           Conv2d-12            [8, 64, 57, 57]          36,928
      BatchNorm2d-13            [8, 64, 57, 57]             128
             ReLU-14            [8, 64, 57, 57]               0
           Conv2d-15            [8, 64, 57, 57]          36,928
      BatchNorm2d-16            [8, 64, 57, 57]             128
             ReLU-17            [8, 64, 57, 57]               0
         Residual-18            [8, 64, 57, 57]               0
           Conv2d-19           [8, 128, 29, 29]          73,856
      BatchNorm2d-20           [8, 128, 29, 29]             256
             ReLU-21           [8, 128, 29, 29]               0
           Conv2d-22           [8, 128, 29, 29]         147,584
      BatchNorm2d-23           [8, 128, 29, 29]             256
           Conv2d-24           [8, 128, 29, 29]           8,320
             ReLU-25           [8, 128, 29, 29]               0
         Residual-26           [8, 128, 29, 29]               0
           Conv2d-27           [8, 128, 29, 29]         147,584
      BatchNorm2d-28           [8, 128, 29, 29]             256
             ReLU-29           [8, 128, 29, 29]               0
           Conv2d-30           [8, 128, 29, 29]         147,584
      BatchNorm2d-31           [8, 128, 29, 29]             256
             ReLU-32           [8, 128, 29, 29]               0
         Residual-33           [8, 128, 29, 29]               0
           Conv2d-34           [8, 256, 15, 15]         295,168
      BatchNorm2d-35           [8, 256, 15, 15]             512
             ReLU-36           [8, 256, 15, 15]               0
           Conv2d-37           [8, 256, 15, 15]         590,080
      BatchNorm2d-38           [8, 256, 15, 15]             512
           Conv2d-39           [8, 256, 15, 15]          33,024
             ReLU-40           [8, 256, 15, 15]               0
         Residual-41           [8, 256, 15, 15]               0
           Conv2d-42           [8, 256, 15, 15]         590,080
      BatchNorm2d-43           [8, 256, 15, 15]             512
             ReLU-44           [8, 256, 15, 15]               0
           Conv2d-45           [8, 256, 15, 15]         590,080
      BatchNorm2d-46           [8, 256, 15, 15]             512
             ReLU-47           [8, 256, 15, 15]               0
         Residual-48           [8, 256, 15, 15]               0
           Conv2d-49             [8, 512, 8, 8]       1,180,160
      BatchNorm2d-50             [8, 512, 8, 8]           1,024
             ReLU-51             [8, 512, 8, 8]               0
           Conv2d-52             [8, 512, 8, 8]       2,359,808
      BatchNorm2d-53             [8, 512, 8, 8]           1,024
           Conv2d-54             [8, 512, 8, 8]         131,584
             ReLU-55             [8, 512, 8, 8]               0
         Residual-56             [8, 512, 8, 8]               0
           Conv2d-57             [8, 512, 8, 8]       2,359,808
      BatchNorm2d-58             [8, 512, 8, 8]           1,024
             ReLU-59             [8, 512, 8, 8]               0
           Conv2d-60             [8, 512, 8, 8]       2,359,808
      BatchNorm2d-61             [8, 512, 8, 8]           1,024
             ReLU-62             [8, 512, 8, 8]               0
         Residual-63             [8, 512, 8, 8]               0
AdaptiveAvgPool2d-64             [8, 512, 1, 1]               0
          Flatten-65                   [8, 512]               0
           Linear-66                    [8, 10]           5,130
================================================================
Total params: 11,176,970
Trainable params: 11,176,970
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.59
Forward/backward pass size (MB): 524.02
Params size (MB): 42.64
Estimated Total Size (MB): 571.25
----------------------------------------------------------------
"""
