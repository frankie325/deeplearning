import torch
import torch.nn as nn
from torchsummary import summary

"""
C3D模型：
C3D模型是一种用于视频分类的卷积神经网络模型，由多个卷积层、池化层和全连接层组成。

C3D模型的结构如下：
输入层：接受一个形状为(3, 16, 112, 112)的视频数据，3表示通道数（RGB三通道），16表示视频帧数，112表示视频帧的高度和宽度。

第一个块：
卷积层1：使用64个3D卷积核（3, 3, 3, 3）进行卷积操作，三个方向填充为(1,1,1)，步长为(1,1,1)，输出为(64, 16, 112, 112)
池化层1：使用3D最大池化核（1, 2, 2）进行池化操作，三个方向填充为0，步长为(1,2,2)，输出为(64, 16, 56, 56)

第二个块：
卷积层2：使用128个3D卷积核（64，3, 3, 3）进行卷积操作，三个方向填充为(1,1,1)，步长为(1,1,1)，输出为(128, 16, 56, 56)
池化层2：使用3D最大池化核（2, 2, 2）进行池化操作，三个方向填充为0，步长为(2,2,2)，输出为(128, 8, 28, 28) 

第三个块：
卷积层3：使用256个3D卷积核（128, 3, 3, 3）进行卷积操作，三个方向填充为(1,1,1)，步长为(1,1,1)，输出为(256, 8, 28, 28)
卷积层4：使用256个3D卷积核（256, 3, 3, 3）进行卷积操作，三个方向填充为(1,1,1)，步长为(1,1,1)，输出为(256, 8, 28, 28)
池化层3：使用3D最大池化核（2, 2, 2）进行池化操作，三个方向填充为0，步长为(2,2,2)，输出为(256, 4, 14, 14) 

第四个块：
卷积层5：使用512个3D卷积核（256, 3, 3, 3）进行卷积操作，三个方向填充为(1,1,1)，步长为(1,1,1)，输出为(512, 4, 14, 14)
卷积层6：使用512个3D卷积核（512, 3, 3, 3）进行卷积操作，三个方向填充为(1,1,1)，步长为(1,1,1)，输出为(512, 4, 14, 14)
池化层4：使用3D最大池化核（2, 2, 2）进行池化操作，三个方向填充为0，步长为(2,2,2)，输出为(512, 2, 7, 7) 

第五个块：
卷积层7：使用512个3D卷积核（512, 3, 3, 3）进行卷积操作，三个方向填充为(1,1,1)，步长为(1,1,1)，输出为(512, 2, 7, 7)
卷积层8：使用512个3D卷积核（512, 3, 3, 3）进行卷积操作，三个方向填充为(1,1,1)，步长为(1,1,1)，输出为(512, 2, 7, 7)
池化层5：使用3D最大池化核（2, 2, 2）进行池化操作，三个方向填充为(0,1,1)，步长为(2,2,2)，输出为(512, 1, 4, 4) 

全连接层：
输入展平：512 * 1 * 4 * 4 = 8192
全连接层1：8192 -> 4096
全连接层2：4096 -> 4096
全连接层3：4096 -> num_classes
"""

class C3D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))  

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))
        self.conv4 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        self.conv5 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))
        self.conv6 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        self.conv7 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))
        self.conv8 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2,2,2), padding=(0,1,1), stride=(2,2,2))

        self.fc1 = nn.Linear(512 * 1 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool3(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool4(x)

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.pool5(x)
        # x的输入为(Batch_Size, 512, 1, 4, 4)
        # 展平为(Batch_Size, 512 * 1 * 4 * 4)
        # -1充当Batch_Size的占位符
        x = x.view(-1, 512 * 1 * 4 * 4)

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C3D(num_classes=101).to(device)
    summary(model, (3, 16, 112, 112), batch_size=8)

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1      [8, 64, 16, 112, 112]           5,248
              ReLU-2      [8, 64, 16, 112, 112]               0
         MaxPool3d-3        [8, 64, 16, 56, 56]               0
            Conv3d-4       [8, 128, 16, 56, 56]         221,312
              ReLU-5       [8, 128, 16, 56, 56]               0
         MaxPool3d-6        [8, 128, 8, 28, 28]               0
            Conv3d-7        [8, 256, 8, 28, 28]         884,992
              ReLU-8        [8, 256, 8, 28, 28]               0
            Conv3d-9        [8, 256, 8, 28, 28]       1,769,728
             ReLU-10        [8, 256, 8, 28, 28]               0
        MaxPool3d-11        [8, 256, 4, 14, 14]               0
           Conv3d-12        [8, 512, 4, 14, 14]       3,539,456
             ReLU-13        [8, 512, 4, 14, 14]               0
           Conv3d-14        [8, 512, 4, 14, 14]       7,078,400
             ReLU-15        [8, 512, 4, 14, 14]               0
        MaxPool3d-16          [8, 512, 2, 7, 7]               0
           Conv3d-17          [8, 512, 2, 7, 7]       7,078,400
             ReLU-18          [8, 512, 2, 7, 7]               0
           Conv3d-19          [8, 512, 2, 7, 7]       7,078,400
             ReLU-20          [8, 512, 2, 7, 7]               0
        MaxPool3d-21          [8, 512, 1, 4, 4]               0
           Linear-22                  [8, 4096]      33,558,528
             ReLU-23                  [8, 4096]               0
          Dropout-24                  [8, 4096]               0
           Linear-25                  [8, 4096]      16,781,312
             ReLU-26                  [8, 4096]               0
          Dropout-27                  [8, 4096]               0
           Linear-28                   [8, 101]         413,797
================================================================
Total params: 78,409,573
Trainable params: 78,409,573
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 18.38
Forward/backward pass size (MB): 3116.57
Params size (MB): 299.11
Estimated Total Size (MB): 3434.05
----------------------------------------------------------------
"""
