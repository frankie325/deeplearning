import torch
import torch.nn as nn
from torchsummary import summary


"""
骨干网：论文原文使用前20个卷积层进行预训练
"""


class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入3通道448x448的图像
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )
        # 输出64通道112x112的图像
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )
        # 输出192通道56x56的图像
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )
        # 输出512通道28x28的图像
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )
        # 输出1024通道14x14的图像
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
        )
        # 输出1024通道14x14的图像

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class YOLOV1(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.S = 7  # 网格数
        self.B = 2  # 每个grid cell预测的box数，每个box有5个参数（x, y, w, h, conf）
        self.num_classes = num_classes  # 预测类别数，每个grid cell预测20个类别的概率

        # 骨干网
        self.backbone = BackBone()
        # 输出1024通道14x14的图像
        self.end_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(),
        )
        # 输出1024通道7x7的图像
        self.end_block2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
        )
        # 输出1024通道7x7的图像

        self.fc = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.num_classes)),
            nn.Sigmoid(),  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )
        # 输出形状为(bs, 7*7*30)

    def forward(self, x):
        x = self.backbone(x)
        x = self.end_block1(x)
        x = self.end_block2(x)
        # 输出形状为(bs, 1024, 7, 7)
        x = x.view(-1, 1024 * 7 * 7)
        x = self.fc(x)
        # 形状转换为为(bs, 7, 7, 30)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOV1().to(device)
    summary(model, (3, 448, 448), batch_size=8)

"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param # 
================================================================
            Conv2d-1          [8, 64, 224, 224]           9,472 
         LeakyReLU-2          [8, 64, 224, 224]               0 
         MaxPool2d-3          [8, 64, 112, 112]               0 
            Conv2d-4         [8, 192, 112, 112]         110,784 
         LeakyReLU-5         [8, 192, 112, 112]               0 
         MaxPool2d-6           [8, 192, 56, 56]               0 
            Conv2d-7           [8, 128, 56, 56]          24,704
         LeakyReLU-8           [8, 128, 56, 56]               0
            Conv2d-9           [8, 256, 56, 56]         295,168
        LeakyReLU-10           [8, 256, 56, 56]               0
           Conv2d-11           [8, 256, 56, 56]          65,792
        LeakyReLU-12           [8, 256, 56, 56]               0
           Conv2d-13           [8, 512, 56, 56]       1,180,160
        LeakyReLU-14           [8, 512, 56, 56]               0
        MaxPool2d-15           [8, 512, 28, 28]               0
           Conv2d-16           [8, 256, 28, 28]         131,328
        LeakyReLU-17           [8, 256, 28, 28]               0
           Conv2d-18           [8, 512, 28, 28]       1,180,160
        LeakyReLU-19           [8, 512, 28, 28]               0
           Conv2d-20           [8, 256, 28, 28]         131,328
        LeakyReLU-21           [8, 256, 28, 28]               0
           Conv2d-22           [8, 512, 28, 28]       1,180,160
        LeakyReLU-23           [8, 512, 28, 28]               0
           Conv2d-24           [8, 256, 28, 28]         131,328
        LeakyReLU-25           [8, 256, 28, 28]               0
           Conv2d-26           [8, 512, 28, 28]       1,180,160
        LeakyReLU-27           [8, 512, 28, 28]               0
           Conv2d-28           [8, 256, 28, 28]         131,328
        LeakyReLU-29           [8, 256, 28, 28]               0
           Conv2d-30           [8, 512, 28, 28]       1,180,160
        LeakyReLU-31           [8, 512, 28, 28]               0
           Conv2d-32           [8, 512, 28, 28]         262,656
        LeakyReLU-33           [8, 512, 28, 28]               0
           Conv2d-34          [8, 1024, 28, 28]       4,719,616
        LeakyReLU-35          [8, 1024, 28, 28]               0
        MaxPool2d-36          [8, 1024, 14, 14]               0
           Conv2d-37           [8, 512, 14, 14]         524,800
        LeakyReLU-38           [8, 512, 14, 14]               0
           Conv2d-39          [8, 1024, 14, 14]       4,719,616
        LeakyReLU-40          [8, 1024, 14, 14]               0
           Conv2d-41           [8, 512, 14, 14]         524,800
        LeakyReLU-42           [8, 512, 14, 14]               0
           Conv2d-43          [8, 1024, 14, 14]       4,719,616
        LeakyReLU-44          [8, 1024, 14, 14]               0
         BackBone-45          [8, 1024, 14, 14]               0
           Conv2d-46          [8, 1024, 14, 14]       9,438,208
        LeakyReLU-47          [8, 1024, 14, 14]               0
           Conv2d-48            [8, 1024, 7, 7]       9,438,208
        LeakyReLU-49            [8, 1024, 7, 7]               0
           Conv2d-50            [8, 1024, 7, 7]       9,438,208
        LeakyReLU-51            [8, 1024, 7, 7]               0
           Conv2d-52            [8, 1024, 7, 7]       9,438,208
        LeakyReLU-53            [8, 1024, 7, 7]               0
           Linear-54                  [8, 4096]     205,524,992
        LeakyReLU-55                  [8, 4096]               0
          Dropout-56                  [8, 4096]               0
           Linear-57                  [8, 1470]       6,022,590
          Sigmoid-58                  [8, 1470]               0
================================================================
Total params: 271,703,550
Trainable params: 271,703,550
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 18.38
Forward/backward pass size (MB): 1820.05
Params size (MB): 1036.47
Estimated Total Size (MB): 2874.90
----------------------------------------------------------------
"""
