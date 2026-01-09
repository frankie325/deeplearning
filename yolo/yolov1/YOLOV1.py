import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from data import S, IOU

"""
骨干网：论文原文使用前20个卷积层进行预训练
"""


class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入3通道448x448的图像
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )
        # 输出64通道112x112的图像
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )
        # 输出192通道56x56的图像
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=192, out_channels=128, kernel_size=1, padding=0, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )
        # 输出512通道28x28的图像
        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
        )
        # 输出1024通道14x14的图像
        self.block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1
            ),
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
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2
            ),
            nn.LeakyReLU(),
        )
        # 输出1024通道7x7的图像
        self.end_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1
            ),
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
        # self.fc = nn.Sequential(
        #     nn.Linear(1024 * 7 * 7, 100),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(100, self.S * self.S * (self.B * 5 + self.num_classes)),
        #     nn.Sigmoid(),  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        # )
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

    def init_weight(self):
        #   for name, module in self.named_modules():
        #       print(name, module)
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class YOLOLoss(nn.Module):
    def __init__(self, S=S, B=2, num_classes=20):
        super().__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def MSE(self, predictions, targets, reduction="mean"):
        # reduction表示是否对所有元素求平均，默认是mean
        #   return (predictions.item() - targets.item()) ** 2
        return F.mse_loss(predictions, targets, reduction=reduction)

    def forward(self, predictions, targets):
        # predictions: (bs, S, S, B*5 + num_classes)
        # targets: (bs, S, S, B*5 + num_classes)
        xy_loss = 0  # 坐标损失
        wh_loss = 0  # 宽高损失
        conf_loss = 0  # 置信度损失
        class_loss = 0  # 类别损失

        batch_size = predictions.shape[0]
        for batch in range(batch_size):
            for grid_i in range(self.S):
                for grid_j in range(self.S):
                    ground_true = targets[batch, grid_i, grid_j, :5]  # 真实框
                    box1 = predictions[batch, grid_i, grid_j, :5]  # 预测框1
                    box2 = predictions[batch, grid_i, grid_j, 5:10]  # 预测框2
                    targets_classes = targets[batch, grid_i, grid_j, 10:]  # 真实框类别
                    predictions_classes = predictions[
                        batch, grid_i, grid_j, 10:
                    ]  # 预测框类别
                    has_object = ground_true[4] == 1
                    #   print(ground_true)
                    #   print(box1)
                    #   print(box2)
                    # !grid cell中有物体，即有真实框的中心坐标位于该grid cell中
                    if has_object:
                        # !计算真实框与预测框的IOU
                        box1_iou = IOU(ground_true[0:4], box1[0:4], grid_i, grid_j)
                        box2_iou = IOU(ground_true[0:4], box2[0:4], grid_i, grid_j)

                        if box1_iou > box2_iou:
                            #! 选择置信度大的box作为预测框去拟合真实框，box1为正样本，box2为负样本
                            # !计算坐标损失
                            xy_loss += self.lambda_coord * (
                                self.MSE(box1[0], ground_true[0])
                                + self.MSE(box1[1], ground_true[1])
                            )

                            # !计算宽高损失
                            # torch.sqrt 当预测的宽或高接近 0 时， sqrt 的导数（梯度）会趋向于无穷大，导致梯度爆炸，进而使权重更新为 NaN
                            # 为了避免这个问题，我们在计算宽高损失时，对预测的宽高和真实的宽高都加上一个小的常数 1e-6
                            wh_loss += self.lambda_coord * (
                                self.MSE(
                                    torch.sqrt(box1[2] + 1e-6),
                                    torch.sqrt(ground_true[2] + 1e-6),
                                )
                                + self.MSE(
                                    torch.sqrt(box1[3] + 1e-6),
                                    torch.sqrt(ground_true[3] + 1e-6),
                                )
                            )

                            # !计算置信度损失
                            # 使用预测的置信度 box1[4] 与 真实IOU box1_iou 进行MSE
                            # !IOU box1_iou 是一个标量，需要使用 .detach() 方法将其从计算图中分离出来，避免梯度回传时的错误
                            conf_loss += self.MSE(box1[4], box1_iou.detach())
                            # !负样本，计算没有物体的置信度损失
                            conf_loss += self.lambda_noobj * self.MSE(
                                box2[4], torch.tensor(0.0, device=box2.device)
                            )
                        else:
                            # !box1为负样本，box2为正样本
                            # !计算坐标损失
                            xy_loss += self.lambda_coord * (
                                self.MSE(box2[0], ground_true[0])
                                + self.MSE(box2[1], ground_true[1])
                            )

                            # !计算宽高损失
                            wh_loss += self.lambda_coord * (
                                self.MSE(
                                    torch.sqrt(box2[2] + 1e-6),
                                    torch.sqrt(ground_true[2] + 1e-6),
                                )
                                + self.MSE(
                                    torch.sqrt(box2[3] + 1e-6),
                                    torch.sqrt(ground_true[3] + 1e-6),
                                )
                            )

                            # !计算置信度损失
                            # 使用预测的置信度 box2[4] 与 真实IOU box2_iou 进行MSE
                            conf_loss += self.MSE(box2[4], box2_iou.detach())
                            # !负样本，计算没有物体的置信度损失
                            conf_loss += self.lambda_noobj * self.MSE(
                                box1[4], torch.tensor(0.0, device=box1.device)
                            )

                        class_loss += self.MSE(
                            predictions_classes, targets_classes, reduction="sum"
                        )

                    else:
                        # !grid cell中没有物体，只需要对两个预测box与标签值进行置信度损失计算
                        # print(box1[4], box2[4])
                        # !因为没有物体所以标签值的置信度为0
                        conf_loss += self.lambda_noobj * (
                            self.MSE(box1[4], torch.tensor(0.0, device=box1.device))
                            + self.MSE(box2[4], torch.tensor(0.0, device=box2.device))
                        )

        sum_loss = (xy_loss + wh_loss + conf_loss + class_loss) / batch_size
        #   print(sum_loss)
        return (
            sum_loss,
            xy_loss / batch_size,
            wh_loss / batch_size,
            conf_loss / batch_size,
            class_loss / batch_size,
        )


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
