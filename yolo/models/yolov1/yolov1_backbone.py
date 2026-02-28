from typing import Any
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


# ResNet的ImageNet pretrained权重的链接
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


# --------------------- 基础模块 -----------------------


# 3x3卷积，填充为默认1
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


# 1x1 卷积
def conv1x1(in_planes, out_planes, stride=1):
    # 当卷积层后紧跟批量归一化(Batch Normalization)时，偏置项会被BN层的参数抵消，因为BN会对输出进行归一化（减去均值、除以方差），所以此时不需要bias
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 残差块
class BasicBlock(nn.Module):
    expansion = 1

    """
    定义残差块的结构:

    1. 卷积层1: 3x3卷积，步长为1，填充为1

    2. 批量归一化层1: 对卷积层1的输出进行批量归一化

    3. ReLU激活函数

    4. 卷积层2: 3x3卷积，步长为1，填充为1

    5. 批量归一化层2: 对卷积层2的输出进行批量归一化

    6. 下采样层: 如果步长为2，则对输入进行下采样，否则不进行下采样

    7. 短接层: 如果步长为2，则对输入进行下采样，否则不进行下采样
    """

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample  # 步长为2时，输入通道数需要下采样
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 输入和输出相加
        out += identity
        out = self.relu(out)
        return out

# 残差块
class Bottleneck(nn.Module):
    expansion = 4
    """
    新结构中的中间3×3的卷积层首先在一个降维1×1卷积层下减少了计算，然后在另一个1×1的卷积层下做了还原，既保持了精度又减少了计算量。
    第一个1×1的卷积把256维channel降到64维，然后在最后通过1×1卷积恢复，整体上用的参数数目：1x1x256x64 + 3x3x64x64 + 1x1x64x256 = 69632，
    而不使用bottleneck的话就是两个3x3x256的卷积，参数数目: 3x3x256x256x2 = 1179648，差了16.94倍
    对于Bottleneck Design的ResNet通常用于更深的如101这样的网络中，目的是减少计算和参数量
    """
    def __init__(self, in_planes, planes, stride=1, downsample=None) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes,planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

# --------------------- ResNet网络 -----------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 所有残差网络都可以划分为4个块层，结构都类似
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # ResNet-18 和 ResNet-50的残差块中，有卷积步长为2时，输入通道数需要下采样
        if stride != 1 or self.in_planes != planes * block.expansion:
            # 下采样卷积
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: list[Any] = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, C, H, W]
        Output:
            c5: (Tensor) -> [B, C, H/32, W/32]

        32倍率下采样
        """
        c1 = self.conv1(x)  # [B, C, H/2, W/2]
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)  # [B, C, H/4, W/4]

        c2 = self.layer1(c1)  # [B, C, H/4, W/4]
        c3 = self.layer2(c2)  # [B, C, H/8, W/8]
        c4 = self.layer3(c3)  # [B, C, H/16, W/16]
        c5 = self.layer4(c4)  # [B, C, H/32, W/32]
        return c5


# --------------------- 构建ResNet网络的函数 -----------------------
## 搭建ResNet-18网络
def resnet18(pretrained=False, **kwargs):
    """搭建 ResNet-18 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # 从指定 URL 下载并加载权重
        # strict设置为False，因为模型结构与预训练权重不完全匹配，不需要预训练的全连接层权重
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]), strict=False)
    return model


## 搭建ResNet-34网络
def resnet34(pretrained=False, **kwargs):
    """搭建 ResNet-34 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]), strict=False)
    return model

    ## 搭建ResNet-50网络


def resnet50(pretrained=False, **kwargs):
    """搭建 ResNet-50 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]), strict=False)
    return model


## 搭建ResNet-101网络
def resnet101(pretrained=False, **kwargs):
    """搭建 ResNet-101 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]), strict=False)
    return model


## 搭建ResNet-152网络
def resnet152(pretrained=False, **kwargs):
    """搭建 ResNet-152 model.

    Args:
        pretrained (bool): 如果为True，则加载imagenet预训练权重
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]), strict=False)
    return model


## 搭建ResNet网络
def build_backbone(model_name="resnet18", pretrained=False):
    if model_name == "resnet18":
        model = resnet18(pretrained)
        feat_dim = 512  # 网络的最终输出的feature的通道维度为512
    elif model_name == "resnet34":
        model = resnet34(pretrained)
        feat_dim = 512  # 网络的最终输出的feature的通道维度为512
    elif model_name == "resnet50":
        model = resnet50(pretrained)
        feat_dim = 2048  # 网络的最终输出的feature的通道维度为2048
    elif model_name == "resnet101":
        model = resnet101(pretrained)
        feat_dim = 2048  # 网络的最终输出的feature的通道维度为2048
    elif model_name == "resnet152":
        model = resnet152(pretrained)
        feat_dim = 2048  # 网络的最终输出的feature的通道维度为2048

    return model, feat_dim


if __name__ == "__main__":
    # 这是一段测试代码，方便测试能否正常的下载ResNet权重和调用ResNet网络
    # model, feat_dim = build_backbone(model_name="resnet18", pretrained=True)
    # model, feat_dim = build_backbone(model_name="resnet34", pretrained=True)
    # model, feat_dim = build_backbone(model_name="resnet50", pretrained=True)
    # model, feat_dim = build_backbone(model_name="resnet101", pretrained=True)
    model, feat_dim = build_backbone(model_name="resnet152", pretrained=True)

    # 打印模型的结构
    print(model)

    # 输入图像的参数
    batch_size    = 2
    image_channel = 3
    image_height  = 224
    image_width   = 224

    # 随机生成一张图像
    image = torch.randn(batch_size, image_channel, image_height, image_width)

    # 模型推理
    output = model(image)

    # 查看模型的输出的shape
    print(output.shape)