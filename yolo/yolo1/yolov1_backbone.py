import torch
import torch.nn as nn

# --------------------- 基础模块 -----------------------

# 3x3卷积，填充为默认1
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    
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
        self.bn2=  nn.BatchNorm2d(out_planes)
        self.downsample = downsample # 步长为2时，输入通道数需要下采样
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


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes=64, layers[0])
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # ResNet-18 和 ResNet-50的残差块中，卷积步长为2时，输入通道数需要下采样
        if stride !=1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes,stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)


