import torch
import torch.nn as nn

# --------------------- 基础模块 -----------------------

# 3x3卷积，填充为默认1
def conv3x3(in_places, out_places, stride=1):
    return nn.Conv3d(in_places, out_places, kernel_size=3, stride=stride, padding=1, bias=False)
    
# 1x1 卷积
def conv1x1(in_places, out_places, stride=1):
    # 当卷积层后紧跟批量归一化(Batch Normalization)时，偏置项会被BN层的参数抵消，因为BN会对输出进行归一化（减去均值、除以方差），所以此时不需要bias
    return nn.Conv2d(in_places, out_places, kernel_size=1, stride=stride, bias=False)

# 残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplaces, places, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_places=inplaces, out_places=places, stride)