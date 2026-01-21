import torch
import torch.nn as nn

# --------------------- 基础模块 -----------------------

# 3x3卷积，填充为1
def conv3x3(in_places, out_places, stride=1):
    return nn.Conv3d(in_places, out_places, kernel_size=3, stride=stride, padding=1, bias=False)
    
# 1x1 卷积
def conv1x1(in_places, out_places, stride=1):
    return nn.Conv2d(in_places, out_places, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplaces, places, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self