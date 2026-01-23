import torch
import torch.nn as nn
from .yolov1_basic import Conv

# SPPF模块
class SPPF(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=5):
        super(SPPF, self).__init__()
