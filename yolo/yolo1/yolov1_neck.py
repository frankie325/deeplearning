import torch
import torch.nn as nn
from .yolov1_basic import Conv


# SPPF模块：空间金字塔池化模块
class SPPF(nn.Module):
    """
    该代码参考YOLOv5的官方代码实现 https://github.com/ultralytics/yolov5

    假设输入512个通道
    先经过1x1卷积，输出256个通道

    然后经过3个5x5最大池化层，通道数不变，分别输出256个通道
    进行拼接256 + 256 + 256 + 256 = 1024

    最后经过1x1卷积，输出512个通道
    """
    
    def __init__(
        self,
        in_dim,
        out_dim,
        expand_ratio=0.5,
        pooling_size=5,
        act_type="lrelu",
        norm_type="BN",
    ):
        super(SPPF, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Conv(in_dim, inter_dim, 1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(
            inter_dim * 4, out_dim, k=1, act_type=act_type, norm_type=norm_type
        )
        self.m = nn.MaxPool2d(
            kernel_size=pooling_size, stride=1, padding=pooling_size // 2
        )

    def farward(self, x):
        x = self.cv1(x)

        # 3个最大池化层
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # 按照通道数进行cat
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


# 搭建Neck网络
def build_neck(cfg, in_dim, out_dim):
    model = cfg["neck"]
    print("==============================")
    print("Neck: {}".format(model))
    # build neck
    if model == "sppf":
        neck = SPPF(
            in_dim=in_dim,
            out_dim=out_dim,
            expand_ratio=cfg["expand_ratio"],
            pooling_size=cfg["pooling_size"],
            act_type=cfg["neck_act"],
            norm_type=cfg["neck_norm"],
        )
    else:
        raise NotImplementedError("Neck {} not implemented.".format(cfg["neck"]))
    
    return neck
