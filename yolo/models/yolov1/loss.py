import torch
import torch.nn.functional as F

# Criterion类用于完成训练阶段的<标签分配>和<损失计算>两个重要环节


class Criterion(object):
    def __init__(self, cfg, device, num_classes=80):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.loss_obj_weight = cfg["loss_obj_weight"]
        self.loss_cls_weight = cfg["loss_cls_weight"]
        self.loss_box_weight = cfg["loss_box_weight"]

    def __call__(self, pred, target):
        pass


def build_criterion(cfg, device, num_classes):
    criterion = Criterion(cfg=cfg, device=device, num_classes=num_classes)

    return criterion
