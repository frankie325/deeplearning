#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .loss import build_criterion
from .yolov1 import YOLOv1


# 构建 YOLOv1 网络
def build_yolov1(args, cfg, device, num_classes=80, trainable=False, deploy=False):
    print("==============Build YOLOv1================")
    print("Build {} ...".format(args.model.upper()))

    print("Model Configuration: \n", cfg)

    # -------------- 构建YOLOv1 --------------
    model = YOLOv1(
        cfg=cfg,
        device=device,
        img_size=args.img_size,
        num_classes=num_classes,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        trainable=trainable,
        deploy=deploy,
    )

    # -------------- 初始化YOLOv1的pred层参数 --------------
    # Init bias
    init_prob = 0.01
    bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
    # = -log(0.99 / 0.01) = -log(99) ≈ 4.6 作用：通过 sigmoid 函数后输出约 0.01 的初始概率 sigmoid(4.6) ≈ 0.99，sigmoid(-4.6) ≈ 0.01
    # obj pred
    b = model.obj_pred.bias.view(
        1, -1
    )  # （历史写法，其实可以直接使用fill填充）view(1, -1) 确保 bias 是二维的，方便 fill_ 操作，最后再 view(-1) 变回一维
    b.data.fill_(bias_value.item())
    model.obj_pred.bias = torch.nn.Parameter(
        b.view(-1), requires_grad=True
    )  # 将张量标记为模型的可学习参数，使其能被优化器更新。

    # cls pred
    b = model.cls_pred.bias.view(1, -1)
    b.data.fill_(bias_value.item())
    model.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # reg pred 权重为 0，偏置为 1 → 输出初始值接近 1，经过 exp() 后宽高预测约为 e^1 ≈ 2.7，乘以 stride 后是合理的初始框大小
    b = model.reg_pred.bias.view(
        -1,
    )
    b.data.fill_(1.0)
    model.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    w = model.reg_pred.weight
    w.data.fill_(0.0)
    model.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    # -------------- 构建用于计算标签分配和计算损失的Criterion类 --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, device, num_classes)
    print("==============Build YOLOv1================")
    return model, criterion
