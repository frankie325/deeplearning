#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch

# YOLO series
from .yolov1.build import build_yolov1


def build_model(
    args,
    model_cfg,  # 模型参数配置
    device,
    num_classes=80,
    trainable=False,
    deploy=False,
):

    # YOLOv1
    if args.model == "yolov1":
        model, criterion = build_yolov1(
            args, model_cfg, device, num_classes, trainable, deploy
        )

    if trainable:
        return model, criterion
    else:
        return model
