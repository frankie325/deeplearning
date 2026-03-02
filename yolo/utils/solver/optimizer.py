'''
Author: frank fugangkuang4@gmail.com
Date: 2026-03-01 22:39:51
LastEditors: frank fugangkuang4@gmail.com
LastEditTime: 2026-03-01 23:05:55
FilePath: /deeplearning/yolo/utils/solver/optimizer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn


def build_yolo_optimizer(cfg, model, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(cfg['optimizer']))
    print('--base lr: {}'.format(cfg['lr0']))
    print('--momentum: {}'.format(cfg['momentum']))
    print('--weight_decay: {}'.format(cfg['weight_decay']))

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(g[2], lr=cfg['lr0'])  # adjust beta1 to momentum
    elif cfg['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(g[2], lr=cfg['lr0'], weight_decay=0.0)
    elif cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(g[2], lr=cfg['lr0'], momentum=cfg['momentum'], nesterov=True)
    else:
        raise NotImplementedError('Optimizer {} not implemented.'.format(cfg['optimizer']))

    optimizer.add_param_group({'params': g[0], 'weight_decay': cfg['weight_decay']})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})                  # add g1 (BatchNorm2d weights)

    start_epoch = 0
    if resume is not None:
        """
        checkpoint = {
            "epoch": 50,              # 当前epoch
            "optimizer": {...},       # 优化器状态（momentum、lr等）
            "model": {...},           # 模型权重
            "scheduler": {...},       # 学习率调度器状态
        }
        """
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("optimizer")
        optimizer.load_state_dict(checkpoint_state_dict)
        start_epoch = checkpoint.pop("epoch")
        del checkpoint, checkpoint_state_dict
                                                        
    return optimizer, start_epoch
