import math
import torch


def build_lr_scheduler(cfg, optimizer, epochs):
    """Build learning rate scheduler from cfg file."""
    print('==============================')
    print('Lr Scheduler: {}'.format(cfg['scheduler']))
    """
        ## 学习策略的超参数
        self.lr_schedule_dict = {
            "scheduler": "cosine",  # 使用YOLOX官方的Cosine衰减策略
            "lrf": 0.05,  # 最终学习率与初始学习率的比值，即最终学习率=lr0 * lrf
        }
    """
    # Cosine LR scheduler
    if cfg['scheduler'] == 'cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg['lrf'] - 1) + 1
    # Linear LR scheduler
    elif cfg['scheduler'] == 'linear':
        """
            线性学习率衰减策略
            lf(0)   = (1 - 0/100) * 0.9 + 0.1 = 1.0 * 0.9 + 0.1 = 1.0
            lf(50)  = (1 - 50/100) * 0.9 + 0.1 = 0.5 * 0.9 + 0.1 = 0.55
            lf(100) = (1 - 100/100) * 0.9 + 0.1 = 0.0 * 0.9 + 0.1 = 0.1

            学习率因子
            1.0 ─┐
                 │╲
                 │ ╲
                 │  ╲
            0.55 ─   ╲
                 │    ╲
            0.1 ─└─────╲
                 0   50  100  epoch

            每个epoch的最终学习率 = 初始学习率 × 每个epoch的学习率因子
        """
        lf = lambda x: (1 - x / epochs) * (1.0 - cfg['lrf']) + cfg['lrf']

    else:
        print('unknown lr scheduler.')
        exit(0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    return scheduler, lf
