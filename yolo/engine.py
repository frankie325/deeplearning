"""
该python文件包含了若干个训练常用的Trainer类，包括以下两种：
1. YOLOv8Trainer：该Trainer类主要用于训练YOLOv1~v5等YOLO模型，相关参数如optimizer参数、学习策略等均采用默认设置；
2. YoloxTrainer：该Trainer类主要用于训练YOLOX和笔者实现的较为简单的YOLOv7模型，相关参数如optimizer参数、学习策略等均采用默认设置；
读者可以根据自己的需求来调整所使用的Trainer类的参数
"""
import os
from tqdm import tqdm


# ----------------- Dataset Components -----------------
from dataset.build import build_dataset, build_transform

# YOLOv8 Trainer: 主要用于训练YOLOv1~v5等YOLO模型
class Yolov8Trainer(object):
    def __init__(
        self,
        args,
        data_cfg,
        model_cfg,
        trans_cfg,
        device,
        model,
        criterion,
        world_size,
    ):
        # ------------------- 基础参数 -------------------
        self.args = args
        self.epoch = 0
        self.best_map = -1.0
        self.last_opt_step = 0
        self.no_aug_epoch = args.no_aug_epoch
        self.clip_grad = 10
        self.device = device
        self.criterion = criterion
        self.world_size = world_size
        self.heavy_eval = False
        self.second_stage = False

        # 创建路径，用于保存模型的训练文件
        self.path_to_save = os.path.join(
            args.save_folder, args.dataset, args.model
        )

        os.makedirs(self.path_to_save, exist_ok=True)

        # ---------------------------- 构建Dataset、Model和Transforms所需的config变量 ----------------------------
        ## 数据集的config
        self.data_cfg = data_cfg
        ## 模型的config
        self.model_cfg = model_cfg
        ## 数据预处理的config
        self.trans_cfg = trans_cfg

        # ---------------------------- 构建数据预处理 Transform类 ----------------------------
        ## 构建训练(Train)所需的数据预处理
        self.train_transform, self.trans_cfg = build_transform(
            args=self.args,
            trans_config=self.trans_cfg,
            max_stride=self.model_cfg["max_stride"],
            is_train=True,
        )
        ## 构建测试(Evaluate)所需的数据预处理
        self.val_transform, _ = build_transform(
            args=self.args,
            trans_config=self.trans_cfg,
            max_stride=self.model_cfg["max_stride"],
            is_train=False,
        )

        # ---------------------------- 构建Dataset & Dataloader ----------------------------
        ## 构建Dataset，用于读取数据集的图像和标签
        self.dataset, self.dataset_info = build_dataset(
            self.args,
            self.data_cfg,
            self.trans_cfg,
            self.train_transform,
            is_train=True,
        )
    # 训练模型的主函数
    def train(self, train_loader, val_loader):
        for epoch in tqdm(range(self.start_epoch, self.args.max_epoch)):


class YoloxTrainer(object):
    pass


# Build Trainer
def build_trainer(
    args, data_cfg, model_cfg, trans_cfg, device, model, criterion, world_size
):
    if model_cfg["trainer_type"] == "yolov8":
        return Yolov8Trainer(
            args,
            data_cfg,
            model_cfg,
            trans_cfg,
            device,
            model,
            criterion,
            world_size,
        )
    elif model_cfg["trainer_type"] == "yolox":
        return YoloxTrainer(
            args,
            data_cfg,
            model_cfg,
            trans_cfg,
            device,
            model,
            criterion,
            world_size,
        )
    else:
        raise NotImplementedError
