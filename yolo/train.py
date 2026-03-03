import torch
import argparse
from torchsummary import summary
import wandb

# ----------------- Config Components -----------------
from config import build_dataset_config, build_model_config, build_trans_config

# ----------------- Model Components -----------------
from models import build_model

# ----------------- Train Components -----------------
from engine import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-Tutorial")
    # Basic
    parser.add_argument("--cuda", action="store_true", default=True, help="use cuda.")
    parser.add_argument(
        "-size", "--img_size", default=640, type=int, help="input image size"
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of workers used in dataloading",
    )
    parser.add_argument(
        "--tfboard", action="store_true", default=False, help="use tensorboard"
    )
    parser.add_argument(
        "--save_folder",
        default="./yolo/weights/",
        type=str,
        help="path to save weight",
    )
    parser.add_argument(
        "--eval_first",
        action="store_true",
        default=False,
        help="evaluate model before training.",
    )

    # 是否开启混合精度训练
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=False,
        help="Adopting mix precision training.",
    )

    # 可视化训练阶段的数据
    parser.add_argument(
        "--vis_tgt",
        action="store_true",
        default=False,
        help="visualize training data.",
    )
    parser.add_argument(
        "--vis_aux_loss",
        action="store_true",
        default=False,
        help="visualize aux loss.",
    )

    # Batchsize
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=32,
        type=int,
        help="batch size on all the GPUs.",
    )

    # Epoch
    # 训练最大轮次
    parser.add_argument("--max_epoch", default=300, type=int, help="max epoch.")
    parser.add_argument(
        "--wp_epoch", default=1, type=int, help="warmup epoch."
    )  # 设置前面n个epoch为warmup阶段，默认为1，用于调整学习率
    parser.add_argument(
        "--eval_epoch",
        default=10,
        type=int,
        help="after eval epoch, the model is evaluated on val dataset.",
    )

    # 表示在训练最后几个 epoch 关闭强数据增强
    parser.add_argument(
        "--no_aug_epoch",
        default=20,
        type=int,
        help="cancel strong augmentation.",
    )

    # Model
    parser.add_argument("-m", "--model", default="yolov1", type=str, help="build yolo")
    parser.add_argument(
        "-ct",
        "--conf_thresh",
        default=0.005,
        type=float,
        help="confidence threshold",
    )
    parser.add_argument(
        "-nt", "--nms_thresh", default=0.6, type=float, help="NMS threshold"
    )
    parser.add_argument(
        "--topk", default=1000, type=int, help="topk candidates for evaluation"
    )
    parser.add_argument(
        "-p",
        "--pretrained",
        default=None,
        type=str,
        help="load pretrained weight",
    )

    # resume从之前中断的地方继续训练模型
    parser.add_argument(
        "-r",
        "--resume",
        # default="/home/tipriest/Documents/Lessons/weights/voc/yolov1/yolov1_best.pth",
        default=None,
        type=str,
        help="keep training",
    )

    # Dataset
    # 数据集根目录
    # parser.add_argument(
    #     "--root", default="/Users/frank/code/ai/yolo_data", help="data root"
    # )
    parser.add_argument(
        "--root", default="D:/my code/yolo_data", help="data root"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="voc",
        help="coco, voc, widerface, crowdhuman",
    )
    parser.add_argument(
        "--load_cache",
        action="store_true",
        default=False,
        help="load data into memory.",
    )

    # Train trick
    # 是否开启多尺度训练技巧：动态改变输入图像的尺寸，使模型能够检测不同大小的目标，多尺度训练让模型同时学会检测大小不同的目标。
    parser.add_argument(
        "-ms",
        "--multi_scale",
        action="store_true",
        default=False,
        help="Multi scale",
    )
    parser.add_argument("--ema", action="store_true", default=False, help="Model EMA")

    # 最小的目标框大小，训练阶段目标框大小小于该值的目标框将被过滤掉
    parser.add_argument(
        "--min_box_size",
        default=8.0,
        type=float,
        help="min size of target bounding box.",
    )
    parser.add_argument(
        "--mosaic", default=None, type=float, help="mosaic augmentation."
    )
    parser.add_argument("--mixup", default=None, type=float, help="mixup augmentation.")
    parser.add_argument(
        "--grad_accumulate", default=1, type=int, help="gradient accumulation"
    )

    # DDP train 用于启用 PyTorch 分布式数据并行训练，简称 DDP
    parser.add_argument(
        "-dist",
        "--distributed",
        action="store_true",
        default=False,
        help="distributed training",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="number of distributed processes",
    )
    parser.add_argument("--sybn", action="store_true", default=False, help="use sybn.")
    return parser.parse_args()


def wandb_config_add_args(config, args):
    config.model = args.model
    config.img_size = args.img_size
    config.num_workers = args.num_workers
    config.eval_first = args.eval_first
    config.use_fp16 = args.fp16
    config.batch_size = args.batch_size
    config.max_epoch = args.max_epoch
    config.wp_epoch = args.wp_epoch
    config.eval_epoch = args.eval_epoch
    config.no_aug_epoch = args.no_aug_epoch
    config.conf_thresh = args.conf_thresh
    config.nms_thresh = args.nms_thresh
    config.topk = args.topk
    config.pretrained = args.pretrained
    config.resume = args.resume
    config.dataset = args.dataset
    config.load_cache = args.load_cache
    config.multi_scale = args.multi_scale
    config.ema = args.ema
    config.min_box_size = args.min_box_size
    config.mosaic = args.mosaic
    config.mixup = args.mixup
    config.grad_accumulate = args.grad_accumulate
    config.distributed = args.distributed
    config.world_size = args.world_size
    config.sybn = args.sybn
    return config


def train():
    args = parse_args()
    wandb_config = wandb.config
    wandb_config = wandb_config_add_args(wandb_config, args)
    # print(123)
    print("==============args================")
    print(args)
    print("==============args================")

    world_size = 1

    # 如果args.cuda为True，则使用GPU来训练，否则使用CPU来训练（强烈不推荐）
    if args.cuda:
        print("use GPU to train")
        device = torch.device("cuda")
    else:
        print("use CPU to train")
        device = torch.device("cpu")

    # 构建训练所用到的 Dataset & Model & Transform相关的config变量
    data_cfg = build_dataset_config(args)
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg["trans_type"])

    # 构建YOLO模型.L
    model, criterion = build_model(
        args, model_cfg, device, data_cfg["num_classes"], trainable=True
    )
    # 监视模型的梯度和参数
    wandb.watch(model, criterion, log="all")

    # 将模型切换至train模式
    model = model.to(device).train()
    # 打印模型结构
    # summary(model, (3, args.img_size, args.img_size), batch_size=8)

    # 标记单卡模式的model，方便我们做一些其他的处理，省去了DDP模式下的model.module的判断
    model_without_ddp = model
    # 构建训练所需的Trainer类
    trainer = build_trainer(
        args,
        data_cfg,
        model_cfg,
        trans_cfg,
        device,
        model_without_ddp,
        criterion,
        world_size,
    )

    # --------------------------------- Train: Start ---------------------------------
    ## 如果args.eval_first为True，则在训练开始前，先测试模型的性能
    # if args.eval_first and distributed_utils.is_main_process():
    #     # to check whether the evaluator can work
    #     model_eval = model_without_ddp
    #     trainer.eval(model_eval)

    ## 开始训练我们的模型
    trainer.train(model)
    # --------------------------------- Train: End ---------------------------------

    # 训练完毕后，清空占用的GPU显存
    del trainer
    if args.cuda:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Start a new wandb run to track this script.
    wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="deeplearning_frank",
        # Set the wandb project where this run will be logged.
        project="yolo",
        dir="./yolo",
        
    )

    train()
