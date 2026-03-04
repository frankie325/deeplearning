# 训练

`train.py`

| 参数名称 | 短参数 | 类型 | 默认值 | 描述 |
|---------|---------|------|---------|------|
| cuda | --cuda | action='store_true' | False | 是否使用GPU训练，默认使用CPU |
| size | --img_size | int | 640 | 输入图像尺寸 |
| num_workers | --num_workers | int | 8 | 数据加载的工作进程数 |
| tfboard | --tfboard | action='store_true' | False | 是否使用TensorBoard |
| save_folder | --save_folder | str | ./yolo/weights/ | 权重保存路径 |
| eval_first | --eval_first | action='store_true' | False | 训练前是否先评估模型 |
| fp16 | --fp16 | action='store_true' | False | 是否开启混合精度训练 |
| vis_tgt | --vis_tgt | action='store_true' | False | 是否可视化训练数据 |
| vis_aux_loss | --vis_aux_loss | action='store_true' | False | 是否可视化辅助损失 |
| batch_size | -bs, --batch_size | int | 32 | 批次大小 |
| max_epoch | --max_epoch | int | 300 | 最大训练轮次 |
| wp_epoch | --wp_epoch | int | 1 | warmup轮次，用于调整学习率 |
| eval_epoch | --eval_epoch | int | 10 | 每隔多少轮评估一次模型 |
| no_aug_epoch | --no_aug_epoch | int | 20 | 训练最后几个轮次关闭强数据增强 |
| model | -m, --model | str | yolov1 | 要构建的YOLO模型类型 |
| conf_thresh | -ct, --conf_thresh | float | 0.005 | 置信度阈值 |
| nms_thresh | -nt, --nms_thresh | float | 0.6 | NMS阈值 |
| topk | --topk | int | 1000 | 评估时的topk候选框数 |
| pretrained | -p, --pretrained | str | None | 预训练权重路径 |
| resume | -r, --resume | str | None | 从指定权重继续训练 |
| root | --root | str | D:/my code/yolo_data | 数据集根目录 |
| dataset | -d, --dataset | str | voc | 数据集类型 (coco, voc, widerface, crowdhuman) |
| load_cache | --load_cache | action='store_true' | False | 是否将数据加载到内存 |
| multi_scale | -ms, --multi_scale | action='store_true' | False | 是否开启多尺度训练 |
| ema | --ema | action='store_true' | False | 是否使用模型EMA（指数移动平均） |
| min_box_size | --min_box_size | float | 8.0 | 最小目标框尺寸，小于该值的目标将被过滤 |
| mosaic | --mosaic | float | None | mosaic数据增强概率 |
| mixup | --mixup | float | None | mixup数据增强概率 |
| grad_accumulate | --grad_accumulate | int | 1 | 梯度累积步数 |
| distributed | -dist, --distributed | action='store_true' | False | 是否开启分布式训练（DDP） |
| dist_url | --dist_url | str | env:// | 分布式训练的URL |
| world_size | --world_size | int | 1 | 分布式训练的进程数 |
| sybn | --sybn | action='store_true' | False | 是否使用SyncBatchNorm |

**使用示例：**
```bash
# 基础训练
python yolo/train.py -m yolov1 -d voc --cuda --batch_size 32

# 使用预训练权重
python yolo/train.py -m yolov1 -p pretrained.pth --cuda

# 从检查点继续训练
python yolo/train.py -m yolov1 -r checkpoint.pth --cuda

# 开启多尺度和EMA
python yolo/train.py -m yolov1 --multi_scale --ema --cuda
```

# 验证参数

`eval.py` 	计算 mAP 等评估指标

| 参数名称 | 短参数 | 类型 | 默认值 | 描述 |
|---------|---------|------|---------|------|
| img_size | -size | int | 640 | 输入图像最大尺寸 |
| cuda | --cuda | action='store_true' | False | 是否使用GPU推理 |
| model | -m, --model | str | yolov1 | 要构建的YOLO模型类型 |
| weight | --weight | str | None | 已训练好的模型权重路径 |
| conf_thresh | -ct, --conf_thresh | float | 0.005 | 置信度阈值 |
| nms_thresh | -nt, --nms_thresh | float | 0.6 | NMS阈值 |
| topk | --topk | int | 1000 | 测试时的topk候选框数 |
| no_decode | --no_decode | action='store_true' | False | 推理时不进行解码处理 |
| fuse_conv_bn | --fuse_conv_bn | action='store_true' | False | 是否融合Conv和BN层 |
| root | --root | str | D:/my code/yolo_data | 数据集根目录 |
| dataset | -d, --dataset | str | coco | 数据集类型 (coco, voc) |
| mosaic | --mosaic | float | None | mosaic数据增强 |
| mixup | --mixup | float | None | mixup数据增强 |
| load_cache | --load_cache | action='store_true' | False | 是否将数据加载到内存 |
| test_aug | -tta, --test_aug | action='store_true' | False | 是否使用测试时增强（TTA） |

**使用示例：**
```bash
# VOC数据集验证
python yolo/eval.py -m yolov1 -d voc --cuda --weight yolov1_last_mosaic_epoch.pth

# COCO数据集验证
python yolo/eval.py -m yolov1 -d coco --cuda --weight yolov1_coco.pth

# 使用测试时增强（TTA）
python yolo/eval.py -m yolov1 -d voc --cuda --weight model.pth --test_aug

# 调整置信度和NMS阈值
python yolo/eval.py -m yolov1 -d voc --cuda --weight model.pth -ct 0.01 -nt 0.5
```

## 测试

`test.py` 	进行模型测试，可视化检测结果

| 参数名称 | 短参数 | 类型 | 默认值 | 描述 |
|---------|---------|------|---------|------|
| img_size | -size | int | 640 | 输入图像最大尺寸 |
| show | --show | action='store_true' | False | 是否显示可视化结果 |
| save | --save | action='store_true' | False | 是否保存可视化结果 |
| cuda | --cuda | action='store_true' | False | 是否使用GPU推理 |
| save_folder | --save_folder | str | det_results/ | 结果保存目录 |
| visual_threshold | -vt | float | 0.4 | 可视化的置信度阈值 |
| window_scale | -ws | float | 1.0 | cv2显示窗口的缩放比例 |
| resave | --resave | action='store_true' | False | 重新保存权重（不含优化器状态） |
| model | -m | str | yolov1 | 要构建的YOLO模型类型 |
| weight | --weight | str | None | 已训练好的模型权重路径 |
| conf_thresh | -ct | float | 0.1 | 置信度阈值 |
| nms_thresh | -nt | float | 0.5 | NMS阈值 |
| topk | --topk | int | 100 | 测试时的topk候选框数 |
| no_decode | --no_decode | action='store_true' | False | 推理时不进行解码处理 |
| fuse_conv_bn | --fuse_conv_bn | action='store_true' | False | 是否融合Conv和BN层 |
| root | --root | str | /mnt/share/ssd2/dataset | 数据集根目录 |
| dataset | -d | str | coco | 数据集类型 (coco, voc) |
| min_box_size | --min_box_size | float | 8.0 | 最小目标框尺寸 |
| mosaic | --mosaic | float | None | mosaic数据增强 |
| mixup | --mixup | float | None | mixup数据增强 |
| load_cache | --load_cache | action='store_true' | False | 是否将数据加载到内存 |

**使用示例：**
```bash
# 测试单张图片并显示结果
python yolo/test.py -m yolov1 -d voc --cuda --weight model.pth --show

# 测试并保存结果
python yolo/test.py -m yolov1 -d voc --cuda --weight model.pth --save

# 调整置信度阈值
python yolo/test.py -m yolov1 --weight model.pth -vt 0.5 --show
```

## demo

`demo.py` 	展示模型推理结果（支持图片、视频、摄像头）

| 参数名称 | 短参数 | 类型 | 默认值 | 描述 |
|---------|---------|------|---------|------|
| img_size | -size | int | 640 | 输入图像最大尺寸 |
| mosaic | --mosaic | float | None | mosaic数据增强 |
| mixup | --mixup | float | None | mixup数据增强 |
| mode | --mode | str | image | 数据来源类型 (image, video, camera) |
| cuda | --cuda | action='store_true' | False | 是否使用GPU推理 |
| path_to_img | --path_to_img | str | dataset/demo/images/ | 图片文件路径 |
| path_to_vid | --path_to_vid | str | dataset/demo/videos/ | 视频文件路径 |
| path_to_save | --path_to_save | str | det_results/demos/ | 检测结果保存路径 |
| vis_thresh | -vt | float | 0.4 | 可视化的置信度阈值 |
| show | --show | action='store_true' | False | 是否显示可视化结果 |
| gif | --gif | action='store_true' | False | 是否生成GIF动图 |
| model | -m | str | yolov1 | 要构建的YOLO模型类型 |
| num_classes | -nc | int | 80 | 类别数量 |
| weight | --weight | str | None | 已训练好的模型权重路径 |
| conf_thresh | -ct | float | 0.1 | 置信度阈值 |
| nms_thresh | -nt | float | 0.5 | NMS阈值 |
| topk | --topk | int | 100 | 测试时的topk候选框数 |
| deploy | --deploy | action='store_true' | False | 是否为部署模式 |
| fuse_repconv | --fuse_repconv | action='store_true' | False | 是否融合RepConv层 |
| fuse_conv_bn | --fuse_conv_bn | action='store_true' | False | 是否融合Conv和BN层 |

**使用示例：**
```bash
# 图片检测
python yolo/demo.py -m yolov1 --cuda --weight model.pth --mode image --show

# 视频检测
python yolo/demo.py -m yolov1 --cuda --weight model.pth --mode video --show

# 摄像头实时检测
python yolo/demo.py -m yolov1 --cuda --weight model.pth --mode camera --show

# 生成GIF
python yolo/demo.py -m yolov1 --cuda --weight model.pth --mode video --gif
```