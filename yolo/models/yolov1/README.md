

# 训练参数

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

# YOLOV1 整体训练流程

## 数据预处理

使用 VOC2007 数据集

```
yolo_data/
├── VOC2007/
│   ├── JPEGImages/          # 原始图像
│   │   ├── 000005.jpg
│   │   └── ...
│   ├── Annotations/         # XML标注文件
│   │   ├── 000005.xml
│   │   └── ...
│   └── ImageSets/Main/     # 数据集划分
│       ├── train.txt
│       ├── val.txt
│       └── trainval.txt
└── VOC2012/
    └── ... (相同结构)
```

经过预处理数据增强，dataset 格式为

```dataset数据格式为（image, target）：
    下面的坐标参数，都为经过了图像增强、变换处理后的坐标
    image.shape = [C, H, W]
    target = {
       'boxes':
           [
              [x1, y1, x2, y2],  # 目标1的边界框
              [x1, y1, x2, y2],  # 目标2的边界框
              [x1, y1, x2, y2],  # 目标3的边界框
           ],
        'labels':
           [
              [0],  # 目标1的类别标签索引
              [2],  # 目标2的类别标签索引
              [3],  # 目标3的类别标签索引
           ],
        'orig_size': [height, width] #图形变化之前的宽高
    }
```

经过 dataloader 组装后，格式为

```
    每一批数据格式为：
    images.shape: [B, C, H, W] 为Tensor张量
    targets:  [target1, target2, ..., targetB] 为列表格式
```

## 模型训练

**训练前会将 images / 255 进行归一化处理，注意：targets 真实框的坐标不做归一化处理**

images 经过模型训练，因为做了归一化处理，预测的坐标值范围为 0-1，所以会经过`decode_boxes`解算处理，还原成真实的坐标位置

YOLOv1 模型输出为：

**完整输出形状**:

```
                输入images :(16, 512, 640, 640)  [B=16, C=512, H=640, W=640]
                                            ↓
                                    经过解耦检测头输出
                                            ↓
                         ┌───────────────────────────────────────────────────────┐
                         │                                                       │
                         ↓                                                       ↓
                类别分支 (16, 512, 20, 20)                             回归分支 (16, 512, 20, 20)
                         │                                                       │
                         ↓                                                       │
             ┌──────────────────────────┐                                        │
             ↓                          │                                        │
 经过预测层1通道1x1卷积处理      经过预测层num_classes通道1x1卷积处理        经过预测层4通道1x1卷积处理
             │                          │                                        │
             ↓                          ↓                                        ↓
          形状变换                    形状变换                                 形状变换
                        [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]              ↓
             │                          │                                 decode_boxes解算真实坐标
             ↓                          ↓                                        ↓
    置信度预测: (16, 400, 1)           类别预测:(16, 400, 20)               位置预测:(16, 400, 4)
```

```python
# 网络输出
outputs = {
    "pred_obj": obj_pred,  # (torch.Tensor) [B, M, 1] 置信度预测
    "pred_cls": cls_pred,  # (torch.Tensor) [B, M, C] 类别预测
    "pred_box": box_pred,  # (torch.Tensor) [B, M, 4] 位置参数预测
    "stride": self.stride,  # (Int)
    "fmp_size": fmp_size,  # (List[int, int])
}
```

## 损失计算

损失计算需要将网络输出和目标 targets 进行比较，计算置信度损失、类别损失和回归损失。具体步骤如下：

1. **有无物体置信度损失**：计算预测的置信度与真实目标存在与否的差异（暂时采用简单的二分类标签 0/1 作为置信度的学习标签。这样改进并不表示二分类标签比将 IoU 作为学习标签的方法更好,而仅仅是图方便）
2. **类别置信度损失**：计算预测的类别与真实类别的交叉熵损失。
3. **位置参数损失**：计算预测的位置与真实边界框的差异，通常使用 IOU（交并比）作为损失函数。

计算损失之前，需要将 targets 转换为网络输出相同的格式，这是通过 YoloMatcher 实现的

```
YoloMatcher的作用就是建立模型预测输出和目标标签之间的对应关系，假设网络输出为20 * 20 个网格，将targets中原图片的坐标映射到20*20对应的网格上

比如：
有无物体置信度：
网络输出预测结果：[B, grid_y*grid_x, 0.5（假设预测有无物体置信度值为0.5）]
标签结果转化为：[B, grid_y*grid_x, 1（真实标签的有无物体置信度值为1）]

类别置信度：
网络输出预测结果：[B, grid_y*grid_x, 20(0, 0.2, 0.3 , 0.4, 0.65, 0.23 ...)]
标签结果转化为：[B, grid_y*grid_x, 20(0, 1, 0 , 0, 0, 0 ...)]

统一格式，然后计算损失
```

# YOLOV1 验证流程

## 数据预处理

## 模型验证

每一个验证图片经过模型预测，然后通过 nms 处理输出为:

```
    bboxes: (numpy.array) -> [N, 4] 每个边界框的两个坐标 x1, y1, x2, y2
    score:  (numpy.array) -> [N,]   每个边界框的最高类别置信度
    labels: (numpy.array) -> [N,]   每个边界框的类别索引
```

然后将所有图片的预测结果保存在`det_results/eval/voc_eval/detections.pkl`文件中，保存的 python 对象格式为

```python
    all_boxes = [
        [  # 类别0: dog
            np.array([[10, 10, 50, 50, 0.9],    # 图像0检测到2个dog
                      [30, 30, 70, 70, 0.7]]),
            np.array([[20, 20, 60, 60, 0.8]])    # 图像1检测到1个dog
        ],
        [  # 类别1: cat
            np.array([[15, 15, 55, 55, 0.85]]),  # 图像0检测到1个cat
            np.array([])                          # 图像1没有检测到cat
        ],
        [  # 类别2: bird
            np.array([[40, 40, 80, 80, 0.6]]),   # 图像0检测到1个bird
            np.array([[25, 25, 65, 65, 0.75]])   # 图像1检测到1个bird
        ]
        ...
    ]
```

同时也将 all_boxes 转换为 VOC 评估格式并存入文件，后续计算 mAP 要用到

```
输出文件格式：每个类别一个文件，共 20 个文件（VOC有20个类别）。
VOC2007/results/
├── det_test_person.txt
├── det_test_aeroplane.txt
├── det_test_bicycle.txt
├── det_test_bird.txt
├── ...
└── det_test_tvmonitor.t
内容示例：
000005 0.987 10.0 15.0 100.0 150.0
000005 0.923 120.0 30.0 200.0 180.0
000005 0.856 50.0 60.0 150.0 200.0
000006 0.912 20.0 25.0 80.0 90.0
000007 0.945 15.0 20.0 120.0 130.0
```

## 计算 mAP

**先分别计算每个类别的 AP 值，然后再计算 mAP 值**

1. 先读取 VOC 数据集的所有真实框信息，`VOC2007/Annotations/%s.xml`将其解析为如下格式：

```
    recs保存的是验证集每个图片的所有真实框信息，格式为:
    {
        'img_001': [
                {'name': 'person', 'bbox': [10, 20, 100, 150], 'difficult': False},
                {'name': 'dog', 'bbox': [20, 25, 79, 89], 'difficult': False}, ...
                ],
        'img_002': [...], ...
    }
```

保存在`VOC2007/annotations_cache/annots.pkl`做缓存，避免每次都需要解析 xml 文件

> 假设计算`person`类

2. 将 recs -> class_recs，class_recs 保存的是验证集每个图片的属于 person 类的真实框信息，格式为

```
    class_recs保存的是验证集每个图片的属于person类的真实框信息，格式为:
    class_recs = {
        "img_001": {"bbox": [...], "difficult": [...], "det": [...]},
        "img_002": {"bbox": [...], "difficult": [...], "det": [...]},
        ...
    }
```

3. 读取类别对应的预测结果文件，例如`VOC2007/results/det_test_person.txt`，格式如下：

```
000005 0.987 10.0 15.0 100.0 150.0
000005 0.923 120.0 30.0 200.0 180.0
000005 0.856 50.0 60.0 150.0 200.0
000006 0.912 20.0 25.0 80.0 90.0
000007 0.945 15.0 20.0 120.0 130.0
```

4. 遍历该类别的所有预测框
    - 找到该预测框所属图片的所有真实框
    - 和所有真实框计算IOU，并找到IOU最大的真实框
    - 如果IoU > 阈值 且 未被检测过 且 非difficult，则该预测框为TP，否则为FP
    - 计算该类别的AP
  
5. 计算所有类别的mAP，即所有类别的AP的平均值