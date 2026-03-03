import torch
import numpy as np


# YoloMatcher类用于完成训练阶段的<标签分配>
class YoloMatcher(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    @torch.no_grad()
    def __call__(self, fmp_size, stride, targets):
        """
        输入参数的解释:
            img_size: (Int) 输入图像的尺寸
            stride:   (Int) YOLOv1网络的输出步长
            targets:  (List[Dict]) 为List类型，包含一批数据的标签，每一个数据标签为Dict类型，其主要的数据结构为：
                             dict{'boxes':  (torch.Tensor) [N, 4], 一张图像中的N个目标边界框坐标
                                  'labels': (torch.Tensor) [N,], 一张图像中的N个目标类别标签
                                  ...}
            target 格式为：下面的坐标参数，都为经过了图像增强、变换处理后的坐标               
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
        """
        # 准备后续处理会用到的变量
        bs = len(targets)
        fmp_h, fmp_w = fmp_size
        gt_objectness = np.zeros([bs, fmp_h, fmp_w, 1]) 
        gt_classes = np.zeros([bs, fmp_h, fmp_w, self.num_classes]) 
        gt_bboxes = np.zeros([bs, fmp_h, fmp_w, 4])

        # 第一层for循环遍历每一张图像的标签
        for batch_index in range(bs):
            targets_per_image = targets[batch_index]
            # [N,]
            tgt_cls = targets_per_image["labels"].numpy()
            # [N, 4]
            tgt_box = targets_per_image['boxes'].numpy()

            # 第二层for循环遍历该张图像的每一个目标的标签
            for gt_box, gt_label in zip(tgt_box, tgt_cls):
                # 获得该目标的边界框坐标
                x1, y1, x2, y2 = gt_box

                # 计算目标框的中心点坐标和宽高
                xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
                bw, bh = x2 - x1, y2 - y1

                # 检查该目标边界框是否有效
                if bw < 1. or bh < 1.:
                    continue    

                # 计算中心点所在的网格坐标
                xs_c = xc / stride
                ys_c = yc / stride
                grid_x = int(xs_c)
                grid_y = int(ys_c)

                #  检查网格坐标是否有效
                if grid_x < fmp_w and grid_y < fmp_h:
                    # 标记objectness标签，即此处的网格有物体，对应一个正样本
                    # 暂时采用简单的二分类标签0/1作为置信度的学习标签。这样改进并不表示二分类标签比将IoU作为学习标签的方法更好,而仅仅是图方便
                    gt_objectness[batch_index, grid_y, grid_x] = 1.0

                    # 标记正样本处的类别标签，采用one-hot格式
                    cls_ont_hot = np.zeros(self.num_classes) # 负样本全为0
                    cls_ont_hot[int(gt_label)] = 1.0
                    gt_classes[batch_index, grid_y, grid_x] = cls_ont_hot

                    # 标记正样本处的bbox标签
                    gt_bboxes[batch_index, grid_y, grid_x] = np.array([x1, y1, x2, y2])

        """        
        假设输出特征图的尺寸为13x13，经过上面的处理 grid_y，grid_x表示box中心点所在的网格坐标

        gt_objectness: [B, grid_y, grid_x, 1]  第四维为1则表示此处的网格有物体，对应一个正样本
        gt_classes: [B, grid_y, grid_x, num_classes] 第四维为one-hot格式 [0,1,0,0,0,0,0,0,0,0] 第二个分类概率为1
        gt_bboxes: [B, grid_y, grid_x, 4] 第四维为4个坐标参数
        """
        # 将标签数据的shape从 [B, H, W, C] 的形式reshape成 [B, M, C] ，其中M = HW，以便后续的处理
        gt_objectness = gt_objectness.reshape(bs, -1, 1)
        gt_classes = gt_classes.reshape(bs, -1, self.num_classes)
        gt_bboxes = gt_bboxes.reshape(bs, -1, 4)

        """
        YoloMatcher的作用就是建立模型预测输出和目标标签之间的对应关系
        比如： 
        预测结果：[B, grid_y*grid_x, 0.5（假设预测置信度值为0.5）]
        标签结果：[B, grid_y*grid_x, 1]
        统一格式
        """
        # 将numpy.array类型转换为torch.Tensor类型
        gt_objectness = torch.from_numpy(gt_objectness).float()
        gt_classes = torch.from_numpy(gt_classes).float()
        gt_bboxes = torch.from_numpy(gt_bboxes).float()

        return gt_objectness, gt_classes, gt_bboxes
