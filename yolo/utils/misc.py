
import torch
import torch.nn as nn
import numpy as np
# ---------------------------- NMS ----------------------------
## basic NMS (基础非极大值抑制)
def nms(bboxes, scores, nms_thresh):
    """
    标准NMS算法：根据IoU阈值抑制重叠的边界框
    
    Args:
        bboxes: (np.ndarray) [N, 4] - 边界框坐标 [x1, y1, x2, y2]
        scores: (np.ndarray) [N] - 每个边界框的置信度分数
        nms_thresh: (float) - IoU阈值，超过此值则抑制
    
    Returns:
        keep: (list) [M] - 保留的边界框索引列表
    """
    # 提取所有边界框的坐标
    # shape: [N] - 每个数组包含所有框的对应坐标
    x1 = bboxes[:, 0]  # 左上角x坐标
    # 输入: x1.shape = (N,), 示例: [10, 25, 38, 42, ...]
    
    y1 = bboxes[:, 1]  # 左上角y坐标
    # 输入: y1.shape = (N,), 示例: [15, 30, 45, 50, ...]
    
    x2 = bboxes[:, 2]  # 右下角x坐标
    # 输入: x2.shape = (N,), 示例: [50, 80, 95, 100, ...]
    
    y2 = bboxes[:, 3]  # 右下角y坐标
    # 输入: y2.shape = (N,), 示例: [60, 100, 110, 120, ...]

    # 计算每个边界框的面积
    # shape: [N] - 每个框的面积
    areas = (x2 - x1) * (y2 - y1)
    # 输入: areas.shape = (N,), 示例: [2500, 2750, 2750, 2400, ...]

    # 按置信度分数降序排序，获取索引
    # shape: [N] - 从高到低的索引顺序
    order = scores.argsort()[::-1]
    # 输入: order.shape = (N,), 示例: [5, 2, 1, 3, 0, 4, ...] - 索引5分数最高

    # 初始化保留列表
    keep = []
    # 输入: keep = []

    # 循环处理，直到所有边界框都被处理
    while order.size > 0:
        # 每次取当前分数最高的边界框
        i = order[0]
        # 输入: i = 整数，当前最高分框的索引
        # 示例: i = 5
        
        # 将该框加入保留列表
        keep.append(i)
        # 输入: keep = [5]
        
        # 计算当前框与剩余所有框的IoU
        # shape: [N-1] - 与剩余框的坐标交并
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 交集左上角x
        # 输入: xx1.shape = (N-1,), 示例: [50, 38, 42, ...]
        
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 交集左上角y
        # 输入: yy1.shape = (N-1,), 示例: [60, 45, 50, ...]
        
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 交集右下角x
        # 输入: xx2.shape = (N-1,), 示例: [70, 80, 75, ...]
        
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 交集右下角y
        # 输入: yy2.shape = (N-1,), 示例: [80, 100, 95, ...]

        # 计算交集的宽度和高度，使用1e-10防止负数
        # shape: [N-1] - 交集的宽度和高度
        w = np.maximum(1e-10, xx2 - xx1)  # 交集宽度
        # 输入: w.shape = (N-1,), 示例: [20, 42, 33, ...]
        
        h = np.maximum(1e-10, yy2 - yy1)  # 交集高度
        # 输入: h.shape = (N-1,), 示例: [20, 55, 45, ...]
        
        inter = w * h  # 交集面积
        # 输入: inter.shape = (N-1,), 示例: [400, 2310, 1485, ...]

        # 计算IoU (Intersection over Union)
        # IoU = 交集 / (并集) = 交集 / (面积1 + 面积2 - 交集)
        # shape: [N-1] - 每个框与当前框的IoU
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        # 输入: iou.shape = (N-1,), 示例: [0.08, 0.42, 0.27, ...]
        # 说明: 添加1e-14防止除零
        
        # 保留IoU小于等于阈值的边界框
        # shape: [M] - 保留框的索引（相对于order[1:]）
        inds = np.where(iou <= nms_thresh)[0] # np.where() 返回一个元组，取第一个元素
        # 输入: inds.shape = (M,), 示例: [0, 3, 5, ...] - 索引0,3,5的IoU小于阈值
        # 说明: 这些框与当前框重叠度低，需要保留
        
        # 更新order，保留未被抑制的框
        # inds+1是因为order[1:]去掉了第一个元素，需要加1恢复到原索引
        # shape: [M] - 下一轮要处理的框的索引
        order = order[inds + 1]
        # 输入: order.shape = (M,), 示例: [2, 4, 6, ...] - 下一轮处理这些框

    # 返回所有保留的边界框索引
    # shape: [M] - 最终保留的框的索引
    return keep
    # 输出: keep = [5, 2, 1, ...] - 保留的框索引列表

## class-agnostic NMS 
def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh):
    # nms
    keep = nms(bboxes, scores, nms_thresh)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes

## class-aware NMS (类别感知的多类别NMS)
def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes):
    """
    类别感知的多类别NMS：对每个类别分别执行NMS，避免不同类别间的相互抑制
    
    Args:
        scores: (np.ndarray) [N] - 每个边界框的置信度分数
        labels: (np.ndarray) [N] - 每个边界框的类别标签
        bboxes: (np.ndarray) [N, 4] - 每个边界框的坐标 [x1, y1, x2, y2]
        nms_thresh: (float) - NMS的IoU阈值
        num_classes: (int) - 类别总数
    
    Returns:
        scores: (np.ndarray) [M] - NMS后保留的边界框置信度
        labels: (np.ndarray) [M] - NMS后保留的边界框类别标签
        bboxes: (np.ndarray) [M, 4] - NMS后保留的边界框坐标
    """
    # 初始化一个全0数组，用于标记哪些边界框被保留
    # shape: [N], dtype=int32, 初始值全为0
    keep = np.zeros(len(bboxes), dtype=np.int32)
    # 输入: keep.shape = (N,), 示例: (100,)
    
    # 遍历每个类别，对每个类别单独执行NMS
    for i in range(num_classes):
        # 找到当前类别i的所有边界框的索引
        inds = np.where(labels == i)[0]
        # 输入: inds.shape = (K,), K为当前类别的边界框数量
        # 示例: 假设类别0有15个框, inds = [0, 3, 7, 12, ...]
        
        # 如果当前类别没有边界框，跳过
        if len(inds) == 0:
            # 输入: inds.shape = (0,)
            continue
        
        # 提取当前类别的所有边界框
        c_bboxes = bboxes[inds]
        # 输入: c_bboxes.shape = (K, 4)
        # 示例: (15, 4) - 15个边界框，每个框4个坐标值
        
        # 提取当前类别的所有置信度分数
        c_scores = scores[inds]
        # 输入: c_scores.shape = (K,)
        # 示例: (15,) - 15个边界框的置信度
        
        # 对当前类别的边界框执行NMS
        # 返回: c_keep是保留的边界框在c_bboxes/c_scores中的索引
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        # 输入: c_keep.shape = (M,), M为NMS后保留的边界框数量 (M <= K)
        # 示例: (8,) - 从15个框中保留了8个
        
        # 在全局keep数组中标记保留的边界框
        # inds[c_keep]将类别内部的索引转换为全局索引
        keep[inds[c_keep]] = 1
        # 输入: keep.shape = (N,), 当前类别对应的索引位置被设置为1
        # 示例: keep = [0,1,0,1,1,0,...] - 1表示保留，0表示去除
    
    # 找到所有标记为1的索引（即所有被保留的边界框）
    keep = np.where(keep > 0)
    # 输入: keep是一个元组 (array([idx1, idx2, ...]),)
    # 输出: keep[0].shape = (M,), M为所有类别总共保留的边界框数量
    # 示例: (42,) - 总共保留了42个边界框
    
    # 根据keep索引提取保留的边界框信息
    scores = scores[keep]
    # 输入: scores.shape = (M,)
    # 示例: (42,) - 42个边界框的置信度
    
    labels = labels[keep]
    # 输入: labels.shape = (M,)
    # 示例: (42,) - 42个边界框的类别标签
    
    bboxes = bboxes[keep]
    # 输入: bboxes.shape = (M, 4)
    # 示例: (42, 4) - 42个边界框的坐标

    return scores, labels, bboxes

## multi-class NMS 
def multiclass_nms(scores, labels, bboxes, nms_thresh, num_classes, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes)