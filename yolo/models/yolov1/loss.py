import torch
import torch.nn.functional as F

from utils.box_ops import get_ious

from .matcher import YoloMatcher


# Criterion类用于完成训练阶段的<标签分配>和<损失计算>两个重要环节
class Criterion(object):
    def __init__(self, cfg, device, num_classes=80):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.loss_obj_weight = cfg["loss_obj_weight"]
        self.loss_cls_weight = cfg["loss_cls_weight"]
        self.loss_box_weight = cfg["loss_box_weight"]

        # 标签分配（制作正样本）
        self.matcher = YoloMatcher(num_classes=num_classes)

    def loss_objectness(self, pred_obj, gt_obj):
        """计算objectness损失"""
        loss_obj = F.binary_cross_entropy_with_logits(
            pred_obj, gt_obj, reduction="none"
        )

        return loss_obj

    def loss_classes(self, pred_cls, gt_label):
        """计算classification损失"""
        loss_cls = F.binary_cross_entropy_with_logits(
            pred_cls, gt_label, reduction="none"
        )

        return loss_cls

    def loss_bboxes(self, pred_box, gt_box):
        """计算bbox regression损失"""
        ious = get_ious(pred_box, gt_box, box_mode="xyxy", iou_type="giou")
        loss_box = 1.0 - ious

        return loss_box

    def __call__(self, outputs, targets, epoch=0):
        device = outputs["pred_cls"][0].device  # 设备
        stride = outputs["stride"]  # 网络的步长（每个网格所占的size）
        fmp_size = outputs["fmp_size"]  # 输出特征图的尺寸

        # List[B, M, C] -> [B, M, C] -> [BM, C]
        pred_obj = outputs["pred_obj"].view(-1)  # [BM,] 置信度预测
        pred_cls = outputs["pred_cls"].view(-1, self.num_classes)  # [BM, C] 类别预测
        pred_box = outputs["pred_box"].view(-1, 4)  # [BM, 4] 位置参数预测

        # ------------------ 标签分配 ------------------
        """
        M =grid_y*grid_x  grid_y, grid_x表示真实box中心点所在的网格坐标
        
        gt_objectness: [B, M, 1]  第四维为1则表示此处的网格有物体，对应一个正样本
        gt_classes: [B, M, num_classes] 第四维为one-hot格式 [0,1,0,0,0,0,0,0,0,0] 第二个分类概率为1
        gt_bboxes: [B, M, 4] 第四维为4个坐标参数   
        """
        gt_objectness, gt_classes, gt_bboxes = self.matcher(
            fmp_size=fmp_size, stride=stride, targets=targets
        )

        # 将标签的shape处理成和预测的shape相同的形式，以便后续计算损失
        gt_objectness = gt_objectness.view(-1).to(device).float()  # [BM,]
        gt_classes = gt_classes.view(-1, self.num_classes).to(device).float()  # [BM, C]
        gt_bboxes = gt_bboxes.view(-1, 4).to(device).float()  # [BM, 4]

        # pos_mask: 正样本标记
        """
        # 假设 gt_objectness = [0, 1, 0, 1, 1, 0, 0, 1, 0]
        pos_masks = [False, True, False, True, True, False, False, True, False]
        #           ↓      ↓            ↓     ↓              ↓
        #           负样本  正样本       正样本  正样本         正样本
        """
        pos_masks = gt_objectness > 0
        # num_fgs: 正样本数量
        num_fgs = pos_masks.sum()

        # 如果使用多卡做分布式训练，需要在多张卡上计算正样本数量的均值
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_fgs)
        # num_fgs = (num_fgs / get_world_size()).clamp(1.0)

        # ------------------ 计算损失 ------------------
        ## 计算objectness损失，即边界框的置信度、或有无物体的置信度的损失
        loss_obj = self.loss_objectness(pred_obj, gt_objectness)
        loss_obj = loss_obj.sum() / num_fgs

        ## 计算classification损失，即类别的置信度的损失
        """
        下面两行代码的意思是只取正样本进行类别损失计算
        因为负样本没有类别标签，无法计算类别损失
        
        损失	    正样本	    负样本	             原因
        objectness	✅计算	  ✅计算	        都需要学习预测有无物体
        class	    ✅计算	  ❌不计算	    负样本无类别标签
        box	        ✅计算	  ❌不计算	    负样本无边界框标签
        """
        pred_cls_pos = pred_cls[pos_masks]
        gt_classes_pos = gt_classes[pos_masks]
        loss_cls = self.loss_classes(pred_cls_pos, gt_classes_pos)
        loss_cls = loss_cls.sum() / num_fgs

        ## 计算box regression损失，即边界框回归的损失
        pred_box_pos = pred_box[pos_masks]
        gt_bboxes_pos = gt_bboxes[pos_masks]
        loss_box = self.loss_bboxes(pred_box_pos, gt_bboxes_pos)
        loss_box = loss_box.sum() / num_fgs

        ## 计算总的损失，即上面三个损失的加权和
        # Loss = λobj * Loss_obj + λcls * Loss_cls + λbox * Loss_box
        losses = (
            self.loss_obj_weight * loss_obj
            + self.loss_cls_weight * loss_cls
            + self.loss_box_weight * loss_box
        )

        ## 最后，将所有的loss保存在Dict中，以便后续的处理
        loss_dict = dict(
            loss_obj=loss_obj, loss_cls=loss_cls, loss_box=loss_box, losses=losses
        )

        return loss_dict


def build_criterion(cfg, device, num_classes):
    criterion = Criterion(cfg=cfg, device=device, num_classes=num_classes)

    return criterion


if __name__ == "__main__":
    pass
