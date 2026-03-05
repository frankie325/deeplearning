import torch
import torch.nn as nn

from .yolov1_backbone import build_backbone
from .yolov1_neck import build_neck
from .yolov1_head import build_head
import numpy as np
from utils.misc import multiclass_nms


class YOLOv1(nn.Module):
    def __init__(
        self,
        cfg,
        device,
        img_size=None,
        num_classes=20,
        conf_thresh=0.01,
        nms_thresh=0.5,
        trainable=False,
        deploy=False,
    ):
        super(YOLOv1, self).__init__()
        # ------------------- 基础参数 -------------------
        self.cfg = cfg  # 模型配置文件
        self.img_size = img_size  # 输入图像大小
        self.device = device  # cuda或者是cpu
        self.num_classes = num_classes  # 类别的数量
        self.trainable = trainable  # 训练的标记
        self.conf_thresh = conf_thresh  # 得分阈值
        self.nms_thresh = nms_thresh  # NMS阈值
        self.stride = 32  # 网络的最大步长（每个网格所占的size）
        self.deploy = deploy

        # ------------------- 网络结构 -------------------
        ## 主干网络
        self.backbone, feat_dim = build_backbone(
            model_name=cfg["backbone"], pretrained=trainable & cfg["pretrained"]
        )

        ## 颈部网络
        self.neck = build_neck(cfg, in_dim=feat_dim, out_dim=512)
        head_dim = self.neck.out_dim  # 512

        ## 检测头
        self.head = build_head(cfg, head_dim, head_dim, num_classes)

        ## 预测层：采用当下主流的做法,即使用 1x1 的卷积层在特征图上做预测
        self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1)

    @torch.no_grad()
    def inference(self, x):
        # 主干网络
        feat = self.backbone(x)

        # 颈部网络
        feat = self.neck(feat)

        # 检测头
        cls_feat, reg_feat = self.head(feat)

        # 预测层
        obj_pred = self.obj_pred(cls_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        fmp_size = obj_pred.shape[-2:]

        # 对 pred 的 size 做一些 view 调整，便于后续的处理
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 测试时，笔者默认 batch 是1，
        # 因此，我们不需要用batch这个维度，用[0]将其取走。
        obj_pred = obj_pred[0]  # [H*W, 1]
        cls_pred = cls_pred[0]  # [H*W, NC]
        reg_pred = reg_pred[0]  # [H*W, 4]

        # 每个边界框的得分
        # [H*W, num_classes] 20个类别对应的置信度分数
        scores = torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid())

        # 解算边界框, 并归一化边界框: [H*W, 4]
        bboxes = self.decode_boxes(reg_pred, fmp_size)

        if self.deploy:
            # 这段代码和ONNX部署有关，读者不必关注这段if的代码
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)

            return outputs
        else:
            # 将 bbox 和 score 预测都放在 cpu 处理上，以便进行后处理
            scores = scores.cpu().numpy()
            bboxes = bboxes.cpu().numpy()

            # 后处理
            bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels

    def create_grid(self, fmp_size):
        """
        用于生成G矩阵，其中每个元素都是特征图上的像素坐标。
        假设 hs=3, ws=4：

        torch.arange(hs) → [0, 1, 2]
        torch.arange(ws) → [0, 1, 2, 3]

        grid_y 每行相同（表示行号），grid_x 每列相同（表示列号）。
        grid_y:              grid_x:
        [[0, 0, 0, 0],       [[0, 1, 2, 3],
         [1, 1, 1, 1],        [0, 1, 2, 3],
         [2, 2, 2, 2]]        [0, 1, 2, 3]]

        后续通过 torch.stack([grid_x, grid_y], dim=-1) 组合成 [H, W, 2] # 形状: [3, 4, 2]

        结果:
        [
          [[0, 0], [1, 0], [2, 0], [3, 0]],   # 第0行：所有网格的坐标
          [[0, 1], [1, 1], [2, 1], [3, 1]],   # 第1行
          [[0, 2], [1, 2], [2, 2], [3, 2]]    # 第2行
        ]

        展平后的结果：
        [
          [0, 0], [1, 0], [2, 0], [3, 0],  # 第0行展开
          [0, 1], [1, 1], [2, 1], [3, 1],  # 第1行展开
          [0, 2], [1, 2], [2, 2], [3, 2]   # 第2行展开
        ]
        """
        # 特征图的宽和高
        ws, hs = fmp_size

        # 生成网格的x坐标和y坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])

        # 将xy两部分的坐标拼起来：[H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

        # [H, W, 2] -> [HW, 2] -> [HW, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)

        return grid_xy

    def decode_boxes(self, pred_reg, fmp_size):
        """
        将YOLO预测的 (tx, ty)、(tw, th) 转换为bbox的左上角坐标 (x1, y1) 和右下角坐标 (x2, y2)。
        输入:
            pred_reg: (torch.Tensor) -> [B, HxW, 4] or [HxW, 4]，网络预测的txtytwth
            fmp_size: (List[int, int])，包含输出特征图的宽度和高度两个参数
        输出:
            pred_box: (torch.Tensor) -> [B, HxW, 4] or [HxW, 4]，解算出的边界框坐标

        假设h=13, w=13
        """
        # 生成网格坐标矩阵
        grid_cell = self.create_grid(fmp_size)

        # 计算预测边界框的中心点坐标和宽高
        # 取最后1维的前两个数，也就是预测中心坐标（tx, ty）
        # pred_ctr: [B, 169, 2] grid_cell: [169, 2] 会自动广播到 [B, 169, 2] 与 pred_reg  相加
        pred_ctr = (torch.sigmoid(input=pred_reg[..., :2]) + grid_cell) * self.stride
        # 取最后1维的后两个数，也就是预测中心坐标（tw, th）
        pred_wh = torch.exp(pred_reg[..., 2:]) * self.stride

        # 将所有bbox的中心带你坐标和宽高换算成x1y1x2y2形式
        pred_x1y1 = pred_ctr - pred_wh * 0.5  # 左上角坐标
        pred_x2y2 = pred_ctr + pred_wh * 0.5  # 右下角坐标
        pred_box = torch.cat(
            [pred_x1y1, pred_x2y2], dim=-1
        )  # 最后一维拼接成 [B, 169, 4]

        return pred_box

    def postprocess(self, bboxes, scores):
        """
        后处理环节，包括<阈值筛选>和<非极大值抑制(NMS)>两个环节
        输入:
            bboxes: (numpy.array) -> [HxW, 4]
            scores: (numpy.array) -> [HxW, num_classes]
        输出:
            bboxes: (numpy.array) -> [N, 4]
            score:  (numpy.array) -> [N,]
            labels: (numpy.array) -> [N,]
        """
        # 将得分最高的类别作为预测的类别标签
        labels = np.argmax(scores, axis=1)  # 找出置信度最高的类别索引 （13*13，）

        # 使用高级索引获取每个网格最高类别的置信度值
        """
        np.arange(3)        # [0, 1, 2]
        labels              # [1, 2, 0]
        索引展开：组合成坐标对
        scores = [scores[0, 1], scores[1, 2], scores[2, 0]]

        scores存储的是网格的最高类别的置信度值
        """
        scores = scores[(np.arange(scores.shape[0]), labels)]  # （169，）

        # 阈值筛选
        # keep = (array([3, 7, 12, 45, ...]),)  元组，包含满足条件的索引
        keep = np.where(scores >= self.conf_thresh)

        # 应用阈值筛选
        """
        元组索引取值进行过滤 示例：
        # = arr[(array([0, 3, 5]),)]
        # = arr[[0, 3, 5]]
        # = [arr[0], arr[3], arr[5]]
        # = [10, 40, 60]
        """
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False
        )

        """
        经过nms后筛选出的输出结果形状
        bboxes: (numpy.array) -> [N, 4] 每个边界框的两个坐标 x1, y1, x2, y2 
        score:  (numpy.array) -> [N,]   每个边界框的最高类别置信度
        labels: (numpy.array) -> [N,]   每个边界框的类别索引
        """
        return bboxes, scores, labels

    def forward(self, x):
        if not self.trainable:
        # 验证阶段进行推理
            return self.inference(x)
        else:
            # 主干网络
            feat = self.backbone(x)

            # 颈部网络
            feat = self.neck(feat)

            # 检测头
            cls_feat, reg_feat = self.head(feat)

            # 预测层
            obj_pred = self.obj_pred(cls_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            fmp_size = obj_pred.shape[
                -2:
            ]  # -2:表示取最后两个形状，也就是输出特征图的尺寸

            # 对 pred 的 size 做一些 view 调整，便于后续的解算预测框处理
            # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            # 解算边界框坐标
            box_pred = self.decode_boxes(reg_pred, fmp_size)

            # 网络输出
            outputs = {
                "pred_obj": obj_pred,  # (torch.Tensor) [B, M, 1] 有无物体置信度预测
                "pred_cls": cls_pred,  # (torch.Tensor) [B, M, C] 类别置信度预测
                "pred_box": box_pred,  # (torch.Tensor) [B, M, 4] 位置参数预测
                "stride": self.stride,  # (Int)
                "fmp_size": fmp_size,  # (List[int, int])
            }

            # print(f"YOLOV1置信度预测特征图形状：{obj_pred.shape}")
            # print(f"YOLOV1类别预测特征图形状：{cls_pred.shape}")
            # print(f"YOLOV1位置参数预测测特征图形状：{reg_pred.shape}")

            return outputs
