from dataset.voc import VOCDetection, VOC_CLASSES
import os
import time
import numpy as np
import pickle
import xml.etree.ElementTree as ET

from utils.box_ops import rescale_bboxes
import wandb


class VOCAPIEvaluator:
    """VOC AP Evaluation class"""

    def __init__(
        self,
        data_dir,
        device,
        transform,
        set_type="test",
        year="2007",
        display=False,
    ):
        self.data_dir = data_dir
        self.device = device
        self.labelmap = VOC_CLASSES
        self.set_type = set_type
        self.year = year
        self.display = display
        self.map = 0.0

        # transform
        self.transform = transform

        # path 验证集数据文件路径
        self.devkit_path = os.path.join(data_dir, "VOC" + year)
        self.annopath = os.path.join(data_dir, "VOC2007", "Annotations", "%s.xml")
        self.imgpath = os.path.join(data_dir, "VOC2007", "JPEGImages", "%s.jpg")
        self.imgsetpath = os.path.join(
            data_dir, "VOC2007", "ImageSets", "Main", set_type + ".txt"
        )
        self.output_dir = self.get_output_dir(
            "./yolo/det_results/eval/voc_eval/", self.set_type
        )

        # dataset
        self.dataset = VOCDetection(
            data_dir=data_dir, image_sets=[("2007", set_type)], is_train=False
        )

    def evaluate(self, net):
        net.eval()
        num_images = len(self.dataset)
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        self.all_boxes = [
            [[] for _ in range(num_images)] for _ in range(len(self.labelmap))
        ]

        # timers
        det_file = os.path.join(self.output_dir, "detections.pkl")
        for i in range(num_images):
            img, _ = self.dataset.pull_image(i)
            orig_h, orig_w = img.shape[:2]

            # preprocess 图片预处理
            x, _, deltas = self.transform(img)
            # 添加 batch 维度	(3, H, W) → (1, 3, H, W)，并归一化
            x = x.unsqueeze(0).to(self.device) / 255.0

            # forward 前向传播
            """
            bboxes: (numpy.array) -> [N, 4] 每个边界框的两个坐标 x1, y1, x2, y2 
            score:  (numpy.array) -> [N,]   每个边界框的最高类别置信度
            labels: (numpy.array) -> [N,]   每个边界框的类别索引
            """
            t0 = time.time()
            bboxes, scores, labels = net(x)
            detect_time = time.time() - t0

            # rescale bboxes 将图像还原成真实的大小
            origin_img_size = [orig_h, orig_w]
            cur_img_size = [*x.shape[-2:]]
            bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)

            """
            将模型的检测结果按类别分组存储，为后续 mAP 计算做准备

            # labelmap = ['dog', 'cat', 'bird']  # 假设有3个类别
            # === j = 0 (狗) ===
            inds = np.where(labels == 0)[0]  # [0, 2]
            c_bboxes = bboxes[[0, 2]]  # [[10,10,50,50], [30,30,70,70]]
            c_scores = scores[[0, 2]]  # [0.9, 0.7]
            c_dets = np.hstack((c_bboxes, [[0.9], [0.7]]))
            # c_dets = [[10, 10, 50, 50, 0.9],
            #           [30, 30, 70, 70, 0.7]]
            self.all_boxes[0][i] = c_dets  # 图片i的"狗"检测结果

            # === j = 1 (猫) ===
            inds = np.where(labels == 1)[0]  # [1]
            c_dets = [[20, 20, 60, 60, 0.8]]
            self.all_boxes[1][i] = c_dets  # 图片i的"猫"检测结果

            # === j = 2 (鸟) ===
            inds = np.where(labels == 2)[0]  # [3]
            c_dets = [[40, 40, 80, 80, 0.6]]
            self.all_boxes[2][i] = c_dets  # 图片i的"鸟"检测结果

            最终输出all_boxes:
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
            ]
            """
            for j in range(len(self.labelmap)):
                inds = np.where(labels == j)[0]  # 每个边界框的类别索引与分类j相同的索引
                if len(inds) == 0:
                    self.all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = bboxes[inds]
                c_scores = scores[inds]
                # hstack表示水平拼接
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False
                )
                self.all_boxes[j][i] = c_dets

            if i % 500 == 0:
                print(
                    "im_detect: {:d}/{:d} {:.3f}s".format(
                        i + 1, num_images, detect_time
                    )
                )

        # 用于将 Python 对象保存到文件
        with open(det_file, "wb") as f:
            pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)

        # print("Evaluating detections")
        self.evaluate_detections(self.all_boxes)
        # print("Mean AP: ", self.map)

    def parse_rec(self, filename):
        """Parse a PASCAL VOC xml file"""
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall("object"):
            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            obj_struct["pose"] = obj.find("pose").text
            obj_struct["truncated"] = int(obj.find("truncated").text)
            obj_struct["difficult"] = int(obj.find("difficult").text)
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [
                int(bbox.find("xmin").text),
                int(bbox.find("ymin").text),
                int(bbox.find("xmax").text),
                int(bbox.find("ymax").text),
            ]
            objects.append(obj_struct)

        return objects

    def get_output_dir(self, name, phase):
        """Return the directory where experimental artifacts are placed.
        If the directory does not exist, it is created.
        A canonical path is built using the name from an imdb and a network
        (if not None).
        """
        filedir = os.path.join(name, phase)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        return filedir

    def get_voc_results_file_template(self, cls):
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = "det_" + self.set_type + "_%s.txt" % (cls)
        filedir = os.path.join(self.devkit_path, "results")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def do_python_eval(self, use_07=True):
        cachedir = os.path.join(self.devkit_path, "annotations_cache")
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        for i, cls in enumerate(self.labelmap):
            filename = self.get_voc_results_file_template(cls)
            rec, prec, ap = self.voc_eval(
                detpath=filename,
                classname=cls,
                cachedir=cachedir,
                ovthresh=0.5,
                use_07_metric=use_07_metric,
            )
            aps += [ap]
            # print('AP for {} = {:.4f}'.format(cls, ap))
            wandb.log({f"AP for {cls}": ap})
            with open(os.path.join(self.output_dir, cls + "_pr.pkl"), "wb") as f:
                pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if self.display:
            self.map = np.mean(aps)
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("--------------------------------------------------------------")
        else:
            self.map = np.mean(aps)
            # print('Mean AP = {:.4f}'.format(np.mean(aps)))
            wandb.log({"Mean AP =": np.mean(aps)})

    def voc_ap(self, rec, prec, use_07_metric=True):
        """
        计算 AP (Average Precision) - Precision-Recall曲线下面积

        输入:
            rec: 召回率数组，np.ndarray 形状 (N,)
            prec: 精确率数组，np.ndarray 形状 (N,)
            use_07_metric: 是否使用VOC07的11点插值方法，默认True

        输出:
            ap: float 标量，平均精度

        【计算方法说明】
        ==================== 方法1: VOC07 11点插值 ====================
        在11个等间距的Recall点 {0.0, 0.1, 0.2, ..., 1.0} 上，
        取该点右侧所有Precision的最大值，然后求平均

        【具体例子】
        假设 prec = [1.0, 1.0, 0.667, 0.5, 0.5]
             rec  = [0.5, 1.0, 1.0,  1.0, 1.0]

        t=0.0: 找 rec >= 0.0 的所有点 → 全部5个点
               取这些点的最大Precision → max([1.0, 1.0, 0.667, 0.5, 0.5]) = 1.0

        t=0.1: 找 rec >= 0.1 的所有点 → 全部5个点
               max(P) = 1.0

        t=0.2~0.4: 同上，max(P) = 1.0

        t=0.5: 找 rec >= 0.5 的所有点 → 全部5个点
               max(P) = 1.0

        t=0.6~1.0: 找 rec >= 0.6 的所有点 → 后4个点 (rec[1:]=[1.0,1.0,1.0,1.0])
               max(P) = max([1.0, 0.667, 0.5, 0.5]) = 1.0

        AP = (1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0) / 11 = 1.0

        注意：VOC07插值取的是右侧最大值（>=），不是左侧最大值（>）

        ==================== 方法2: 精确AP计算（Area Under Curve）====================
        直接计算P-R曲线下的积分面积

        【数据形状变换】
        输入: rec=[0.5, 1.0, 1.0, 1.0, 1.0], 形状 (5,)
              prec=[1.0, 1.0, 0.667, 0.5, 0.5], 形状 (5,)

        步骤1: 添加哨兵值
        mrec = [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0], 形状 (7,)
        mpre = [0.0, 1.0, 1.0, 0.667, 0.5, 0.5, 0.0], 形状 (7,)
        作用: 确保曲线从(0,0)开始，到(1,0)结束

        步骤2: 计算包络线（从右向左）
        从右往左遍历，每个位置取自己和右侧的最大值
        mpre[6]=0.0 → 不变
        mpre[5]=max(0.5, 0.0)=0.5
        mpre[4]=max(0.5, 0.5)=0.5
        mpre[3]=max(0.667, 0.5)=0.667
        mpre[2]=max(1.0, 0.667)=1.0
        mpre[1]=max(1.0, 1.0)=1.0
        mpre[0]=max(0.0, 1.0)=1.0
        结果: mpre = [1.0, 1.0, 1.0, 0.667, 0.5, 0.5, 0.0]
        效果: 包络线保证Precision单调递减

        步骤3: 找Recall变化的点
        mrec[1:] = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
        mrec[:-1] = [0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        比较: [True, True, False, False, False, False]
        i = [0, 1] (Recall变化的索引位置)

        步骤4: 计算梯形面积
        i = [0, 1]
        Δrecall   = [0.5, 0.5]
        mpre[i+1] = [1.0, 1.0]
        面积 = 0.5×1.0 + 0.5×1.0 = 1.0

        几何意义: 用梯形法近似P-R曲线下的面积
        AP = Σ (Δrecall × precision)
        """
        if use_07_metric:
            # ==================== VOC07 11点插值方法 ====================
            ap = 0.0
            # 遍历11个等间距的Recall点: 0.0, 0.1, 0.2, ..., 1.0
            for t in np.arange(0.0, 1.1, 0.1):
                # 【数据形状】 t: float 标量，当前Recall阈值

                # 找出所有Recall >= 当前阈值t的索引
                # 【数据形状】 rec >= t 返回布尔数组，形状 (N,)
                # 【数据形状】 np.sum(rec >= t) 返回满足条件的元素个数（int）

                if np.sum(rec >= t) == 0:
                    # 如果没有满足条件的点，Precision = 0
                    p = 0
                else:
                    # 从满足条件的点中，取最大Precision
                    # 【数据形状】 prec[rec >= t] 返回满足条件的Precision子数组
                    # 【数据形状】 np.max(prec[rec >= t]) 返回最大Precision（float）

                    # 注意：这是取所有 rec >= t 的点的最大Precision
                    #      即取该阈值右侧（包括当前点）的最大值
                    p = np.max(prec[rec >= t])

                # 累加到ap，除以11取平均
                # 【数据形状】 ap: float 累计值
                ap = ap + p / 11.0

        else:
            # ==================== 精确AP计算（Area Under Curve）====================
            # correct AP calculation
            # first append sentinel values at the end
            # 添加哨兵值，确保曲线从(0,0)开始，到(1,0)结束
            mrec = np.concatenate(([0.0], rec, [1.0]))
            # 【数据形状】 mrec = [0.0] + rec + [1.0]，形状 (N+2,)
            # 例: rec=[0.5,1.0,1.0] → mrec=[0.0, 0.5, 1.0, 1.0, 1.0]

            mpre = np.concatenate(([0.0], prec, [0.0]))
            # 【数据形状】 mpre = [0.0] + prec + [0.0]，形状 (N+2,)
            # 例: prec=[1.0,0.667,0.5] → mpre=[0.0, 1.0, 0.667, 0.5, 0.0]

            # compute the precision envelope
            # 计算Precision包络线：从右向左遍历，取自己和右侧的最大值
            # 目的：保证Precision单调递减，避免计算面积时出现负值
            for i in range(mpre.size - 1, 0, -1):
                # 【数据形状】 i: int，从 mpre.size-1 倒序到 1
                # i = mpre.size-1, mpre.size-2, ..., 2, 1

                # 取当前位置i-1和i的较大值，赋给i-1
                # 【数据形状】 mpre[i-1] 和 mpre[i] 都是float标量
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
                # 效果：从右往左，每个位置的值都变成自己右侧的最大值

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            # 找出Recall值发生变化的点（用于分段计算面积）
            i = np.where(mrec[1:] != mrec[:-1])[0]
            # 【数据形状】
            #   mrec[1:]: 从第2个元素开始的数组，形状 (N+1,)
            #   mrec[:-1]: 到倒数第2个元素的数组，形状 (N+1,)
            #   mrec[1:] != mrec[:-1]: 比较相邻元素，返回布尔数组，形状 (N+1,)
            #   np.where(...)[0]: 满足条件的索引数组，形状 (K,)

            # and sum (\Delta recall) * prec
            # 计算每个Recall变化段的梯形面积并求和
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            # 【数据形状分解】
            #   mrec[i + 1] - mrec[i]: np.ndarray，形状 (K,)，每段的Δrecall
            #   mpre[i + 1]: np.ndarray，形状 (K,)，每段的Precision值（高度）
            #   乘积: np.ndarray，形状 (K,)，每段的梯形面积
            #   np.sum(...): float，所有段面积之和 = AP

        return ap

    def voc_eval(self, detpath, classname, cachedir, ovthresh=0.5, use_07_metric=True):
        """
        计算单个类别的AP (Average Precision) 和 mAP

        【具体例子：评估 "person" 类别】

        假设测试集有3张图片，评估 "person" 类别：

        真实标注（Ground Truth）:
        - img_001: person1 (10,20,100,150), difficult=False
        - img_002: person2 (50,60,150,200), difficult=False
        - img_003: person3 (30,40,120,160), difficult=True  ← 困难样本，不计入npos

        模型预测（已按置信度降序）:
        1. img_001 (12,22,98,148) conf=0.95 → IoU=0.89, TP
        2. img_002 (55,65,145,195) conf=0.90 → IoU=0.84, TP
        3. img_001 (200,200,300,300) conf=0.85 → IoU=0.00, FP
        4. img_002 (10,10,50,50) conf=0.70 → IoU=0.02, FP
        5. img_003 (35,45,115,155) conf=0.65 → IoU=0.85, 但difficult，忽略

        关键参数:
            detpath: 预测结果文件路径模板，如 'VOCdevkit/VOC2007/results/det_test_%s.txt'
            classname: 要评估的类别名称，如 'aeroplane'
            cachedir: 缓存目录，用于存储解析后的标注
            ovthresh: IoU阈值，默认0.5，用于判断预测框是否正确
            use_07_metric: 是否使用VOC07的AP计算方法（11点插值），默认True

        返回:
            rec: 召回率 (Recall) 数组，形状 (nd,)
            prec: 精确率 (Precision) 数组，形状 (nd,)
            ap: 平均精度 (Average Precision) 标量值

        数据形状变换全过程:
            imagenames: List[str] 长度=图像数 (例: ['img_001', 'img_002', 'img_003'])
            class_recs: Dict[str, Dict] 结构 → class_recs['img_001']={'bbox': (N,4), 'difficult': (N,), 'det': (N,)}
            npos: int 标量，该类别的真实目标总数（不计difficult）
            splitlines: List[List[str]] 长度=预测框数
            confidence: np.ndarray 形状 (M,) M=总预测框数
            BB: np.ndarray 形状 (M, 4) 所有预测框坐标
            BB (排序后): np.ndarray 形状 (M, 4) 按置信度降序
            tp: np.ndarray 初始形状 (M,) → cumsum后形状 (M,)
            fp: np.ndarray 初始形状 (M,) → cumsum后形状 (M,)
            rec: np.ndarray 形状 (M,) 召回率数组
            prec: np.ndarray 形状 (M,) 精确率数组
            ap: float 标量
            mAP: float 标量 = 所有类别AP的平均值
        
        关键点总结
            TP（True Positive）：IoU > 阈值 且 未被检测过 且 非difficult
            FP（False Positive）：IoU ≤ 阈值 或 重复检测
            Recall = 累计TP / npos（npos=非difficult真实框数）
            Precision = 累计TP / (累计TP + 累计FP)
            AP = P-R曲线下面积（VOC07用11点插值）
            mAP = 所有类别AP的平均值
        """

        # ==================== 第1步: 创建缓存目录 ====================
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
            # 创建缓存目录用于存储解析后的XML标注文件
            # 示例: 'VOCdevkit/VOC2007/annotations_cache/'

        # 缓存文件路径
        cachefile = os.path.join(cachedir, "annots.pkl")
        # 示例: 'VOCdevkit/VOC2007/annotations_cache/annots.pkl'
        # 用途: 避免每次评估都重新解析XML文件，提升速度

        # ==================== 第2步: 读取图像列表 ====================
        with open(self.imgsetpath, "r") as f:
            lines = f.readlines()
        # self.imgsetpath = 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'
        # lines = ['img_001\n', 'img_002\n', 'img_003\n', ...]

        imagenames = [x.strip() for x in lines]
        # 【数据形状】 List[str]，长度=图像数
        # imagenames = ['img_001', 'img_002', 'img_003']
        # 本例: len(imagenames) = 3

        # ==================== 第3步: 加载或解析标注 ====================
        if not os.path.isfile(cachedir):
            # 缓存文件不存在，需要解析XML文件
            # load annots
            recs = {}  # 存储所有图像的标注信息
            # 【数据形状】 Dict[str, List[Dict]]
            # 结构: recs = {'img_001': [obj1, obj2, ...], 'img_002': [...], ...}

            for i, imagename in enumerate(imagenames):
                # 解析单个图像的XML标注文件
                recs[imagename] = self.parse_rec(self.annopath % (imagename))
                # self.annopath = 'VOCdevkit/VOC2007/Annotations/%s.xml'
                # imagename = 'img_001'
                # 实际路径: 'VOCdevkit/VOC2007/Annotations/img_001.xml'

                # parse_rec() 返回的格式:
                # [
                #   {'name': 'person', 'bbox': [10, 20, 100, 150], 'difficult': False},
                #   {'name': 'dog', 'bbox': [20, 25, 79, 89], 'difficult': False},
                #   ...
                # ]

                """
                recs保存的是验证集每个图片的真实框信息，格式为:
                {
                    'img_001': [
                            {'name': 'person', 'bbox': [10, 20, 100, 150], 'difficult': False}, 
                            {'name': 'dog', 'bbox': [20, 25, 79, 89], 'difficult': False}, ...
                            ], 
                    'img_002': [...], ...
                }
                """

                # 打印进度
                if i % 100 == 0 and self.display:
                    print(
                        "Reading annotation for {:d}/{:d}".format(
                            i + 1, len(imagenames)
                        )
                    )
                    # 输出: "Reading annotation for 100/4952"

            # save: 保存解析结果到缓存文件
            if self.display:
                print("Saving cached annotations to {:s}".format(cachefile))
                # 输出: "Saving cached annotations to VOCdevkit/VOC2007/annotations_cache/annots.pkl"

            with open(cachefile, "wb") as f:
                pickle.dump(recs, f)
                # 将recs字典序列化为pkl文件
                # 下次评估时直接加载，无需重新解析XML
        else:
            # load: 缓存文件存在，直接加载
            with open(cachefile, "rb") as f:
                recs = pickle.load(f)
                # 从缓存文件加载所有图像的标注信息

        # ==================== 第4步: 提取指定类别的真实目标 ====================
        class_recs = {}  # 存储指定类别的标注
        # 【数据形状】 Dict[str, Dict[str, np.ndarray]]
        # 结构: class_recs = {'img_001': {'bbox': (N,4), 'difficult': (N,), 'det': (N,)}, ...}
        # N = 该图像中该类别的目标数量

        npos = 0  # 该类别的真实目标总数（不包括difficult目标）
        # 【数据形状】 int 标量

        for imagename in imagenames:
            # 筛选当前图像中属于指定类别的目标
            R = [obj for obj in recs[imagename] if obj["name"] == classname]
            # classname = 'person'
            # 本例 R = [
            #   {'name': 'person', 'bbox': [10, 20, 100, 150], 'difficult': False},  ← img_001
            # ]
            # (注: img_003的person是difficult，也会被筛选进来，但后续不计入npos)

            # 提取边界框坐标
            bbox = np.array([x["bbox"] for x in R])
            # 【数据形状】 np.ndarray，形状 (N, 4)
            # img_001: bbox.shape = (1, 4) → [[10, 20, 100, 150]]

            # 提取difficult标志
            difficult = np.array([x["difficult"] for x in R]).astype(bool)
            # 【数据形状】 np.ndarray，形状 (N,)
            # img_001: difficult = [False]
            # img_002: difficult = [False]
            # img_003: difficult = [True] ← 困难样本

            # 初始化检测标志（标记该目标是否已被正确预测）
            det = [False] * len(R)
            # 【数据形状】 List[bool]，长度N
            # det = [False]

            # 统计非difficult的目标数量（就是TP + FN的数量）
            npos = npos + sum(~difficult)
            # 【数据形状】 int 标量（累加）
            # img_001: npos = 0 + sum(~[False]) = 0 + 1 = 1
            # img_002: npos = 1 + sum(~[False]) = 1 + 1 = 2
            # img_003: npos = 2 + sum(~[True]) = 2 + 0 = 2 ← difficult不计入
            # 最终: npos = 2

            # 存储该图像该类别的标注信息
            """
                class_recs保存的是验证集每个图片的属于person类的真实框信息，格式为:
                class_recs = {
                    "img_001": {"bbox": [...], "difficult": [...], "det": [...]},
                    "img_002": {"bbox": [...], "difficult": [...], "det": [...]},
                    ...
                }
            """

            class_recs[imagename] = {
                "bbox": bbox,           # 边界框坐标，形状 (N, 4)
                "difficult": difficult, # 是否为困难样本，形状 (N,)
                "det": det,             # 是否已被检测，长度N
            }

        # 本例: npos = 2 (测试集中person类的非difficult真实目标总数)

        # ==================== 第5步: 读取模型预测结果 ====================
        detfile = detpath.format(classname)
        # detpath = 'VOCdevkit/VOC2007/results/det_test_%s.txt'
        # classname = 'person'
        # detfile = 'VOCdevkit/VOC2007/results/det_test_person.txt'

        """读取的是所有person类的预测框"""
        with open(detfile, "r") as f:
            lines = f.readlines()
        # 文件内容示例（本例）:
        # img_001 0.95 12.0 22.0 98.0 148.0
        # img_002 0.90 55.0 65.0 145.0 195.0
        # img_001 0.85 200.0 200.0 300.0 300.0
        # img_002 0.70 10.0 10.0 50.0 50.0
        # img_003 0.65 35.0 45.0 115.0 155.0

        # 检查是否有预测结果
        if any(lines) == 1:
            # 解析每一行预测结果
            splitlines = [x.strip().split(" ") for x in lines]
            # 【数据形状】 List[List[str]]，长度=预测框数
            # 本例: splitlines = [
            #   ['img_001', '0.95', '12.0', '22.0', '98.0', '148.0'],
            #   ['img_002', '0.90', '55.0', '65.0', '145.0', '195.0'],
            #   ['img_001', '0.85', '200.0', '200.0', '300.0', '300.0'],
            #   ['img_002', '0.70', '10.0', '10.0', '50.0', '50.0'],
            #   ['img_003', '0.65', '35.0', '45.0', '115.0', '155.0'],
            # ]
            # len(splitlines) = 5

            # 提取图像ID
            image_ids = [x[0] for x in splitlines]
            # 【数据形状】 List[str]，长度=预测框数
            # image_ids = ['img_001', 'img_002', 'img_001', 'img_002', 'img_003']

            # 提取置信度分数
            confidence = np.array([float(x[1]) for x in splitlines])
            # 【数据形状】 np.ndarray，形状 (M,)
            # confidence = [0.95, 0.90, 0.85, 0.70, 0.65]
            # M = 5 (总预测框数)

            # 提取边界框坐标
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
            # 【数据形状】 np.ndarray，形状 (M, 4)
            # BB = [[12.0, 22.0, 98.0, 148.0],
            #       [55.0, 65.0, 145.0, 195.0],
            #       [200.0, 200.0, 300.0, 300.0],
            #       [10.0, 10.0, 50.0, 50.0],
            #       [35.0, 45.0, 115.0, 155.0]]

            # ==================== 第6步: 按置信度降序排序 ====================
            sorted_ind = np.argsort(-confidence)
            # argsort(-confidence) 降序排序
            # 【数据形状】 np.ndarray，形状 (M,)
            # sorted_ind = [0, 1, 2, 3, 4] (已按置信度降序排列)

            sorted_scores = np.sort(-confidence)
            # sorted_scores = [-0.95, -0.90, -0.85, -0.70, -0.65]

            # 应用排序
            BB = BB[sorted_ind, :]
            # 【数据形状】 np.ndarray，形状 (M, 4) 按置信度降序排列
            # 本例BB已按降序，不变

            image_ids = [image_ids[x] for x in sorted_ind]
            # 【数据形状】 List[str]，长度M，按置信度降序排列
            # image_ids = ['img_001', 'img_002', 'img_001', 'img_002', 'img_003']

            # ==================== 第7步: 计算TP和FP ====================
            nd = len(image_ids)
            # 【数据形状】 int 标量，总预测框数
            # nd = 5

            # 初始化TP和FP数组
            tp = np.zeros(nd)  # True Positives
            fp = np.zeros(nd)  # False Positives
            # 【数据形状】 np.ndarray，形状 (nd,)
            # tp = [0., 0., 0., 0., 0.]
            # fp = [0., 0., 0., 0., 0.]

            # 遍历每个预测框
            for d in range(nd):
                # 获取该预测框对应图像的真实标注
                """
                class_recs保存的是验证集每个图片的属于person类的真实框信息，格式为:
                class_recs = {
                    "img_001": {"bbox": [...], "difficult": [...], "det": [...]},
                    "img_002": {"bbox": [...], "difficult": [...], "det": [...]},
                    ...
                }
                """
                R = class_recs[image_ids[d]]
                # 【数据形状】 Dict[str, np.ndarray]
                # d=0: image_ids[0]='img_001'
                #   R = {'bbox': np.array([[10, 20, 100, 150]]),
                #        'difficult': np.array([False]),
                #        'det': [False]}

                # 当前预测框
                bb = BB[d, :].astype(float)
                # 【数据形状】 np.ndarray，形状 (4,)
                # d=0: bb = [12.0, 22.0, 98.0, 148.0]

                # 最大IoU
                ovmax = -np.inf
                # 【数据形状】 float 标量，初始化为负无穷
                # ovmax = -inf

                # 该图像的真实边界框
                BBGT = R["bbox"].astype(float)
                # 【数据形状】 np.ndarray，形状 (N, 4) N=该图像中该类别的真实目标数
                # d=0: BBGT.shape = (1, 4)
                # BBGT = [[10, 20, 100, 150]]

                # 如果该图像有该类别的真实目标
                if BBGT.size > 0:
                    # compute overlaps: 计算IoU
                    # intersection: 计算交集
                    ixmin = np.maximum(BBGT[:, 0], bb[0])  # 交集左上角x
                    # 【数据形状】 np.ndarray，形状 (N,)
                    # d=0: ixmin = max([10], 12) = [12]

                    iymin = np.maximum(BBGT[:, 1], bb[1])  # 交集左上角y
                    # 【数据形状】 np.ndarray，形状 (N,)
                    # d=0: iymin = max([20], 22) = [22]

                    ixmax = np.minimum(BBGT[:, 2], bb[2])  # 交集右下角x
                    # 【数据形状】 np.ndarray，形状 (N,)
                    # d=0: ixmax = min([100], 98) = [98]

                    iymax = np.minimum(BBGT[:, 3], bb[3])  # 交集右下角y
                    # 【数据形状】 np.ndarray，形状 (N,)
                    # d=0: iymax = min([150], 148) = [148]

                    iw = np.maximum(ixmax - ixmin, 0.0)     # 交集宽度
                    # 【数据形状】 np.ndarray，形状 (N,)
                    # d=0: iw = max([98-12], 0) = [86]

                    ih = np.maximum(iymax - iymin, 0.0)     # 交集高度
                    # 【数据形状】 np.ndarray，形状 (N,)
                    # d=0: ih = max([148-22], 0) = [126]

                    inters = iw * ih                         # 交集面积
                    # 【数据形状】 np.ndarray，形状 (N,)
                    # d=0: inters = [86 * 126] = [10836]

                    # union: 计算并集
                    uni = (
                        (bb[2] - bb[0]) * (bb[3] - bb[1])           # 预测框面积
                        # 【数据形状】 float 标量
                        # d=0: (98-12) * (148-22) = 86 * 126 = 10836
                        + (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1])  # 真实框面积
                        # 【数据形状】 np.ndarray，形状 (N,)
                        # d=0: (100-10) * (150-20) = 90 * 130 = [11700]
                        - inters
                        # 【数据形状】 np.ndarray，形状 (N,)
                    )  # 并集面积 = 预测框 + 真实框 - 交集
                    # d=0: uni = [10836 + 11700 - 10836] = [11700]

                    overlaps = inters / uni  # IoU = 交集 / 并集
                    # 【数据形状】 np.ndarray，形状 (N,)
                    # d=0: overlaps = [10836 / 11700] = [0.926]

                    # 找到IoU最大的真实目标
                    ovmax = np.max(overlaps)  # 最大IoU
                    # 【数据形状】 float 标量
                    # d=0: ovmax = 0.926

                    jmax = np.argmax(overlaps) # 最大IoU的真实目标框的索引
                    # 【数据形状】 int 标量
                    # d=0: jmax = 0

                # ==================== 第8步: 判断TP或FP ====================
                # 【本例具体计算结果】
                # d=0: img_001预测框(12,22,98,148) vs 真实框(10,20,100,150)
                #      IoU=0.926 > 0.5, difficult=False, det=False → tp[0]=1.0
                # d=1: img_002预测框(55,65,145,195) vs 真实框(50,60,150,200)
                #      IoU=0.835 > 0.5, difficult=False, det=False → tp[1]=1.0
                # d=2: img_001预测框(200,200,300,300) vs 真实框(10,20,100,150)
                #      IoU=0.00 < 0.5 → fp[2]=1.0
                # d=3: img_002预测框(10,10,50,50) vs 真实框(50,60,150,200)
                #      IoU=0.02 < 0.5 → fp[3]=1.0
                # d=4: img_003预测框(35,45,115,155) vs 真实框(30,40,120,160)
                #      IoU=0.85 > 0.5, 但difficult=True → 忽略，tp[4]=0, fp[4]=0
                

                """
                 TP（True Positive）：IoU > 阈值 且 未被检测过 且 非difficult
                 FP（False Positive）：IoU ≤ 阈值 或 重复检测
                """
                if ovmax > ovthresh:  # IoU > 阈值
                    # 与真实目标有足够重叠
                    if not R["difficult"][jmax]:  # 不是difficult样本
                        if not R["det"][jmax]:     # 该真实目标未被检测过
                            tp[d] = 1.0           # True Positive
                            # 【数据形状】 tp[d] 是float标量
                            R["det"][jmax] = 1    # 标记为已检测
                        else:
                            fp[d] = 1.0           # False Positive (重复检测)
                    else:
                        # difficult样本，既不算TP也不算FP（tp[d]=0, fp[d]=0）
                        pass
                else:
                    fp[d] = 1.0  # False Positive (IoU不够)

            # 本例最终结果（d=0~4）:
            # tp = [1.0, 1.0, 0.0, 0.0, 0.0]
            # fp = [0.0, 0.0, 1.0, 1.0, 0.0]

            # compute precision recall
             # 累计求和函数，返回数组元素的累计和：第一次计算一个预测框，第二次计算总共两个预测框，依此类推
            fp = np.cumsum(fp)
            # 【数据形状】 np.ndarray，形状 (nd,)
            # cumsum后: fp = [0., 0., 1., 2., 2.]

            tp = np.cumsum(tp)
            # 【数据形状】 np.ndarray，形状 (nd,)
            # cumsum后: tp = [1., 2., 2., 2., 2.]

            """
                召回率定义: Recall = TP / (TP + FN)
                         = TP / 所有真实框总数
                
                在VOC评估中:
                - TP: 正确检测的框数
                - FN: 未检测到的真实框数
                - npos: 非difficult真实框总数
                
                关键关系: npos = TP + FN
                因为: 
                - 每个非difficult真实框要么被正确检测(TP)，要么没被检测(FN)
                - 没有其他可能性
                
                所以: TP / (TP + FN) = TP / npos
            """
            rec = tp / float(npos)
            # 【数据形状】 np.ndarray，形状 (nd,)
            # rec = [1./2, 2./2, 2./2, 2./2, 2./2] = [0.5, 1.0, 1.0, 1.0, 1.0]
            # Recall = 累计TP / npos（npos=2）

            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            # 【数据形状】 np.ndarray，形状 (nd,)
            # tp+fp = [1., 2., 3., 4., 4.]
            # prec = [1./1, 2./2, 2./3, 2./4, 2./4] = [1.0, 1.0, 0.667, 0.5, 0.5]
            # Precision = 累计TP / (累计TP + 累计FP)

            ap = self.voc_ap(rec, prec, use_07_metric)
            # 【数据形状】 float 标量
            # AP = Precision-Recall曲线下面积
            # VOC07方法：在11个Recall点 {0,0.1,...,1.0} 上取最大Precision，然后求平均
            # 本例P-R曲线：
            #   Recall=0.5时，Precision=1.0
            #   Recall=1.0时，Precision=0.5（从0.5降到0.667再到0.5）
            # VOC07插值：每个Recall点取右侧最大Precision
            #   R=0.0~0.5: max(P) = 1.0
            #   R=0.6~1.0: max(P) = 0.667
            # AP = (6×1.0 + 5×0.667) / 11 = (6 + 3.335) / 11 ≈ 0.85
        else:
            rec = -1.0
            prec = -1.0
            ap = -1.0

        return rec, prec, ap

    # 将模型预测结果写入 VOC 格式的文本文件，用于后续使用 VOC 官方评估工具计算 mAP
    def write_voc_results_file(self, all_boxes):
        """
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
        ]
        """
        # 遍历每个类别
        for cls_ind, cls in enumerate(self.labelmap):
            # cls_ind: 类别索引 (0, 1, 2, ...)
            # cls: 类别名称 ('aeroplane', 'bicycle', ...)

            if self.display:
                print("Writing {:s} VOC results file".format(cls))
                # 输出: "Writing aeroplane VOC results file"

            # 生成输出文件路径
            # filename = "det_test_aeroplane.txt"
            # path = "VOCdevkit/VOC2007/results/det_test_aeroplane.txt"
            filename = self.get_voc_results_file_template(cls)

            # 打开文件并写入检测结果
            with open(filename, "wt") as f:
                # 遍历每张图像
                for im_ind, index in enumerate(self.dataset.ids):
                    # im_ind: 图像索引
                    # index: 图像ID元组 (路径, 图像名)
                    # 示例: ('D:/data/VOC2007', '000005')

                    # 获取当前类别在当前图像上的检测结果
                    dets = all_boxes[cls_ind][im_ind]
                    # dets.shape = (N, 5)
                    # 示例: dets = [[10, 10, 50, 50, 0.9],
                    #                 [30, 30, 70, 70, 0.7]]

                    # 跳过空检测结果
                    if 0 in dets.shape:
                        continue

                    # 遍历该图像的所有检测框
                    for k in range(dets.shape[0]):
                        # 写入检测结果到文件
                        """
                        为什么需要 +1？

                        PyTorch/NumPy 使用 0-based 索引（从0开始）
                        VOC 评估工具使用 1-based 索引（从1开始）
                        """
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index[1],  # 图像ID，如 '000005'
                                dets[k, -1],  # 置信度分数，如 0.900
                                dets[k, 0] + 1,  # x1 坐标 + 1 (VOC使用1-based索引)
                                dets[k, 1] + 1,  # y1 坐标 + 1
                                dets[k, 2] + 1,  # x2 坐标 + 1
                                dets[k, 3] + 1,  # y2 坐标 + 1
                            )
                        )
            """
            输出文件格式：每个类别一个文件，共 20 个文件（VOC有20个类别）。
            VOCdevkit/VOC2007/results/
            ├── det_test_aeroplane.txt
            ├── det_test_bicycle.txt
            ├── det_test_bird.txt
            ├── ...
            └── det_test_tvmonitor.txt

            内容示例：
            000005 0.987 10.0 15.0 100.0 150.0
            000005 0.923 120.0 30.0 200.0 180.0
            000005 0.856 50.0 60.0 150.0 200.0
            000006 0.912 20.0 25.0 80.0 90.0
            000007 0.945 15.0 20.0 120.0 130.0
            """

    def evaluate_detections(self, box_list):
        self.write_voc_results_file(box_list)
        self.do_python_eval()
