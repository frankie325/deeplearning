import os
import cv2
import random
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image, ImageDraw, ImageFont

# img_path = "D:\my code\yolo_data\VOC2012\JPEGImages"
# annotations_path = "D:\my code\yolo_data\VOC2012\Annotations"
# img_path = "/Users/frank/code/ai/yolo_data/VOC2012/JPEGImages"
# annotations_path = "/Users/frank/code/ai/yolo_data/VOC2012/Annotations"
# img_path = "D:\my code\yolo_data\DetData\\train\images"
# annotations_path = "D:\my code\yolo_data\DetData\\train\Annotations"
# img_path = "/Users/frank/code/ai/yolo_data/DetData/train/images"
# annotations_path = "/Users/frank/code/ai/yolo_data/DetData/train/Annotations"
img_path = "D:\my code\yolo_data\VOC2007\JPEGImages"
annotations_path = "D:\my code\yolo_data\VOC2007\Annotations"

ALL_CLASS = [
    "person",
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
    "bottle",
    "chair",
    "diningtable",
    "pottedplant",
    "sofa",
    "tvmonitor",
]

# 20个分类的颜色
COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 128, 0),
    (255, 0, 128),
    (128, 255, 0),
    (0, 255, 128),
    (255, 128, 128),
    (128, 255, 128),
    (128, 128, 255),
    (192, 192, 192),
]

# ALL_CLASS = ["猫羽雫", "喵喵~", "女孩"]

# # 3个分类的颜色
# COLORS = [
#     (0, 255, 0),
#     (255, 0, 0),
#     (0, 0, 255),
# ]
S = 7
IMAGE_SIZE = 448
GRID_CELL_SIZE = IMAGE_SIZE / S


def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    font_path = "/System/Library/Fonts/STHeiti Light.ttc"
    try:
        font = ImageFont.truetype(font_path, text_size, encoding="utf-8")
    except OSError:
        font = ImageFont.load_default()

    # 绘制文本
    draw.text(position, text, fill=text_color[::-1], font=font)

    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def random_hsv(img):
    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 分离H、S、V三个通道
    h, s, v = cv2.split(hsv)

    # 随机调整饱和度(S)和亮度(V)
    # 变化范围为0.66到1.5倍
    scale_s = random.uniform(0.66, 1.5)
    # 随机生成亮度V的缩放比例
    scale_v = random.uniform(0.66, 1.5)

    # 将S通道转换为浮点型并应用缩放系数
    s = s.astype(np.float32) * scale_s
    # 将V通道转换为浮点型并应用缩放系数
    v = v.astype(np.float32) * scale_v

    # 将S通道的值截断在0-255之间，并转换回uint8类型
    s = np.clip(s, 0, 255).astype(np.uint8)
    # 将V通道的值截断在0-255之间，并转换回uint8类型
    v = np.clip(v, 0, 255).astype(np.uint8)

    # 合并H、S、V三个通道
    hsv = cv2.merge((h, s, v))
    # 将图像从HSV颜色空间转换回BGR颜色空间
    img_aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # 返回增强后的图像
    return img_aug


def augment_data(img, boxes):
    # 1. 进行HSV颜色增强
    img = random_hsv(img)

    # 2. 缩放和填充(保持纵横比，居中)
    # 获取图像的高和宽
    h, w = img.shape[:2]
    # 目标尺寸为448x448
    target_size = IMAGE_SIZE

    # 计算缩放比例，取宽和高中缩放比例较小的那个，以保证图像能完整放入目标尺寸
    scale = min(target_size / w, target_size / h)

    # 计算缩放后的新宽度
    new_w = int(w * scale)
    # 计算缩放后的新高度
    new_h = int(h * scale)

    # 创建一个填充灰色的画布(128)，尺寸为448x448
    canvas = np.full((target_size, target_size, 3), 128, dtype=np.uint8)

    # 将原图像进行缩放
    resized_img = cv2.resize(img, (new_w, new_h))

    # 计算水平方向的偏移量，使图像居中
    dx = (target_size - new_w) // 2
    # 计算垂直方向的偏移量，使图像居中
    dy = (target_size - new_h) // 2

    # 将缩放后的图像粘贴到画布的中心位置
    canvas[dy : dy + new_h, dx : dx + new_w] = resized_img

    # 更新图像变量为画布
    img = canvas

    # 用于存储变换后的边界框
    new_boxes = []
    # 遍历所有的边界框
    for box in boxes:
        # 解包边界框信息：左上角x, 左上角y, 右下角x, 右下角y, 类别索引, 类别名称
        xmin, ymin, xmax, ymax, idx = box

        # 对边界框坐标进行相应的缩放和平移
        # 左上角x坐标缩放并加上水平偏移
        xmin = int(xmin * scale + dx)
        # 左上角y坐标缩放并加上垂直偏移
        ymin = int(ymin * scale + dy)
        # 右下角x坐标缩放并加上水平偏移
        xmax = int(xmax * scale + dx)
        # 右下角y坐标缩放并加上垂直偏移
        ymax = int(ymax * scale + dy)

        # 将变换后的边界框添加到列表中
        new_boxes.append([xmin, ymin, xmax, ymax, idx])

    # 返回处理后的图像和边界框
    return img, new_boxes


# 只做图像的resize
def resize_data(img, boxes):
    # 使用 cv2.imread 读取的彩色图像，其形状（shape）是一个包含三个数值的元组： (Height, Width, Channels) ，即 (高度, 宽度, 通道数) 。
    # 获取图像的高和宽
    h, w = img.shape[:2]
    # 目标尺寸为448x448
    target_size = IMAGE_SIZE

    # 直接resize图像到448x448
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # 更新标注框的坐标，保持与图像的比例关系
    for box in boxes:
        xmin, ymin, xmax, ymax, idx = box
        # 对坐标进行缩放，保持比例关系
        xmin = int(xmin / w * target_size)
        ymin = int(ymin / h * target_size)
        xmax = int(xmax / w * target_size)
        ymax = int(ymax / h * target_size)
        # 更新边界框坐标
        box[:5] = [xmin, ymin, xmax, ymax, idx]

    return img, boxes


def loadData():
    images = os.listdir(img_path)
    for img_name in images:
        # 读取图像
        img = cv2.imread(os.path.join(img_path, img_name))
        name, suffix = img_name.split(".")
        # 读取图像对应的xml，从中取出标签
        annotation_name = name + ".xml"
        annotation_path = os.path.join(annotations_path, annotation_name)
        tree = ET.parse(annotation_path)

        all_objects = []
        objects = tree.findall("object")  # 从xml中取出所有object标签
        for object in objects:
            class_name = object.find("name").text
            if class_name not in ALL_CLASS:
                continue
            bbox = object.find("bndbox")
            xmin = int(bbox.findtext("xmin"))
            ymin = int(bbox.findtext("ymin"))
            xmax = int(bbox.findtext("xmax"))
            ymax = int(bbox.findtext("ymax"))
            class_idx = ALL_CLASS.index(class_name)
            all_objects.append([xmin, ymin, xmax, ymax, class_idx])

        # 应用YOLOv1数据增强
        img, all_objects = augment_data(img, all_objects)
        # img, all_objects = resize_data(img, all_objects)

        for obj in all_objects:
            xmin, ymin, xmax, ymax, class_idx = obj
            print(xmin, ymin, xmax, ymax, class_idx)
            color = COLORS[class_idx]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                img,
                ALL_CLASS[class_idx],
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,  # 字体大小调大
                color,
                2,
            )
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 保存处理后的图像
        break


def NMS(boxes):
    iou_threshold = 0.5

    # boxes: (7, 7, 30)
    # 每个grid cell预测两个框，先过滤掉置信度小的那个框
    # boxes = boxes[boxes[:, :, 4] > confidence_threshold]  # 过滤掉置信度小于0.5的框
    # boxes = boxes[boxes[:, :, 5:].max(2)[0] > confidence_threshold]  # 过滤掉类别置信度小于0.5的框
    # boxes = boxes
    predict_boxes = []
    for grid_i in range(S):
        for grid_j in range(S):
            # 取出当前grid cell的两个框
            box1 = boxes[grid_i, grid_j, :5]
            box2 = boxes[grid_i, grid_j, 5:10]
            classes = boxes[grid_i, grid_j, 10:]
            class_score = torch.max(classes)  # 概率最大的值
            class_idx = torch.argmax(classes)  # 概率最大的类别索引
            print(class_idx)

            # !计算置信度：物体分类准确度 × 含有物体的置信度
            box1_conf = box1[4] * class_score  # box1置信度
            box2_conf = box2[4] * class_score  # box2置信度
            # 保留置信度较高的框，过滤一半，还有49个框
            if box1_conf > box2_conf:
                predict_boxes.append(
                    torch.cat(
                        [box1[:4], box1_conf.unsqueeze(0), class_idx.unsqueeze(0)]
                    )
                )
            else:
                predict_boxes.append(
                    torch.cat(
                        [box2[:4], box2_conf.unsqueeze(0), class_idx.unsqueeze(0)]
                    )
                )

    # 按照置信度从大到小排序
    predict_boxes = sorted(predict_boxes, key=lambda x: x[4], reverse=True)
    print(len(predict_boxes))
    print(predict_boxes)


# 计算两个矩形框的交并比（IOU）
# https://zhuanlan.zhihu.com/p/436801032
def IOU(box1, box2, grid_i, grid_j):
    # grid_i, grid_j 是当前grid cell的索引
    # box1, box2 是两个边界框的坐标，格式为(x, y, w, h)
    box1_x, box1_y, box1_w, box1_h = box1
    box2_x, box2_y, box2_w, box2_h = box2
    w_grid = GRID_CELL_SIZE
    h_grid = GRID_CELL_SIZE

    # 复原出边界框的坐标
    # 还原中心点坐标 - 还原宽度的一半
    x1_min = (
        box1_x + grid_i
    ) * w_grid - box1_w * IMAGE_SIZE / 2  # 边界框左上角的 x 坐标
    y1_min = (
        box1_y + grid_j
    ) * h_grid - box1_h * IMAGE_SIZE / 2  # 边界框左上角的 y 坐标
    x1_max = (
        box1_x + grid_i
    ) * w_grid + box1_w * IMAGE_SIZE / 2  # 边界框右下角的 x 坐标
    y1_max = (
        box1_y + grid_j
    ) * h_grid + box1_h * IMAGE_SIZE / 2  # 边界框右下角的 y 坐标
    # print(x1_min, y1_min, x1_max, y1_max)
    x2_min = (
        box2_x + grid_i
    ) * w_grid - box2_w * IMAGE_SIZE / 2  # 边界框左上角的 x 坐标
    y2_min = (
        box2_y + grid_j
    ) * h_grid - box2_h * IMAGE_SIZE / 2  # 边界框左上角的 y 坐标
    x2_max = (
        box2_x + grid_i
    ) * w_grid + box2_w * IMAGE_SIZE / 2  # 边界框右下角的 x 坐标
    y2_max = (
        box2_y + grid_j
    ) * h_grid + box2_h * IMAGE_SIZE / 2  # 边界框右下角的 y 坐标

    # 计算两个矩形面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # 计算交集面积
    x_inter_min = torch.max(x1_min, x2_min)  # 交集的左上角x坐标
    y_inter_min = torch.max(y1_min, y2_min)  # 交集的左上角y坐标
    x_inter_max = torch.min(x1_max, x2_max)  # 交集的右下角x坐标
    y_inter_max = torch.min(y1_max, y2_max)  # 交集的右下角y坐标

    inter_area = torch.max(
        torch.tensor(0.0, device=x1_min.device), x_inter_max - x_inter_min
    ) * torch.max(torch.tensor(0.0, device=x1_min.device), y_inter_max - y_inter_min)

    # 计算并集面积：两个矩形框的面积之和减去交集面积
    union_area = area1 + area2 - inter_area

    # 计算IOU
    if union_area > 0:
        iou = inter_area / union_area
    else:
        iou = (
            torch.tensor(0.0, device=inter_area.device)
            if isinstance(inter_area, torch.Tensor)
            else 0.0
        )

    return iou


class VOCDataset(Dataset):
    def __init__(self, img_path, annotations_path, train=True):
        self.img_path = img_path
        self.annotations_path = annotations_path
        self.size = IMAGE_SIZE  # 图像转换为448的尺寸
        list = os.listdir(img_path)
        # 按照文件名排序，确保数据是有序的，每次划分数据都是一样的
        list = sorted(list)
        # 划分训练集和测试集
        if train:
            self.images = list[: int(len(list) * 0.8)]
        else:
            self.images = list[int(len(list) * 0.8) :]

    def __getitem__(self, index):
        img_name = self.images[index]
        name, suffix = img_name.split(".")
        img = cv2.imread(os.path.join(self.img_path, img_name))
        annotation_name = name + ".xml"
        annotation_path = os.path.join(self.annotations_path, annotation_name)
        tree = ET.parse(annotation_path)

        all_boxes = []
        objects = tree.findall(
            "object"
        )  # 从xml中取出所有object标签，object标签含有标签名和标注框的坐标
        for object in objects:
            class_name = object.find("name").text
            if class_name not in ALL_CLASS:
                continue
            bbox = object.find("bndbox")
            xmin = int(float(bbox.findtext("xmin")))  # 边界框左上角的 x 坐标
            ymin = int(float(bbox.findtext("ymin")))  # 边界框左上角的 y 坐标
            xmax = int(float(bbox.findtext("xmax")))  # 边界框右下角的 x 坐标
            ymax = int(float(bbox.findtext("ymax")))  # 边界框右下角的 y 坐标
            class_idx = ALL_CLASS.index(class_name)
            all_boxes.append([xmin, ymin, xmax, ymax, class_idx])
        # 应用YOLOv1数据增强
        img, boxes = augment_data(img, all_boxes)
        # img, boxes = resize_data(img, all_boxes)

        # 获取一个网格的宽高
        w_grid = GRID_CELL_SIZE
        h_grid = GRID_CELL_SIZE

        # 真实标签，7*7*(10+num_classes)，每个grid cell有10+num_classes个元素，前10个元素为中心坐标和宽高，后num_classes个元素为类别概率
        label = np.zeros((S, S, 10 + len(ALL_CLASS)))
        for box in boxes:
            xmin, ymin, xmax, ymax, class_idx = box
            # 计算真实中心坐标
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # !计算真实框的中心坐标落在哪个grid cell中
            # 这里也可以看出来一个grid cell只有一个真实框，无法处理多个真实框的情况
            x_idx = int(x_center / w_grid)
            y_idx = int(y_center / h_grid)

            # !转换成相对位置，相对于grid cell的左上角（即归一化到[0, 1]）
            x = (x_center / w_grid) - x_idx
            y = (y_center / h_grid) - y_idx
            #  !宽高也归一化到[0, 1]
            w = (xmax - xmin) / self.size
            h = (ymax - ymin) / self.size

            # !前10个元素为中心坐标和宽高，两个box都设置为该真实框
            label[x_idx, y_idx, :10] = [x, y, w, h, 1, x, y, w, h, 1]
            # !后20个元素为类别概率，只有一个为1，其他为0
            label[x_idx, y_idx, 10 + class_idx] = 1

            # print(xmin, ymin, xmax, ymax) # 打印验证IOU是否正确还原坐标
            # IOU([x, y, w, h], [0.5, 0.3, 0.6, 0.8], x_idx, y_idx)
            # color = COLORS[class_idx]
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            # cv2.putText(
            #     img,
            #     ALL_CLASS[class_idx],
            #     (xmin, ymin - 5),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     color,
            #     2,
            # )

        # 将图片像素转换为Tensor，归一化到[0, 1]
        tensor_img = torch.tensor(img, dtype=torch.float32) / 255.0  # (448, 448, 3)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        tensor_img = tensor_img.permute(2, 0, 1)  # 转换为 (3, 448, 448)
        return tensor_img, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    # print(int(3 / 2))
    # loadData()
    dataset = VOCDataset(img_path, annotations_path)
    print("图片数量", len(dataset))
    img, label = dataset[3999]
    print(img.shape)  # (3, 448, 448)
    print(img)
    print(label.shape)  # (7, 7, 30)
    torch.set_printoptions(threshold=float("inf"))  # 确保 Tensor 完整打印，不省略
    print(label)
