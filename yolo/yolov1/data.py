import os
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import Dataset

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

img_path = "D:\my code\yolo_data\VOC2012\JPEGImages"
annotations_path = "D:\my code\yolo_data\VOC2012\Annotations"

def loadData():

    images = os.listdir(img_path)
    for img_name in images:
        # 读取图像
        img = cv2.imread(os.path.join(img_path, img_name))
        # print(img.shape)  # (500, 486, 3)
        name, suffix = img_name.split(".")
        # 读取图像对应的xml，从中取出标签
        annotation_name = name + suffix
        annotation_path = os.path.join(annotations_path, annotation_name)
        tree = ET.parse(annotation_path)
        width = int(tree.find('size').findtext('width'))  # 获取图片的宽度
        height = int(tree.find('size').findtext('height'))  # 获取图片的高度
        objects = tree.findall("object")  #从xml中取出所有object标签，object标签含有标签名和标注框的坐标
        # 获取一个网格的宽高
        w_grid = width / 7
        h_grid = height / 7
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
            print(xmin, ymin, xmax, ymax, class_idx)
        # 按照YOLOv1论文，将图像resize到448×448，并同步缩放平移标注框
        # target_size = 448
        # h, w = img.shape[:2]
        # scale = min(target_size / w, target_size / h)
        # new_w, new_h = int(w * scale), int(h * scale)

        # # 缩放图像
        # resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # # 创建448×448画布，将缩放后的图像居中粘贴
        # canvas = 128 * np.ones((target_size, target_size, 3), dtype=np.uint8)
        # dw, dh = (target_size - new_w) // 2, (target_size - new_h) // 2
        # canvas[dh:dh + new_h, dw:dw + new_w] = resized
        # img = canvas

        # # 同步缩放平移标注框
        # xmin = int(xmin * scale + dw)
        # ymin = int(ymin * scale + dh)
        # xmax = int(xmax * scale + dw)
        # ymax = int(ymax * scale + dh)

        # 直接resize到448×448，不保持原比例
        target_size = 448
        h, w = img.shape[:2]

        # 直接resize图像
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # 同步线性映射标注框坐标
        xmin = int(xmin * target_size / w)
        ymin = int(ymin * target_size / h)
        xmax = int(xmax * target_size / w)
        ymax = int(ymax * target_size / h)

        # 展示图像并将标注框添加到图像中
        color = COLORS[class_idx]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, class_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 保存处理后的图像
        cv2.imwrite("D:\my code\deeplearning\yolo\yolov1\\" + img_name, img)
        break


# loadData()


img_path = "D:\my code\yolo_data\VOC2012\JPEGImages"
annotations_path = "D:\my code\yolo_data\VOC2012\Annotations"


class VOCDataset(Dataset):
    def __init__(self,  transform=None):
        self.transform = transform
        self.images = os.listdir(img_path)

    def __getitem__(self, index):
        img_name = self.images[index]
        name, suffix = img_name.split(".")
        img = cv2.imread(os.path.join(img_path, img_name))
        annotation_name = name + suffix
        annotation_path = os.path.join(annotations_path, annotation_name)
        tree = ET.parse(annotation_path)
        objects = tree.findall("object")  #从xml中取出所有object标签，object标签含有标签名和标注框的坐标
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

            # 直接resize到448×448，不保持原比例
            target_size = 448
            h, w = img.shape[:2]

            # 直接resize图像
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

            # 同步线性映射标注框坐标
            xmin = int(xmin * target_size / w)
            ymin = int(ymin * target_size / h)
            xmax = int(xmax * target_size / w)
            ymax = int(ymax * target_size / h)
            print(xmin, ymin, xmax, ymax, class_idx)

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    dataset = VOCDataset()
    print(len(dataset))
    print(dataset[0])
