"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import random
import torch.utils.data as data
import cv2
import numpy as np
import xml.etree.ElementTree as ET

try:
    from .data_augment.yolov5_augment import yolov5_mosaic_augment, yolov5_mixup_augment, yolox_mixup_augment
except:
    from data_augment.yolov5_augment import yolov5_mosaic_augment, yolov5_mixup_augment, yolox_mixup_augment


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        # VOC类型字段：key是类型名称，value是类型索引
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
            [[真实框的坐标和类别索引], [真实框的坐标和类别索引], ... ]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt if i % 2 == 0 else cur_pt
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_ind]

        return res  # [[x1, y1, x2, y2, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, 
                 img_size=640,
                 data_dir=None,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 trans_config=None,
                 transform=None,
                 is_train=False,
                 load_cache=False
                 ):
        self.root = data_dir
        self.img_size = img_size # 最终转化的图片尺寸，因为每个图片的尺寸不一样，所以需要统一
        self.image_set = image_sets
        self.target_transform = VOCAnnotationTransform()
        self._annopath = osp.join('%s', 'Annotations', '%s.xml') # 标签路径模板
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg') # 图片路径模板
        self.ids = list()
        self.is_train = is_train
        self.load_cache = load_cache
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            """
            读取VOC2007和VOC2012图片ID列表
            D:/my code/yolo_data\VOC2007\ImageSets\Main\trainval.txt
            D:/my code/yolo_data\VOC2012\ImageSets\Main\trainval.txt
            """
            # print(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt'))
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        # augmentation
        self.transform = transform
        self.mosaic_prob = trans_config['mosaic_prob'] if trans_config else 0.0
        self.mixup_prob = trans_config['mixup_prob'] if trans_config else 0.0
        self.trans_config = trans_config
        # 截取前100张图片训练，测试代码完整流程
        self.ids = self.ids[:100]
        print('==============================')
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('==============================')

        # load cache data
        if load_cache:
            self._load_cache()


    def __getitem__(self, index):
        image, target, deltas = self.pull_item(index)
        return image, target, deltas


    def __len__(self) -> int:
        return len(self.ids)


    def _load_cache(self):
        # load image cache
        self.cached_images = []
        self.cached_targets = []
        dataset_size = len(self.ids)

        print('loading data into memory ...')
        for i in range(dataset_size):
            if i % 5000 == 0:
                print("[{} / {}]".format(i, dataset_size))
            # load an image
            image, image_id = self.pull_image(i)
            # print(image.shape) (h, w, channel)
            orig_h, orig_w, _ = image.shape
            
            # resize image
            # 计算缩放比例：目标尺寸 / 原始最大边长
            # 保持宽高比，使得最大边等于 img_size
            r = self.img_size / max(orig_h, orig_w)
            # 输入: self.img_size=640, orig_h=500, orig_w=800
            #       r = 640 / max(500, 800) = 640 / 800 = 0.8
            
            # 如果缩放比例不为1，则执行缩放操作
            if r != 1:
                # 选择线性插值方法，适合大多数图像缩放场景
                interp = cv2.INTER_LINEAR
                # 输入: interp = cv2.INTER_LINEAR (双线性插值)
                
                # 计算新的宽度和高度（保持宽高比）
                new_size = (int(orig_w * r), int(orig_h * r))
                # 输入: orig_w=800, orig_h=500, r=0.8
                #       new_size = (int(800*0.8), int(500*0.8)) = (640, 400)
                # 输出: new_size.shape = (2,) - (宽度, 高度)
                
                # 使用cv2.resize调整图像尺寸
                image = cv2.resize(image, new_size, interpolation=interp)
                # 输入: image.shape = (500, 800, 3) - (h, w, c)
                #       new_size = (640, 400) - (w, h)
                # 输出: image.shape = (400, 640, 3) - 缩放后的图像
            
            # 获取缩放后的图像尺寸
            img_h, img_w = image.shape[:2]
            # 输入: image.shape = (400, 640, 3)
            # 输出: img_h=400, img_w=640
            
            # 将处理后的图像缓存到列表中
            self.cached_images.append(image)
            # 输入: self.cached_images.append(shape=(400, 640, 3))

            # load target cache
            anno = ET.parse(self._annopath % image_id).getroot()
            anno = self.target_transform(anno) 
            anno = np.array(anno).reshape(-1, 5) # 转化为numpy数组，anno.shape = (n, 5)
            boxes = anno[:, :4]
            labels = anno[:, 4]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / orig_w * img_w # x1、x2坐标转换为缩放后的坐标
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / orig_h * img_h # y1、y2坐标转换为缩放后的坐标
            self.cached_targets.append({"boxes": boxes, "labels": labels})
            """
               boxes:
               [
                  [x1, y1, x2, y2],  # 目标1的边界框
                  [x1, y1, x2, y2],  # 目标2的边界框
                  [x1, y1, x2, y2],  # 目标3的边界框
               ]
               labels:
               [
                  [0],  # 目标1的类别标签索引
                  [2],  # 目标2的类别标签索引
                  [3],  # 目标3的类别标签索引
               ]
            """
        

    def load_image_target(self, index):
        if self.load_cache:
            image = self.cached_images[index]
            target = self.cached_targets[index]
            height, width, channels = image.shape
            target["orig_size"] = [height, width]
        else:
            # load an image
            img_id = self.ids[index]
            image = cv2.imread(self._imgpath % img_id)
            height, width, channels = image.shape

            # laod an annotation
            anno = ET.parse(self._annopath % img_id).getroot()
            if self.target_transform is not None:
                anno = self.target_transform(anno)

            # guard against no boxes via resizing
            anno = np.array(anno).reshape(-1, 5) # 转化为numpy数组，anno.shape = (n, 5)
            target = {
                "boxes": anno[:, :4],
                "labels": anno[:, 4],
                "orig_size": [height, width]
            }
            """
               boxes:
               [
                  [x1, y1, x2, y2],  # 目标1的边界框
                  [x1, y1, x2, y2],  # 目标2的边界框
                  [x1, y1, x2, y2],  # 目标3的边界框
               ]
               labels:
               [
                  [0],  # 目标1的类别标签索引
                  [2],  # 目标2的类别标签索引
                  [3],  # 目标3的类别标签索引
               ]
            """
        # image.shape = (height, width, channels)
        return image, target

    # 马赛克增强
    def load_mosaic(self, index):
        # load 4x mosaic image
        index_list = np.arange(index).tolist() + np.arange(index+1, len(self.ids)).tolist()
        id1 = index
        id2, id3, id4 = random.sample(index_list, 3)
        indexs = [id1, id2, id3, id4]

        # load images and targets
        image_list = []
        target_list = []
        for index in indexs:
            img_i, target_i = self.load_image_target(index)
            image_list.append(img_i)
            target_list.append(target_i)

        # Mosaic
        if self.trans_config['mosaic_type'] == 'yolov5_mosaic':
            image, target = yolov5_mosaic_augment(
                image_list, target_list, self.img_size, self.trans_config, self.is_train)

        return image, target

    # 混合增强
    def load_mixup(self, origin_image, origin_target):
        # YOLOv5 type Mixup
        if self.trans_config['mixup_type'] == 'yolov5_mixup':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_mosaic(new_index)
            image, target = yolov5_mixup_augment(
                origin_image, origin_target, new_image, new_target)
        # YOLOX type Mixup
        elif self.trans_config['mixup_type'] == 'yolox_mixup':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_image_target(new_index)
            image, target = yolox_mixup_augment(
                origin_image, origin_target, new_image, new_target, self.img_size, self.trans_config['mixup_scale'])

        return image, target
    

    def pull_item(self, index):
        if random.random() < self.mosaic_prob:
            # load a mosaic image
            mosaic = True
            image, target = self.load_mosaic(index)
        else:
            mosaic = False
            # load an image and target
            image, target = self.load_image_target(index)

        # MixUp
        if random.random() < self.mixup_prob:
            image, target = self.load_mixup(image, target)

        # augment 图像增强
        image, target, deltas = self.transform(image, target, mosaic)

        return image, target, deltas


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index] # img_id = ('D:/my code/yolo_data/VOC2007', '000005')
        # self._imgpath % img_id表示字符串格式化 '%s/JPEGImages/%s.jpg' % ('D:/my code/yolo_data/VOC2007', '000005')
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id


    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt


if __name__ == "__main__":
    import argparse
    from .build import build_transform
    
    parser = argparse.ArgumentParser(description='VOC-Dataset')

    # opt
    # parser.add_argument('--root', default='/Users/frank/code/ai/yolo_data',
    #                     help='data root')
    parser.add_argument('--root', default='D:/my code/yolo_data',
                        help='data root')
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='input image size.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')
    parser.add_argument('--is_train', action="store_true", default=False,
                        help='mixup augmentation.')
    parser.add_argument('--load_cache', action="store_true", default=False,
                        help='load cached data.')
    
    args = parser.parse_args()

    trans_config = {
        'aug_type': 'yolov5',            # 或者改为'ssd'来使用SSD风格的数据增强
        # Basic Augment
        'degrees': 0.0,                  # 可以修改数值来决定旋转图片的程度，如改为YOLOX默认的10.0
        'translate': 0.2,                # 可以修改数值来决定平移图片的程度，
        'scale': [0.1, 2.0],             # 图片尺寸扰动的比例范围
        'shear': 0.0,                    # 可以修改数值来决定旋转图片的程度，如改为YOLOX默认的2.0
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        # Mosaic & Mixup
        'mosaic_prob': 1.0,              # 使用马赛克增强的概率：0～1
        'mixup_prob': 1.0,               # 使用混合增强的概率：0～1
        'mosaic_type': 'yolov5_mosaic',
        'mixup_type': 'yolox_mixup',     # 或者改为'yolov5_mixup'，使用yolov5风格的混合增强
        'mixup_scale': [0.5, 1.5]
    }
    transform, trans_cfg = build_transform(args, trans_config, 32, args.is_train)

    dataset = VOCDetection(
        img_size=args.img_size,
        data_dir=args.root,
        trans_config=trans_config,
        transform=transform,
        is_train=args.is_train,
        load_cache=args.load_cache
        )
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]
    print('Data length: ', len(dataset))

    for i in range(1):
        image, target, deltas = dataset.pull_item(i)
        """
        dataset数据格式为（image, target）：
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
        """
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        # to uint8
        image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            if x2 - x1 > 1 and y2 - y1 > 1:
                cls_id = int(label)
                color = class_colors[cls_id]
                # class name
                label = VOC_CLASSES[cls_id]
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                # put the test on the bbox
                cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)