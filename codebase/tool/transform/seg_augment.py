import random
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import numpy as np
from PIL import Image
from utils.convert import visualize_gt


def identity(img, gt=None, v=0):
    """
    图片本身（不做增强）
    :param img: 图片，PIL.Image
    :param gt:  分割标注，PIL.Image or None
    :param v:   操作强度，float
    :return:    image: PIL.Image, gt: PIL.Image or None
    """
    return img, gt


def shear_x(img, gt=None, v=0.1):
    """
    图片沿X方向倾斜
    :param img: 图片，PIL.Image
    :param gt:  分割标注，PIL.Image or None
    :param v:   操作强度，float
    :return:    image: PIL.Image, gt: PIL.Image or None
    """
    assert 0 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    v = v * random.random()
    img = img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
    if gt is not None:
        gt = gt.transform(gt.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
    return img, gt


def shear_y(img, gt=None, v=0.1):
    """
    图片沿Y方向倾斜
    :param img: 图片，PIL.Image
    :param gt:  分割标注，PIL.Image or None
    :param v:   操作强度，float
    :return:    image: PIL.Image, gt: PIL.Image or None
    """
    assert 0 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    v = v * random.random()
    img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
    if gt is not None:
        gt = gt.transform(gt.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
    return img, gt


def translate_x(img, gt=None, v=0.1):
    """
    图片沿X方向平移
    :param img: 图片，PIL.Image
    :param gt:  分割标注，PIL.Image or None
    :param v:   操作强度，float
    :return:    image: PIL.Image, gt: PIL.Image or None
    """
    assert 0 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    v = v * random.random()
    v = v * img.size[0]
    img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
    if gt is not None:
        gt = gt.transform(gt.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
    return img, gt


def translate_y(img, gt=None, v=0.1):
    """
    图片沿Y方向平移
    :param img: 图片，PIL.Image
    :param gt:  分割标注，PIL.Image or None
    :param v:   操作强度，float
    :return:    image: PIL.Image, gt: PIL.Image or None
    """
    assert 0 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    v = v * random.random()
    v = v * img.size[1]
    img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
    if gt is not None:
        gt = gt.transform(gt.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
    return img, gt


def rotate(img, gt=None, v=0.1):
    """
    图片旋转
    :param img: 图片，PIL.Image
    :param gt:  分割标注，PIL.Image or None
    :param v:   操作强度，float
    :return:    image: PIL.Image, gt: PIL.Image or None
    """
    assert 0 <= v <= 30
    if random.random() > 0.5:
        v = -v
    v = v * random.random()
    img = img.rotate(v)
    if gt is not None:
        gt = gt.rotate(v)
    return img, gt


def flip(img, gt=None, v=0):
    """
    图片水平翻转
    :param img: 图片，PIL.Image
    :param gt:  分割标注，PIL.Image or None
    :param v:   操作强度，float
    :return:    image: PIL.Image, gt: PIL.Image or None
    """
    img = PIL.ImageOps.mirror(img)
    if gt is not None:
        gt = PIL.ImageOps.mirror(gt)
    return img, gt


def augment_list():
    """
    所有增强操作
    :return:    所有增强操作
    """
    operation_list = [
        (identity, 0, 1),
        (rotate, 0, 15),
        (shear_x, 0.0, 0.15),
        (shear_y, 0.0, 0.15),
        (translate_x, 0.0, 0.15),
        (translate_y, 0.0, 0.15),
        (flip, 0, 1)
    ]
    return operation_list


class SegAugment:
    """
    SegAugment

    使用：
        augment = SegAugment(n, m)
        image, gt = augment(image, gt)
        由于需要image和gt两个参数，暂时不支持插入transform
    说明：
        n: int, 0 < n <=7
        m: int, 0 < m <=30

    """
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = augment_list()

    def __call__(self, img, gt=None):
        """
        每次随机应用列表中的n个不同的数据增强操作
        :param img:     图片，PIL.Image
        :param gt:      分割标注，PIL.Image or None
        :return:        image: PIL.Image, gt: PIL.Image or None
        """
        random.shuffle(self.augment_list)
        for i in range(self.n):
            (op, minval, maxval) = self.augment_list[i]
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img, gt = op(img, gt=gt, v=val)
        return img, gt


if __name__ == '__main__':
    print(1)
