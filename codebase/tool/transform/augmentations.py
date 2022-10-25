import random
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import numpy as np
from PIL import Image


def identity(img, v):
    """
    图片本身（不做增强）
    :param img: 图片
    :param v:   操作强度
    :return:    处理后的图片
    """
    return img


def posterize(img, v):
    """
    色调分离
    :param img: 图片
    :param v:   操作强度
    :return:    处理后的图片
    """
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def contrast(img, v):
    """
    调整对比度
    :param img: 图片
    :param v:   操作强度
    :return:    处理后的图片
    """
    assert 0 <= v <= 0.9
    if random.random() > 0.5:
        v = -v
    return PIL.ImageEnhance.Contrast(img).enhance(1 + v * random.random())


def brightness(img, v):
    """
    调整亮度
    :param img: 图片
    :param v:   操作强度
    :return:    处理后的图片
    """
    assert 0 <= v <= 0.9
    if random.random() > 0.5:
        v = -v
    return PIL.ImageEnhance.Brightness(img).enhance(1 + v * random.random())


def sharpness(img, v):
    """
    调整锐度
    :param img: 图片
    :param v:   操作强度
    :return:    处理后的图片
    """
    assert 0 <= v <= 0.9
    if random.random() > 0.5:
        v = -v
    return PIL.ImageEnhance.Sharpness(img).enhance(1 + v * random.random())


def cutout(img, v):
    """
    随机遮挡
    :param img: 图片
    :param v:   操作强度
    :return:    处理后的图片
    """
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img
    v = v * img.size[0]
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def augment_list():
    """
    所有增强操作
    :return:    所有增强操作
    """
    operation_list = [
        (identity, 0, 1),
        (posterize, 8, 4),
        (contrast, 0.0, 0.6),
        (brightness, 0.0, 0.6),
        (sharpness, 0.0, 0.6),
        (cutout, 0.0, 0.05),
    ]
    return operation_list


class RandAugment:
    """
    RandAugment
    """
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = augment_list()

    def __call__(self, img):
        random.shuffle(self.augment_list)
        for i in range(self.n):
            (op, minval, maxval) = self.augment_list[i]
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img

