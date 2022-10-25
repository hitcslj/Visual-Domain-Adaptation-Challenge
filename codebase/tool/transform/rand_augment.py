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
    :param img: 图片，PIL.Image
    :param v:   操作强度，float
    :return:    处理后的图片，PIL.Image
    """
    return img


def posterize(img, v):
    """
    色调分离
    :param img: 图片，PIL.Image
    :param v:   操作强度，float
    :return:    处理后的图片，PIL.Image
    """
    arrange = [8, 4]
    v = v * random.random() * (arrange[1] - arrange[0]) + arrange[0]
    v = int(v + 0.5)
    return PIL.ImageOps.posterize(img, v)


def contrast(img, v):
    """
    调整对比度
    :param img: 图片，PIL.Image
    :param v:   操作强度，float
    :return:    处理后的图片，PIL.Image
    """
    assert 0 <= v <= 0.9
    if random.random() > 0.5:
        v = -v
    return PIL.ImageEnhance.Contrast(img).enhance(1 + v * random.random())


def brightness(img, v):
    """
    调整亮度
    :param img: 图片，PIL.Image
    :param v:   操作强度，float
    :return:    处理后的图片，PIL.Image
    """
    assert 0 <= v <= 0.9
    if random.random() > 0.5:
        v = -v
    return PIL.ImageEnhance.Brightness(img).enhance(1 + v * random.random())


def sharpness(img, v):
    """
    调整锐度
    :param img: 图片，PIL.Image
    :param v:   操作强度，float
    :return:    处理后的图片，PIL.Image
    """
    assert 0 <= v <= 0.9
    if random.random() > 0.5:
        v = -v
    return PIL.ImageEnhance.Sharpness(img).enhance(1 + v * random.random())


def saturation(img, v):
    """
    调整饱和度
    :param img: 图片，PIL.Image
    :param v:   操作强度，float
    :return:    处理后的图片，PIL.Image
    """
    assert 0 <= v <= 0.9
    if random.random() > 0.5:
        v = -v
    return PIL.ImageEnhance.Color(img).enhance(1 + v * random.random())


def cutout(img, v):
    """
    随机遮挡
    :param img: 图片，PIL.Image
    :param v:   操作强度，float
    :return:    处理后的图片，PIL.Image
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


def hue(img, v):
    """
    调整色调
    :param img: 图片，PIL.Image
    :param v:   操作强度，float
    :return:    处理后的图片，PIL.Image
    """
    assert -0.5 <= v <= 0.5
    v = v * random.random()
    if random.random() > 0.5:
        v = -v
    img = np.array(img.convert('HSV')).astype(int)
    img[:, :, 0] = img[:, :, 0] + int(v * 255)
    mask1 = img[:, :, 0] > 255
    mask2 = img[:, :, 0] < 0
    img[:, :, 0][mask1] = img[:, :, 0][mask1] - 255
    img[:, :, 0][mask2] = img[:, :, 0][mask2] + 255
    img = Image.fromarray(img.astype(np.uint8), mode='HSV')
    img = img.convert('RGB')
    return img


def augment_list():
    """
    所有增强操作
    :return:    所有增强操作
    """
    operation_list = [
        (identity, 0, 1),
        (posterize, 0.0, 1.0),
        (contrast, 0.0, 0.6),
        (brightness, 0.0, 0.6),
        (sharpness, 0.0, 0.6),
        (saturation, 0.0, 0.6),
        (hue, 0.0, 0.4),
        (cutout, 0.0, 0.05),
    ]
    return operation_list


class RandAugment:
    """
    RandAugment

    使用：
        augment = RandAugment(n, m)
        image = augment(image)
        也可以插入到transform里：
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        transform.transforms.insert(0, RandAugment(n, m))
    说明：
        n: int, 0 < n <=8
        m: int, 0 < m <=30

    """
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = augment_list()

    def __call__(self, img):
        """
        每次随机应用列表中的n个不同的数据增强操作
        :param img:     图片，PIL.Image
        :return:        增强后的图片，PIL.Image
        """
        random.shuffle(self.augment_list)
        for i in range(self.n):
            (op, minval, maxval) = self.augment_list[i]
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img


 

