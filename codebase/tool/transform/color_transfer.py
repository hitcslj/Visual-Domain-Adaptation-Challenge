import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(0)


def color_transfer(image, target):
    """
    颜色匹配

    思路：
        源于一篇很早之前的工作，似乎是上世纪九十年代
        思路很简单，把图片由rgb颜色空间转为lab颜色空间
        然后将图片lab三通道的均值、标准差调整得与目标图片相同
        最后转回rgb颜色空间即可

    :param image:   要改变颜色的图片，PIL.Image
    :param target:  要匹配的目标图片，PIL.Image
    :return:        匹配后的图片，PIL.Image
    """
    image = np.array(image)
    target = np.array(target)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
    target_mean = np.mean(target_lab, axis=(0, 1), keepdims=True)
    target_std = np.std(target_lab, axis=(0, 1), keepdims=True)
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    mean = np.mean(image_lab, axis=(0, 1), keepdims=True)
    std = np.std(image_lab, axis=(0, 1), keepdims=True)
    image_lab = ((image_lab - mean) / std) * target_std + target_mean
    image_lab[image_lab < 0] = 0
    image_lab[image_lab > 255] = 255
    image = cv2.cvtColor(image_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(image)

 
