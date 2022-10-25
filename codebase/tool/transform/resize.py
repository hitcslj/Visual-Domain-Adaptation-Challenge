from torch.nn import functional as F
from utils.utils import plot_probability
import numpy as np
import torch


def resize(data, label=None, ratio=None, size=None, mode='nearest', hard=False):
    """
    同步改变图片和标签的尺寸

    使用：
        仅指定放缩比例或放缩后的尺寸，同时指定会报冲突，例：
        data, label = resize(data, label, ratio=0.5)

    :param data:    数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    :param label:   标签，tensor4d or None，eg. (1(bs), 5, 1080, 1920)
    :param ratio:   尺寸放缩比例，float
    :param size:    放缩后尺寸大小，list，eg. [540, 960]
    :param mode:    插值模式，包括['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
    :param hard:    是否允许soft-label，bool
    :return:        data, label
    """
    assert mode in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
    if ratio is None and size is None:
        print('<resize>: Please set the ratio or size when using the function')
        return
    if ratio is not None and size is not None:
        print('<resize>: The given ratio and size conflict')
        return
    if ratio is not None:
        data = F.interpolate(data, scale_factor=ratio, mode=mode)
        if label is not None:
            temp = F.interpolate(label, scale_factor=ratio, mode=mode)
            if hard:
                label = (temp > 0.5).type(label.dtype)
            else:
                label = temp.type(label.dtype)
    else:
        data = F.interpolate(data, size=size, mode=mode)
        if label is not None:
            temp = F.interpolate(label, size=size, mode=mode)
            if hard:
                label = (temp > 0.5).type(label.dtype)
            else:
                label = temp.type(label.dtype)
    return data, label


def rand_resize_ratio(data_shape, min_size, ratio):
    """
    随机放缩比例

    思路：
        从标准正态分布手动搞出来了一个分布
        特点：
            1.有严格的数据上下限
            2.能够设定一个值，最终生成的数据有50%的概率小于设定值，有50%概率大于设定值
    说明：
        运行本脚本，可以可视化此分布

    :param data_shape:  数据尺寸，list/turple，eg. [_, _, 1080, 1920]
    :param min_size:    数据最小应保持的尺寸，list，eg. [108, 192]
    :param ratio:       设定尺寸放缩比例值，float
    :return:            实际放缩比例，float
    """
    ratio_min = max(min_size[0] / data_shape[2], min_size[1] / data_shape[3])
    ratio_max = ratio * 2.5
    assert ratio_min < ratio
    a = np.random.normal(loc=0.0, scale=1.0)
    k = 1.75
    if a < 0:
        a = -a
        a = 1 + a ** k
        a = 1 / (a ** (1 / k))
        rand_ratio = (ratio - ratio_min) * a + ratio_min
    else:
        a = 1 + a ** k
        a = 1 - 1 / (a ** (1 / k))
        rand_ratio = (ratio_max - ratio) * a + ratio
    return rand_ratio


if __name__ == '__main__':
    temp = []
    for i in range(100000):
        temp.append(rand_resize_ratio([1, 3, 1080, 1920], [352, 352], 0.5))
    plot_probability(temp)
