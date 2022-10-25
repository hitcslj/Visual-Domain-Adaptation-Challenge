import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import random
import torch
import math


def setup_seed(seed):
    """
    固定随机数种子
    :param seed:    种子
    :return:        无
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def label_to_one_hot(label, n_class=5, return_mask=False):
    """
    标签转为one hot标签

    思路：
        允许gt中出现-1，-1将被翻译为5维全0

    :param label:       标签
    :param n_class:     类别数，int
    :param return_mask: 是否返回mask，bool
    :return:            one hot标签，(mask)
    """
    flag = False
    if len(label.shape) == 2:
        label = label.unsqueeze(dim=0)
        flag = True
    mask = (label == -1)
    label[mask] = 0
    one_hot = nn.functional.one_hot(label.type(torch.int64), num_classes=n_class)
    one_hot = one_hot.permute(0, 3, 1, 2)
    one_hot[mask.repeat(1, n_class, 1, 1)] = 0
    if flag:
        one_hot = one_hot.squeeze(dim=0)
    if return_mask:
        return one_hot, mask
    else:
        return one_hot


def plot_probability(data, interval=0.01, density=0.75, picture_name='', title='', xlabel=''):
    """
    根据数据频数分布，近似画概率密度曲线
    会选取出有数据的区间来画图，但是边上的有的柱子太矮了，画出来的图里可能看不到
    :param data:            数据
    :param interval:        采样区间间隔
    :param density:         柱子的密集程度，取值0-1
    :param picture_name:    图片名称
    :param title:           标题
    :param xlabel:          横坐标
    :return:                无
    """
    min_number = min(data)
    max_number = max(data)
    n1 = math.floor(abs(max_number) / interval) + 1
    n2 = math.floor(abs(min_number) / interval) + 1
    if min_number > 0:
        n2 = -n2
    x = []
    y = []
    for i in range(n1 + n2):
        x1 = i * interval - n2 * interval
        x2 = x1 + interval
        x.append((x1 + x2) / 2)
        sum = 0
        for j in range(len(data)):
            if (data[j] >= x1) and (data[j] < x2):
                sum = sum + 1
        y.append(sum)
    plt.bar(x, y, width=interval * density)
    plt.xlabel(xlabel)
    plt.ylabel('num')
    plt.title(title)
    if picture_name != '':
        plt.savefig(picture_name + '.png', dpi=600)
    plt.show()
