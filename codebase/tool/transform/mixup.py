from utils.convert import visualize_gt, visualize_data
from utils.utils import label_to_one_hot
from copy import deepcopy
from PIL import Image
import numpy as np
import random
import torch
import math


def mix_up(data_target, gt_target, data_source, gt_source, mode='hard', k_max=0.5, k_min=None, p=-1, num=1):
    """
    Mix Up
    统一接口

    思路：
        将源数据中的一些东西mix到目标数据中，之后返回mix后的目标数据
    使用：
        data_target, label_target = mix_up(data_target, gt_target, data_source, gt_source)

    :param data_target:     目标数据，PIL.Image
    :param gt_target:       目标数据的分割标注，PIL.Image
    :param data_source:     源数据，PIL.Image
    :param gt_source:       源数据的分割标注，PIL.Image
    :param mode:            mix up模式，str，['hard', 'object', 'soft', 'replace']
    :param k_max:           mix区域大小比例上界，float，[0, 1]
    :param k_min:           mix区域大小比例下界，float，[0, 1]
    :param p:               执行mix的概率，float，[0, 1]
    :param num:             混合的物体种类数，int，1-4，仅在mode为object或replace时有用
    :return:                data_target: PIL.Image,
                            label_target: tensor3d，eg. (5, 1080, 1920)
    """
    assert mode in ['hard', 'object', 'soft', 'replace']
    label_target = 0
    if p > 0:
        assert 0 < p <= 1
        if random.random() > p:
            if gt_target is not None:
                label_target = label_to_one_hot(torch.tensor(np.array(gt_target))).type(torch.float32)
            return data_target, label_target
    data_target = np.array(data_target)
    gt_target = np.array(gt_target)
    data_source = np.array(data_source)
    gt_source = np.array(gt_source)
    if k_min is None:
        k_min = k_max / 2
    h_min = min(data_target.shape[0], data_source.shape[0])
    w_min = min(data_target.shape[1], data_source.shape[1])
    h = np.floor(np.random.uniform(k_min * h_min, k_max * h_min))
    w = np.floor(np.random.uniform(k_min * w_min, k_max * w_min))
    size = [h, w]
    if mode == 'hard':
        data_target, label_target = hard_mix_up(data_target, gt_target, data_source, gt_source, size)
    elif mode == 'object':
        data_target, label_target = object_mix_up(data_target, gt_target, data_source, gt_source, size, num)
    elif mode == 'soft':
        data_target, label_target = soft_mix_up(data_target, gt_target, data_source, gt_source, size)
    elif mode == 'replace':
        data_target, label_target = replace_mix_up(data_target, gt_target, data_source, gt_source, size, num)
    data_target = Image.fromarray(data_target).convert("RGB")
    return data_target, label_target


def hard_mix_up(data_target, gt_target, data_source, gt_source, size=None):
    """
    硬性混合

    思路：
        用源数据中的一个矩形区域直接替换目标数据中相同大小的一个矩形区域
    使用：
        在mix_up函数中调用，通常不直接使用

    :param data_target:     目标数据，PIL.Image
    :param gt_target:       目标数据的分割标注，PIL.Image
    :param data_source:     源数据，PIL.Image
    :param gt_source:       源数据的分割标注，PIL.Image
    :param size:            混合区域的大小，list，eg. [540, 960]
    :return:                data_target: PIL.Image,
                            label_target: tensor3d，eg. (5, 1080, 1920)
    """
    shape_source = [data_source.shape[0], data_source.shape[1]]
    shape_target = [data_target.shape[0], data_target.shape[1]]
    if size is None:
        size = [min(0.5 * shape_target[0], 0.5 * shape_source[0]), min(0.5 * shape_target[1], 0.5 * shape_source[1])]
    mask_source = get_random_mask(shape_source, size)
    mask_target = get_random_mask(shape_target, size)
    data_target[mask_target, :] = data_source[mask_source, :]
    if gt_target is not None and gt_source is not None:
        gt_target[mask_target] = gt_source[mask_source]
        label_target = label_to_one_hot(torch.tensor(np.array(gt_target))).type(torch.float32)
    else:
        label_target = 0
    return data_target, label_target


def object_mix_up(data_target, gt_target, data_source, gt_source, size=None, num=1):
    """
    物体混合

    思路：
        把源数据中某个区域的某几物体抠出来，放到目标数据中的某个随机位置
    使用：
        在mix_up函数中调用，通常不直接使用

    :param data_target:     目标数据，PIL.Image
    :param gt_target:       目标数据的分割标注，PIL.Image
    :param data_source:     源数据，PIL.Image
    :param gt_source:       源数据的分割标注，PIL.Image
    :param size:            混合区域的大小，list，eg. [540, 960]
    :param num:             混合的物体种类数，int
    :return:                data_target: PIL.Image,
                            label_target: tensor3d，eg. (5, 1080, 1920)
    """
    assert gt_target is not None and gt_source is not None
    # 选取每种类别物体的概率，关注比例即可，不需要归一化
    p_class = np.array([0, 10.0, 3.0, 2.0, 2.0])
    shape_source = [data_source.shape[0], data_source.shape[1]]
    shape_target = [data_target.shape[0], data_target.shape[1]]
    if size is None:
        size = [min(0.5 * shape_target[0], 0.5 * shape_source[0]), min(0.5 * shape_target[1], 0.5 * shape_source[1])]
    mask_source = get_random_mask(shape_source, size)
    mask_target = get_random_mask(shape_target, size)
    gt = torch.tensor(gt_source[mask_source])
    has = (gt.unsqueeze(dim=0) == torch.tensor(list(range(5))).reshape(-1, 1, 1))
    has = (has.sum(dim=2).sum(dim=1) > 0).numpy()
    mask_object = np.zeros(shape=gt.shape, dtype=bool)
    for _ in range(num):
        temp = deepcopy(p_class)
        temp[~has] = 0
        if temp.sum() == 0:
            break
        temp = temp / temp.sum()
        for i in range(1, 5):
            temp[i] = temp[i] + temp[i - 1]
        temp = temp - random.random()
        temp[temp < 0] = 1
        temp[~has] = 1
        index = np.argmin(temp)
        mask_object = ((gt == index).numpy()) | mask_object
        has[index] = False
    mask_source[mask_source] = mask_object
    mask_target[mask_target] = mask_object
    data_target[mask_target, :] = data_source[mask_source, :]
    gt_target[mask_target] = gt_source[mask_source]
    label_target = label_to_one_hot(torch.tensor(np.array(gt_target))).type(torch.float32)
    return data_target, label_target


def soft_mix_up(data_target, gt_target, data_source, gt_source, size=None):
    """
    软性混合

    思路：
        把源数据中某个矩形区域的数据取出，在目标数据中某个相同大小的矩形区域位置，进行a*A+(1-a)*B的运算
    使用：
        在mix_up函数中调用，通常不直接使用

    :param data_target:     目标数据，PIL.Image
    :param gt_target:       目标数据的分割标注，PIL.Image
    :param data_source:     源数据，PIL.Image
    :param gt_source:       源数据的分割标注，PIL.Image
    :param size:            混合区域的大小，list，eg. [540, 960]
    :return:                data_target: PIL.Image,
                            label_target: tensor3d，eg. (5, 1080, 1920)
    """
    shape_source = [data_source.shape[0], data_source.shape[1]]
    shape_target = [data_target.shape[0], data_target.shape[1]]
    if size is None:
        size = [min(0.5 * shape_target[0], 0.5 * shape_source[0]), min(0.5 * shape_target[1], 0.5 * shape_source[1])]
    mask_source = get_random_mask(shape_source, size)
    mask_target = get_random_mask(shape_target, size)
    k = random.random()
    data_target[mask_target, :] = data_source[mask_source, :] * k + data_target[mask_target, :] * (1 - k)
    if gt_target is not None and gt_source is not None:
        label_target = label_to_one_hot(torch.tensor(np.array(gt_target))).type(torch.float32)
        label_source = label_to_one_hot(torch.tensor(np.array(gt_source))).type(torch.float32)
        label_target[:, mask_target] = label_source[:, mask_source] * k + label_target[:, mask_target] * (1 - k)
    else:
        label_target = 0
    return data_target, label_target


def replace_mix_up(data_target, gt_target, data_source, gt_source, size=None, num=1):
    """
    替换混合

    思路：
        与物体混合对偶，选出目标数据中某个区域的某几种物体，然后用源数据中某个随机位置来替换这些物体
    使用：
        在mix_up函数中调用，通常不直接使用

    :param data_target:     目标数据，PIL.Image
    :param gt_target:       目标数据的分割标注，PIL.Image
    :param data_source:     源数据，PIL.Image
    :param gt_source:       源数据的分割标注，PIL.Image
    :param size:            混合区域的大小，list，eg. [540, 960]
    :param num:             混合的物体种类数，int
    :return:                混合后的目标数据
    """
    assert gt_target is not None and gt_source is not None
    # 选取每种类别物体的概率，关注比例即可，不需要归一化
    p_class = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    shape_source = [data_source.shape[0], data_source.shape[1]]
    shape_target = [data_target.shape[0], data_target.shape[1]]
    if size is None:
        size = [min(0.5 * shape_target[0], 0.5 * shape_source[0]), min(0.5 * shape_target[1], 0.5 * shape_source[1])]
    mask_source = get_random_mask(shape_source, size)
    mask_target = get_random_mask(shape_target, size)
    gt = torch.tensor(gt_target[mask_source])
    has = (gt.unsqueeze(dim=0) == torch.tensor(list(range(5))).reshape(-1, 1, 1))
    has = (has.sum(dim=2).sum(dim=1) > 0).numpy()
    mask_object = np.zeros(shape=gt.shape, dtype=bool)
    for _ in range(num):
        temp = deepcopy(p_class)
        temp[~has] = 0
        if temp.sum() == 0:
            break
        temp = temp / temp.sum()
        for i in range(1, 5):
            temp[i] = temp[i] + temp[i - 1]
        temp = temp - random.random()
        temp[temp < 0] = 1
        temp[~has] = 1
        index = np.argmin(temp)
        mask_object = ((gt == index).numpy()) | mask_object
        has[index] = False
    mask_source[mask_source] = mask_object
    mask_target[mask_target] = mask_object
    data_target[mask_target, :] = data_source[mask_source, :]
    gt_target[mask_target] = gt_source[mask_source]
    label_target = label_to_one_hot(torch.tensor(np.array(gt_target))).type(torch.float32)
    return data_target, label_target


def get_random_mask(data_shape, mask_size):
    """
    获得一个随机位置的mask
    :param data_shape:  数据尺寸，list，eg. [1080, 1920]
    :param mask_size:   mask的大小，list，eg. [540, 960]
    :return:            mask，np.array，bool，2d，eg. [1080, 1920]
    """
    h = data_shape[0]
    w = data_shape[1]
    h0 = np.floor(np.random.uniform(h - mask_size[0]))
    w0 = np.floor(np.random.uniform(w - mask_size[1]))
    h0, h1 = int(max(h0, 0)), int(min(h0 + mask_size[0], h))
    w0, w1 = int(max(w0, 0)), int(min(w0 + mask_size[1], w))
    mask = np.zeros((h, w), dtype=bool)
    mask[h0:h1, w0:w1] = True
    return mask


class MixUp:
    """
    MixUp

    使用：
        mix_up = MixUp(mode='hard', k_max=0.5, k_min=None, p=-1)
        data_target, label_target = mix_up(data_target, gt_target, data_source, gt_source)

    """
    def __init__(self, mode='hard', k_max=0.5, k_min=None, p=-1):
        self.mode = mode
        self.k_max = k_max
        self.k_min = k_min
        self.p = p

    def __call__(self, data_target, gt_target, data_source, gt_source):
        return mix_up(data_target, gt_target, data_source, gt_source,
                      mode=self.mode, k_max=self.k_max, k_min=self.k_min, p=self.p)
