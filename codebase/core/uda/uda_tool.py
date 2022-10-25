from tool.transform.restore import denormalize, tensor_to_image
from tool.transform.color_transfer import color_transfer
from tool.transform.gaussian import gaussian_kernel
from tool.transform.rand_augment import RandAugment
from tool.transform.seg_augment import SegAugment
from tool.transform.mixup import mix_up
from torchvision import transforms
from utils.utils import label_to_one_hot
from copy import deepcopy
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import math


def source_target_color_transfer(data, target):
    """
    颜色匹配

    思路：
        这里调用了tool.transform.color_transfer中的color_transfer函数
        但是该函数要求输入为PIL.Image，因此这里做了一些数据类型转换
    使用：
        data1 = source_target_color_transfer(data1, data2)

    :param data:    要进行颜色匹配的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    :param target:  目标颜色的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    :return:        颜色改变后的data数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    """
    data = deepcopy(data)
    s_image = tensor_to_image(target)
    t_image = tensor_to_image(data)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    for i in range(len(s_image)):
        image = color_transfer(t_image[i], s_image[i])
        data[i, :, :, :] = transform(image).to(data.device)
    return data


def source_target_mix_up(s_data, s_gt, t_data, t_gt, mode='replace', p=1.0, num=1):
    """
    source domain & target domain，数据混合

    思路：
        调用tool.transform.mixup中的mixup函数，该函数要求输入为PIL.Image，因此这里做了一些数据类型转换
        mix up之后，对mixed data做相应的数据增强，这是通常的self-learning方法
    使用：
        mixed_data, mixed_label = source_target_mix_up(s_data, s_gt, t_data, t_gt)
    说明：
        如果是replace mix up，则是用target domain的数据来替换source domain数据中的某个物体，得到混合数据
        如果是其他类型mix up的话，则是用source domain的数据混到target domain的数据中，得到混合数据

    :param s_data:  source domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    :param s_gt:    source domain的数据的标注，tensor4d，eg. (1(bs), 5, 1080, 1920)
    :param t_data:  target domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    :param t_gt:    target domain的数据的伪标注，tensor4d，eg. (1(bs), 5, 1080, 1920)
    :param mode:    mix up模式，str，['hard', 'object', 'soft', 'replace']
    :param p:       执行mix的概率，float
    :param num:     混合的物体种类数，int
    :return:        data: tensor4d，eg. (1(bs), 3, 1080, 1920),
                    label: tensor4d，eg. (1(bs), 5, 1080, 1920)
    """
    s_image, s_image_gt = tensor_to_image(s_data, s_gt)
    t_image, t_image_gt = tensor_to_image(t_data, t_gt)
    data = deepcopy(s_data)
    label = label_to_one_hot(s_gt).type(torch.float32)
    for i in range(data.shape[0]):
        if mode == 'replace':
            temp1, temp2 = mix_up(s_image[i], s_image_gt[i], t_image[i], t_image_gt[i],
                                  mode=mode, k_max=1.0, k_min=1.0, p=p, num=num)
        else:
            temp1, temp2 = mix_up(t_image[i], t_image_gt[i], s_image[i], s_image_gt[i],
                                  mode=mode, k_max=1.0, k_min=1.0, p=p, num=num)
        temp1, temp2 = mixed_data_transform(temp1, temp2)
        data[i, :, :, :] = temp1
        label[i, :, :, :] = temp2
    return data, label


def mixed_data_transform(data, label):
    """
    针对混合数据的数据增强操作

    思路：
        这里仿照了DAFormer的做法
        DAFormer中的数据增强包括color jitter、gaussian blur，
        我查了一下，color jitter包含亮度、对比度、饱和度、色调的变化，这与我们的RandAugment很类似
        RandAugment中包含了亮度、对比度、饱和度、锐度、色阶化等多种变化，因此此处使用RandAugment代替color jitter
    使用：
        目前此函数无需外部调用，已在source_target_mix_up中调用
    说明：
        1.RandAugment中目前未包含色调hue的变化，而DAFormer的color jitter中有此种操作
            补充：现已加入，2022/8/22
        2.高斯模糊的设置与DAFormer不完全相同
        3.以上两点见官方代码：
            VisDA-2022.codes.mmseg.models.utils.dacs_transform

    :param data:    数据，PIL.Image
    :param label:   标签，tensor3d，eg. (5, 1080, 1920)
    :return:        data: tensor3d，eg. (3, 1080, 1920),
                    label: tensor3d，eg. (5, 1080, 1920)
    """
    gt = Image.fromarray(label.argmax(dim=0).type(torch.uint8).numpy())
    rand_augment = RandAugment(n=4, m=30)
    seg_augment = SegAugment(n=1, m=20)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    gaussian = gaussian_kernel(kernel_size=11, sigma=2)
    data = rand_augment(data)
    data, gt = seg_augment(data, gt)
    data = transform(data)
    data = gaussian(data.unsqueeze(dim=0)).squeeze(dim=0)
    label = label_to_one_hot(torch.tensor(np.array(gt))).type(torch.float32)
    return data, label


"""
- - - - - - - - - -
以下3个函数目前暂时已经弃用
- - - - - - - - - -
"""


def select_representative_point(prob, threshold=1.5):
    """
    选取代表点
    :param prob:        预测每个点属于各个类别的概率
    :param threshold:   选定为代表点的阈值，阈值越大，选的点越少
    :return:            代表点
    """
    if not (prob.sum(dim=1) == 1).sum() == prob.shape[0] * prob.shape[2] * prob.shape[3]:
        prob = prob.softmax(dim=1)
    result = prob.argmax(dim=1)
    entropy = (-torch.log(prob) * prob).sum(dim=1)
    certainty = 1 / entropy
    for i in range(prob.shape[1]):
        certainty_per_class = certainty[result == i]
        if certainty_per_class.sum() != 0:
            certainty[result == i] = (certainty[
                                          result == i] - certainty_per_class.mean()) / certainty_per_class.std()
        entropy_per_class = entropy[result == i]
        if entropy_per_class.sum() != 0:
            entropy[result == i] = (entropy[result == i] - entropy_per_class.mean()) / entropy_per_class.std()
    representative = (certainty > threshold) | (entropy < -threshold)
    # print(representative.sum() / (prob.shape[0] * prob.shape[2] * prob.shape[3]))
    result[~representative] = -1
    return result


def get_pseudo_seg(representative_point, kernel_size=25, sigma=8, n_class=5, threshold=0.1):
    """
    生成伪语义分割标注
    :param representative_point:    代表点
    :param kernel_size:             核的大小
    :param sigma:                   标准差
    :param n_class:                 类别数
    :param threshold:               选择为置信标注点的阈值，阈值越小选择的点越少，及伪分割中标注点越少
    :return:                        伪分割
    """
    # mask = (representative_point == -1)
    # representative_point[mask] = 0
    # pseudo_prob = nn.functional.one_hot(representative_point, num_classes=n_class)
    # pseudo_prob = pseudo_prob.permute(0, 3, 1, 2)
    # pseudo_prob[mask.repeat(1, n_class, 1, 1)] = 0
    pseudo_prob = label_to_one_hot(representative_point, n_class=n_class)
    device = representative_point.device
    gaussian = gaussian_kernel(kernel_size=kernel_size, sigma=sigma, channels=n_class).to(device)
    pseudo_prob = gaussian(pseudo_prob.type(torch.float32))
    max_prob = pseudo_prob.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    max_prob[max_prob == 0] = 1
    pseudo_prob = pseudo_prob / max_prob
    result = pseudo_prob.argmax(dim=1)
    seg = (pseudo_entropy(pseudo_prob) < threshold)
    result[~seg] = -1
    return result


def pseudo_entropy(prob):
    """
    伪熵
    与预测熵的区别是考虑了可能性的绝对值大小
    :param prob:    可能性
    :return:        伪熵
    """
    if prob.min() <= 0:
        prob = prob - prob.min() + 1e-6
    standard_prob = prob / prob.sum(dim=1, keepdim=True)
    entropy = (-torch.log(standard_prob) * standard_prob).sum(dim=1)
    my_entropy = entropy / prob.sum(dim=1)
    return my_entropy
