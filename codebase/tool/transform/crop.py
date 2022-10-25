import numpy as np
import torch
import math


def random_crop(crop_size, data, label=None):
    """
    Random Crop，随机裁切

    思路：
        由于语义分割是逐pixel的，因此需要对图片和标注进行相同的裁剪操作
        裁切的大小是指定的，位置是随机的
    使用：
        data, label = random_crop(crop_size, data, label)

    :param crop_size:   要裁剪成的形状，list，eg. [224, 224]
    :param data:        数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    :param label:       one hot标签，tensor4d，eg. (1(bs), 5, 1080, 1920)
    :return:            (data, label)
                        data: tensor4d，eg. (1(bs), 3, 224, 224)
                        label: tensor4d，eg. (1(bs), 5, 224, 224)
    """
    h = data.shape[2]
    w = data.shape[3]
    h0 = np.floor(np.random.uniform(h - crop_size[0]))
    w0 = np.floor(np.random.uniform(w - crop_size[1]))
    h0, h1 = int(max(h0, 0)), int(min(h0 + crop_size[0], h))
    w0, w1 = int(max(w0, 0)), int(min(w0 + crop_size[1], w))
    return crop((h0, h1, w0, w1), data, label)


def crop(position, data, label=None):
    """
    Crop，指定位置裁切

    思路：
        由于语义分割是逐pixel的，因此需要对图片和标注进行相同的裁剪操作
        指定要裁切的区域的左上角点和右下角点，执行裁切
    使用：
        目前无外部调用，只在random_crop、cover_crop函数中调用
        但是其实也可以在外部调用：
        data, label = crop(position, data, label)

    :param position:    要裁剪的区域，list，[h0, h1, w0, w1], eg. [128, 228, 128, 328]
                        裁切矩形区域的左上角点为(h0, w0)，右下角点为(h1, w1)
    :param data:        数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    :param label:       one hot标签，tensor4d，eg. (1(bs), 5, 1080, 1920)
    :return:            data: tensor4d，eg. (1(bs), 3, 224, 224),
                        label: tensor4d，eg. (1(bs), 5, 224, 224)
    """
    h0, h1, w0, w1 = position
    data = data[:, :, h0:h1, w0:w1]
    if label is not None:
        label = label[:, :, h0:h1, w0:w1]
    return data, label


def cover_crop(crop_size, data, label=None):
    """
    Cover Crop
    将一张图片crop为若干小patch，能够完全覆盖原图

    思路：
        将一张图片分割为若干相同大小的patch，可能会由重叠，但是能够保证覆盖原图
        首先会判断在h和w方向分别需要分成几块，
        例如，图片h为1080，现在要求分割的patch的h为224，则会分割(1080/224)向上取证，即5块
        会采取这样的策略进行分割：
            第1块的h范围：0*224 -- 1*224-1
            第2块的h范围：1*224 -- 2*224-1
            第3块的h范围：2*224 -- 3*224-1
            第4块的h范围：3*224 -- 4*224-1
            第5块的h范围：1080-224 -- 1080-1
        可以看到，第4块与第5块有重叠
    使用：
        data, label = cover_crop(crop_size, data, label)

    :param crop_size:   要裁剪成的形状，list，eg. [224, 224]
    :param data:        数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    :param label:       one hot标签，tensor4d，eg. (1(bs), 5, 1080, 1920)
    :return:            data: tensor4d，eg. (30(bs * patch_num), 3, 224, 224),
                        label: tensor4d，eg. (30(bs * patch_num), 5, 224, 224)
    """
    h = data.shape[2]
    w = data.shape[3]
    h_num = math.ceil(h / crop_size[0])
    w_num = math.ceil(w / crop_size[1])
    for i in range(h_num):
        for j in range(w_num):
            h0, h1, w0, w1 = i * crop_size[0], (i + 1) * crop_size[0], \
                             j * crop_size[1], (j + 1) * crop_size[1]
            if (i + 1) * crop_size[0] <= h and (j + 1) * crop_size[1] <= w:
                if i == 0 and j == 0:
                    d0, l0 = crop((h0, h1, w0, w1), data, label)
                else:
                    d, l = crop((h0, h1, w0, w1), data, label)
                    d0 = torch.cat([d0, d], dim=0)
                    if label is not None:
                        l0 = torch.cat([l0, l], dim=0)
            else:
                if (i + 1) * crop_size[0] > h:
                    h1 = h
                    h0 = h - crop_size[0]
                if (j + 1) * crop_size[1] > w:
                    w1 = w
                    w0 = w - crop_size[1]
                d, l = crop((h0, h1, w0, w1), data, label)
                d0 = torch.cat([d0, d], dim=0)
                if label is not None:
                    l0 = torch.cat([l0, l], dim=0)
    return d0, l0


def recover_from_crop(crops, data_size):
    """
    从裁剪后的patches恢复出一张完整的图片

    思路：
        与cover_crop的分割策略相同，只是这里会进行其逆过程，将小块拼接起来
        对于有重叠的部分，会取不同小块重叠部分的均值
    使用：
        full = recover_from_crop(crops, data_size)

    :param crops:       patches，tensor4d，eg. (30(bs * patch_num), n_channels, 224, 224),
    :param data_size:   完整数据的size，list/turple，eg，[_, _, h, w]
    :return:            完整数据，tensor4d，eg. (1(bs), n_channels, 1080, 1920)
    """
    h = data_size[2]
    w = data_size[3]
    c_h = crops.shape[2]
    c_w = crops.shape[3]
    h_num = math.ceil(h / c_h)
    w_num = math.ceil(w / c_w)
    if h_num * w_num != crops.shape[0]:
        print('<crop>: The number of patches is wrong .')
        return
    recovered = torch.zeros((1, crops.shape[1], data_size[2], data_size[3])).to(crops.device)
    number = torch.zeros((1, crops.shape[1], data_size[2], data_size[3])).to(crops.device)
    for i in range(h_num):
        for j in range(w_num):
            h0, h1, w0, w1 = i * c_h, (i + 1) * c_h, j * c_w, (j + 1) * c_w
            if not ((i + 1) * c_h <= h and (j + 1) * c_w <= w):
                if (i + 1) * c_h > h:
                    h1 = h
                    h0 = h - c_h
                if (j + 1) * c_w > w:
                    w1 = w
                    w0 = w - c_w
            recovered[:, :, h0:h1, w0:w1] += crops[i * w_num + j, :, :, :].unsqueeze(dim=0)
            number[:, :, h0:h1, w0:w1] += 1
    recovered /= number
    return recovered


if __name__ == '__main__':
    print(1)
