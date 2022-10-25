from torchvision import transforms
import torch


def denormalize(x, mean, std):
    """
    去归一化

    思路：
        简单地乘标准差加平均值

    :param x:       数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    :param mean:    平均值，list or tensor，eg. [128, 127, 126]
    :param std:     标准差，list or tensor，eg. [128, 127, 126]
    :return:        去归一化后的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    """
    mean = torch.tensor(mean).reshape(1, 3, 1, 1).to(x.device)
    std = torch.tensor(std).reshape(1, 3, 1, 1).to(x.device)
    x = x * std + mean
    return x


def tensor_to_image(data, gt=None, de_norm=True):
    """
    tensor还原为PIL图片数据
    :param data:        数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    :param gt:          标注，tensor3d or None，eg. (1(bs), 1080, 1920)
    :param de_norm:     是否要去归一化，bool
    :return:            image: list(PIL.Image),
                        (gt: list(PIL.Image))
    """
    image = []
    image_gt = []
    if de_norm:
        data = denormalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    de_transform = transforms.Compose([transforms.ToPILImage()])
    if gt is not None:
        for i in range(data.shape[0]):
            image.append(de_transform(data[i, :, :, :]))
            image_gt.append(de_transform(gt[i, :, :].type(torch.uint8)))
        return image, image_gt
    else:
        for i in range(data.shape[0]):
            image.append(de_transform(data[i, :, :, :]))
        return image
