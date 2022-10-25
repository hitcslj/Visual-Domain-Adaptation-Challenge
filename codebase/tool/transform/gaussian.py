import torch.nn as nn
import torch
import math


def gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    """
    高斯滤波核

    思路：
        其实就是一个设置好正态权重的卷积核
    使用：
        kernel = gaussian_kernel(kernel_size=kernel_size, sigma=sigma, channels=channels)
        data = kernel(data)
        p.s.  data: tensor4d，eg. (1(bs), 3, 1080, 1920)

    :param kernel_size:     核的大小，int
    :param sigma:           标准差，int
    :param channels:        通道数，int
    :return:                滤波核，nn.Module
    """
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    kernel = 1 / (2 * math.pi * variance) * torch.exp(-torch.sum((xy_grid - mean) ** 2, dim=-1) / (2 * variance))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                groups=channels,
                                bias=False, padding=kernel_size // 2)
    gaussian_filter.weight.data = kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


def gaussian_blur(data, kernel_size=3, sigma=2, channels=3):
    """
    高斯模糊

    使用：
        data = gaussian_blur(data)
    说明：
        使用gaussian_kernel、gaussian_blur都一样

    :param data:            数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    :param kernel_size:     核的大小，int
    :param sigma:           标准差，int
    :param channels:        通道数，int
    :return:                高斯模糊后的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
    """
    the_kernel = gaussian_kernel(kernel_size=kernel_size, sigma=sigma, channels=channels)
    return the_kernel(data)

