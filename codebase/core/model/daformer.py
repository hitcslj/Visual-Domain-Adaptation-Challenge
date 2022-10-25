from mmseg.models import build_segmentor
from copy import deepcopy
import torch.nn as nn
import mmcv


def build_daformer(cfg_path):
    """
    DAFormer使用的是mmcv、mmseg的框架，使用cfg文件构造一个DAFormer模型
    :param cfg_path:    cfg文件的路径，str
    :return:            DAFormer模型，nn.module
    """
    cfg = mmcv.Config.fromfile(cfg_path)
    model = build_segmentor(deepcopy(cfg['model']))
    return model


class DAFormer(nn.Module):
    """
    DAFormer，官方给出的baseline模型
    """
    def __init__(self, cfg_path):
        """
        初始化函数
        :param cfg_path:    cfg文件的路径，str
        """
        super().__init__()
        self.model = build_daformer(cfg_path)

    def forward(self, x):
        """
        forward函数

        说明：
            可能存在bug，还是没有完全搞懂mmseg的接口

        :param x:   data，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :return:    output，tensor4d，eg. (1(bs), 5, 1080, 1920)
        """
        out = self.model.encode_decode(x, [])
        return out


if __name__ == '__main__':
    model = build_daformer('../mmseg/configs/daformer/aformer_mit5.py')
    print(1)
