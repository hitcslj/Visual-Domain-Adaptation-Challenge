from core.model.fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from core.model.daformer import DAFormer
from core.model.clipseg import SegCLIP


class CoreModel:
    """
    不同模型的统一接口
    UDA模型的骨架
    """
    def __init__(self, device, args, n_class=5):
        """
        初始化函数
        :param device:  设备
        :param args:    参数
        :param n_class: 类别数
        """
        self.device = device
        self.args = args
        self.model_name = args.model
        self.n_class = n_class

    def get(self):
        """
        对于不同的模型，只需要在此实现其对应的get函数即可

        使用：
            model = CoreModel(device, args).get()
        说明：
            1.DAFormer模型可能存在bug
            2.本地调试用FCN很方便

        :return:    模型，nn.module
        """
        if self.model_name == 'fcn':
            return self.get_fcn()
        elif self.model_name == 'daformer':
            return DAFormer('./mmseg/configs/daformer/daformer_mit5.py')
        elif self.model_name == 'segclip':
            return SegCLIP(train_clip=(not self.args.freeze_clip))
        else:
            return None

    def get_fcn(self):
        """
        FCN的get函数
        :return:    FCN模型，nn.module
        """
        vgg_model = VGGNet(requires_grad=True, remove_fc=True).to(self.device)
        fcn_model = FCNs(pretrained_net=vgg_model, n_class=self.n_class, p=self.args.drop).to(self.device)
        return fcn_model


if __name__ == '__main__':
    print(1)
