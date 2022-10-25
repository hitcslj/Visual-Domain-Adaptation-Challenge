from core.loss.consistent_loss import LocalConsistentLoss
from core.loss.weighted_loss import weighted_loss
from tool.transform.crop import random_crop
from tool.transform.resize import resize
from utils.convert import visualize_gt, visualize_data
from utils.averager import AverageMeter
from core.uda.uda_tool import *
from copy import deepcopy
import torch.nn as nn
import torch


class UDA(nn.Module):
    """
    Unsupervised Domain Adaption
    使用CoreModel作为成员变量，对外统一的UDA训练接口
    """
    def __init__(self, core_model, args, n_class=5, k_consistent=0.1, weight=None):
        """
        初始化函数
        :param core_model:      骨架模型，nn.module
        :param args:            参数
        :param n_class:         类别数，int
        :param k_consistent:    一致性loss的系数，float，目前暂时未使用
        :param weight:          类别加权loss的权值，list(float)，目前暂未使用
        """
        super().__init__()
        self.core_model = core_model
        self.teacher_model = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.consistent_criterion = LocalConsistentLoss(in_channels=n_class)
        self.n_class = n_class
        self.k_consistent = k_consistent
        self.weight = weight if weight is not None else [1.0, 3.40, 1.06, 12.40, 1.37]
        self.args = args
        self.log = ''
        self.pseudo_label_threshold = 0.0
        self.target_percent = 0.9
        self.percent = AverageMeter('pseudo label percent', ':.2f')

    def forward(self, x):
        """
        forward函数
        :param x:   data，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :return:    output，tensor4d，eg. (1(bs), 5, 1080, 1920)
        """
        out = self.core_model(x)
        return out

    def gt_loss_backward(self, data, label, k=1.0):
        """
        使用source domain的数据和对应的ground truth进行监督训练

        思路：
            使用有标注的数据对模型进行监督训练，可以在此处加入一致性loss、加权loss等不同的损失函数
        使用：
            optimizer.zero_grad()
            uda_model.gt_loss_backward(data, label, k=k)
            optimizer.step()
        说明：
            k会作为一个超参数，人为乘在原本的loss上，但是函数返回的loss值还是原本的数值

        :param data:    source domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param label:   标注，one hot向量，tensor4d，eg. (1(bs), 5, 1080, 1920)
        :param k:       loss放缩系数，float
        :return:        loss: torch.tensor,
                        output: tensor4d，eg. (1(bs), 5, 1080, 1920)
        """
        out = self.forward(data) 
        # consistent_loss = self.consistent_criterion(out, gt) * self.k_consistent
        loss = self.criterion(out, label)
        original_loss = float(loss)
        loss = loss * k
        loss.backward()
        return original_loss, out

    def uda_loss_backward(self, s_data, s_label, t_data, iter_count=0, bs=4):
        """
        使用target domain的数据对预训练的模型进行无监督的域迁移训练

        思路：
            本函数是多个版本UDA训练函数的统一接口，各个版本的具体思路会各自介绍
        使用：
            optimizer.zero_grad()
            uda_model.uda_loss_backward(s_data, s_label, t_data, iter_count)
            optimizer.step()

        :param s_data:      source domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param s_label:     source domain的数据标签，tensor4d，eg. (1(bs), 5, 1080, 1920)
        :param t_data:      target domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param iter_count:  当前的iter数，int
        :return:            loss: torch.tensor
        """
        uda_method = {0: self.uda0, 2: self.uda2, 5: self.uda5, 6: self.uda6, 7: self.uda7}
        method = uda_method[self.args.uda_version]
        return method(s_data, s_label, t_data, iter_count, bs=bs)

    def uda0(self, s_data, s_label, t_data, iter_count=0):
        """
        uda train v0

        思路：
            t_data过teacher model，得到pseudo label
            将s_data, s_label, t_data, pseudo label通过replace mix up的方法混合在一起
            然后对于混合数据做较强的数据增强
            使用增强后的混合数据训练student model，即core_model
        说明：
            1.uda训练基础版本
            2.由于历史缘故，命名没有修改，目前保留的uda方法版本不连续，下同

        :param s_data:      source domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param s_label:     source domain的数据标签，tensor4d，eg. (1(bs), 5, 1080, 1920)
        :param t_data:      target domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param iter_count:  当前的iter数，int
        :return:            loss: torch.tensor
        """
        self.teacher_model.eval()
        with torch.no_grad():
            prob = self.teacher_model(t_data)
        data, label = source_target_mix_up(s_data, s_label.argmax(dim=1), t_data, prob.argmax(dim=1))
        loss = self.criterion(self.forward(data), label)
        loss.backward()
        return loss

    def uda2(self, s_data, s_label, t_data, iter_count=0):
        """
        uda train v2

        思路：
            与v0的区别是，加入了颜色匹配，将target data的颜色匹配为source data的颜色，
            相当于对target data额外做了一个数据增强

        :param s_data:      source domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param s_label:     source domain的数据标签，tensor4d，eg. (1(bs), 5, 1080, 1920)
        :param t_data:      target domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param iter_count:  当前的iter数，int
        :return:            loss: torch.tensor
        """
        self.teacher_model.eval()
        with torch.no_grad():
            prob = self.teacher_model(t_data)
            t_data = source_target_color_transfer(t_data, s_data)
        data, label = source_target_mix_up(s_data, s_label.argmax(dim=1), t_data, prob.argmax(dim=1))
        loss = self.criterion(self.forward(data), label)
        loss.backward()
        return loss

    def uda5(self, s_data, s_label, t_data, iter_count=0):
        """
        uda train v5

        思路：
            与v2的区别是，mix up的模式由replace改为了object

        :param s_data:      source domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param s_label:     source domain的数据标签，tensor4d，eg. (1(bs), 5, 1080, 1920)
        :param t_data:      target domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param iter_count:  当前的iter数，int
        :return:            loss: torch.tensor
        """
        self.teacher_model.eval()
        with torch.no_grad():
            prob = self.teacher_model(t_data)
            t_data = source_target_color_transfer(t_data, s_data)
        data, label = source_target_mix_up(s_data, s_label.argmax(dim=1), t_data, prob.argmax(dim=1),
                                           mode='object', num=4)
        loss = self.criterion(self.forward(data), label)
        loss.backward()
        return loss

    def uda6(self, s_data, s_label, t_data, iter_count=0, bs = 1):
        """
        uda train v6

        思路：
            与v2的区别是，加入了过滤低置信度pseudo label的mask

        :param s_data:      source domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param s_label:     source domain的数据标签，tensor4d，eg. (1(bs), 5, 1080, 1920)
        :param t_data:      target domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param iter_count:  当前的iter数，int
        :return:            loss: torch.tensor
        """
        self.teacher_model.eval()
        with torch.no_grad():
            prob = self.teacher_model(t_data)
            t_data = source_target_color_transfer(t_data, s_data)
        mask, pseudo_label = prob.max(dim=1)
        mask = mask > 0.2
        percent = float(mask.sum() / (mask.shape[0] * mask.shape[1] * mask.shape[2]))
        self.percent.update(percent * 100)
        pseudo_label[~mask] = -1

        # data = None
        # label = None
        # # 4shapes: [bs, 3, 352, 352], [bs, 5, 352, 352], [bs, 3, 352, 352], [bs, 352, 352]
        # for i in range(bs):
        #     data_tmp, label_tmp = source_target_mix_up(s_data[i].unsqueeze(0), s_label.argmax(dim=1)[i].unsqueeze(0), t_data[i].unsqueeze(0), pseudo_label[i].unsqueeze(0)) 
        #     if data == None:
        #         data = data_tmp
        #         label = label_tmp
        #     else:
        #         data = torch.cat([data,data_tmp], dim=0)
        #         label = torch.cat([label,label_tmp], dim=0)
        # # data: [bs, 3, 352, 352], label: [bs, 5, 352, 352]
        # output = self.forward(data) # [bs, 5, 352, 352]
        # loss = self.criterion(output.permute(1,0,2,3)[:, mask], label.permute(1,0,2,3)[:, mask])
        data, label = source_target_mix_up(s_data, s_label.argmax(dim=1), t_data, pseudo_label)
        output = self.forward(data)
        loss = self.criterion(output.unsqueeze(dim=2)[:, :, mask], label.unsqueeze(dim=2)[:, :, mask])
        loss.backward()
        return loss

    def uda7(self, s_data, s_label, t_data, iter_count=0):
        """
        uda train v7

        思路：
            与v6的区别是，加入了过滤低置信度pseudo label的mask的获取方法不同
            并且加入了P控制

        :param s_data:      source domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param s_label:     source domain的数据标签，tensor4d，eg. (1(bs), 5, 1080, 1920)
        :param t_data:      target domain的数据，tensor4d，eg. (1(bs), 3, 1080, 1920)
        :param iter_count:  当前的iter数，int
        :return:            loss: torch.tensor
        """
        self.teacher_model.eval()
        with torch.no_grad():
            prob = self.teacher_model(t_data)
            t_data = source_target_color_transfer(t_data, s_data)
        max_prob, pseudo_label = prob.max(dim=1)
        entropy = -(prob.softmax(dim=1) * torch.log(prob.softmax(dim=1))).sum(dim=1)
        mask = (max_prob - entropy) > self.pseudo_label_threshold
        percent = float(mask.sum() / (mask.shape[0] * mask.shape[1] * mask.shape[2]))
        self.percent.update(percent * 100)
        self.pseudo_label_threshold += 0.01 * (percent - self.target_percent)
        pseudo_label[~mask] = -1
        data, label = source_target_mix_up(s_data, s_label.argmax(dim=1), t_data, pseudo_label)
        output = self.forward(data)
        loss = self.criterion(output.unsqueeze(dim=2)[:, :, mask], label.unsqueeze(dim=2)[:, :, mask])
        loss.backward()
        return loss

    def init_teacher(self):
        """
        初始化tracher model

        思路：
            使用core_model的参数来初始化teacher model，并设置teacher model无梯度
        使用：
            在UDA训练之前，首先应
            uda_model.init_teacher()

        :return:    无
        """
        self.teacher_model = deepcopy(self.core_model)
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

    def update_teacher(self):
        """
        使用指数平滑更新teacher model

        思路：
            指数平滑更新teacher model的权重，保持率为alpha_teacher
        使用：
            每训练一个iter，进行
            uda_model.update_teacher()
            当然，之后也可以尝试训练若干iter之后，再进行一次更新

        :return:    无
        """
        alpha_teacher = 0.999
        for ema_param, param in zip(self.teacher_model.parameters(),
                                    self.core_model.parameters()):
            if not param.data.shape:
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def remove_teacher(self):
        """
        删除teacher model
        :return:    无
        """
        del self.teacher_model
        self.teacher_model = None

    def uda_log(self):
        """
        UDA训练日志
        用于输出UDA内部我们想要记录的一些东西
        会在UDA训练函数中调用
        :return:    log
        """
        log = ''
        if self.args.uda_version == 6:
            log = '{}%'.format(str(self.percent))
            self.percent.reset()
        elif self.args.uda_version == 7:
            log = '{}%, threshold: {}'.format(str(self.percent), round(self.pseudo_label_threshold, 4))
            self.percent.reset()
        return log

