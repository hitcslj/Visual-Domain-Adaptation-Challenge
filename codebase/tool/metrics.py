from utils.averager import AverageMeter


class Metrics:
    """
    评价指标，主要包括mIoU和Acc

    使用：
        metrics = Metrics()
        metrics.update(prediction, gt)
        logger.info(str(metrics))

    """
    def __init__(self, n_class=5, class_names=None):
        self.n_class = n_class
        self.class_names = (class_names if class_names is not None else
                            ['background', 'rigid_plastic', 'cardboard', 'metal', 'soft_plastic'])
        self.iou_average_meter = []
        self.acc_average_meter = []
        self.pred_average_meter = []
        self.freq_average_meter = []
        self.reset()

    def reset(self):
        """
        重置
        :return:    无
        """
        if len(self.iou_average_meter) == 0:
            for i in range(self.n_class):
                self.iou_average_meter.append(AverageMeter('IoU', ':.4f'))
                self.acc_average_meter.append(AverageMeter('Acc', ':.4f'))
                self.pred_average_meter.append(AverageMeter('Pred', ':.4f'))
                self.freq_average_meter.append(AverageMeter('Freq', ':.4f'))
        else:
            for i in range(self.n_class):
                self.iou_average_meter[i].reset()
                self.acc_average_meter[i].reset()
                self.pred_average_meter[i].reset()
                self.freq_average_meter[i].reset()

    def update(self, prediction, gt):
        """
        更新
        :param prediction:  预测分割
        :param gt:          真实分割
        :return:            无
        """
        for i in range(self.n_class):
            mask1 = (prediction == i)
            mask2 = (gt == i)
            num1 = (mask1 & mask2).sum()
            num2 = (mask1 | mask2).sum()
            num_all = gt.shape[0] * gt.shape[1] * gt.shape[2]
            self.iou_average_meter[i].update(num1 / num2, num2)
            self.acc_average_meter[i].update(num1 / mask2.sum(), mask2.sum())
            self.pred_average_meter[i].update(mask1.sum() / num_all, num_all)
            self.freq_average_meter[i].update(mask2.sum() / num_all, num_all)

    def m_iou(self):
        """
        计算mIoU
        :return:    mIoU数值
        """
        m_iou = AverageMeter('mIoU', ':.4f')
        for i in range(self.n_class):
            m_iou.update(self.iou_average_meter[i].average)
        return m_iou.average

    def __str__(self):
        """
        转化为字符串，方便记录
        :return:    字符串结果
        """
        string = ''
        m_iou = AverageMeter('mIoU', ':.4f')
        m_acc = AverageMeter('mAcc', ':.4f')
        for i in range(self.n_class):
            m_iou.update(self.iou_average_meter[i].average)
            m_acc.update(self.acc_average_meter[i].average)
            string += (str(self.iou_average_meter[i]) + ', ' + str(self.acc_average_meter[i])
                       + ', ' + str(self.pred_average_meter[i]) + ', ' + str(self.freq_average_meter[i])
                       + ' -- ' + self.class_names[i] + '\n')
        string += (str(m_iou) + ', ' + str(m_acc))
        return string
    
    def write_to_tensorboard(self, writer, step, mark=''):
        """  
        写入tensorboard
        :return:    无
        """
        m_iou = AverageMeter('mIoU', ':.4f')
        m_acc = AverageMeter('mAcc', ':.4f')
        for i in range(self.n_class):
            m_iou.update(self.iou_average_meter[i].average)
            m_acc.update(self.acc_average_meter[i].average)
            writer.add_scalar(mark + self.class_names[i] + '/' + 'iou_averaged', self.iou_average_meter[i].average, global_step = step)
            writer.add_scalar(mark + self.class_names[i] + '/' + 'acc_averaged', self.acc_average_meter[i].average, global_step = step)
            writer.add_scalar(mark + self.class_names[i] + '/' + 'pred_averaged', self.pred_average_meter[i].average, global_step = step)
            writer.add_scalar(mark + self.class_names[i] + '/' + 'freq_averaged', self.freq_average_meter[i].average, global_step = step)
        writer.add_scalar(mark + 'all_classes' + '/' + 'iou_averaged', m_iou.average, global_step = step)
        writer.add_scalar(mark + 'all_classes' + '/' + 'acc_averaged', m_acc.average, global_step = step)
        return
    
