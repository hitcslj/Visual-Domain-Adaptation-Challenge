import math


class AverageMeter(object):
    """
    平均器

    使用：
        average = AverageMeter('要计算平均值的量', ':.4f')
        average.update(1)
        average.update(2)
        average.update(float('nan'))
        average.update(float('inf'))
        print(str(average))
        输出：
        要计算平均值的量:1.5000(nan:1 inf:1)

    """
    def __init__(self, name, fmt=':f', show_current_value=False):
        self.name = name
        self.fmt = fmt
        self.show_current_value = show_current_value
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0
        self.nan = 0
        self.inf = 0
        self.reset()

    def reset(self):
        """
        重置
        :return:    无
        """
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0
        self.nan = 0
        self.inf = 0

    def update(self, value, n=1):
        """
        更新
        :param value:   数值
        :param n:       数值对应的次数
        :return:        无
        """
        if math.isnan(value):
            self.nan += n
        elif math.isinf(value):
            self.inf += n
        else:
            self.value = value
            self.sum += value * n
            self.count += n
            self.average = self.sum / self.count

    def __str__(self):
        fmt = '{name}:' + (('{value' + self.fmt + '}-') if self.show_current_value else '')\
              + '{average' + self.fmt + '}' + ('(' if (self.nan > 0 or self.inf > 0) else '') \
              + ('nan:{nan}' if self.nan > 0 else '') + (' ' if (self.nan > 0 and self.inf > 0) else '') \
              + ('inf:{inf}' if self.inf > 0 else '') + (')' if (self.nan > 0 or self.inf > 0) else '')
        return fmt.format(**self.__dict__)


if __name__ == '__main__':
    temp = AverageMeter('try', ':.4g')
    temp.update(float('nan'))
    temp.update(float('inf'))
    temp.update(1.1111111e-10)
    print(str(temp))
