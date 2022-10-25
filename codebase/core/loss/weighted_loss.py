

def weighted_loss(prediction, label, gt, criterion, weight):
    """
    各个类别不同权重的损失函数

    思路：
        一个一个类算loss，乘以对应系数，然后相加
    说明：
        1.可能存在bug
        2.未必支持bs>1
        3.目前项目中gt大多数已经弃用，通常使用one hot label

    :param prediction:  预测类别，tensor4d， eg. (1(bs), 5, 1080, 1920)
    :param label:       真实类别，tensor4d， eg. (1(bs), 5, 1080, 1920)
    :param gt:          真实类别，tensor3d， eg. (1(bs), 1080, 1920)
    :param criterion:   原本的损失函数
    :param weight:      各类别的权重，数组
    :return:            loss值
    """
    loss_sum = 0
    for i in range(len(weight)):
        mask = (gt == i)
        if mask.sum() > 0:
            loss = criterion(prediction.permute(1, 0, 2, 3)[:, mask], label.permute(1, 0, 2, 3)[:, mask])
            loss_sum = loss_sum + weight[i] * loss
    return loss_sum
