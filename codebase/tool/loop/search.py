from utils.parser import *
from utils.logger import *
import os


def search(function, args):
    """
    0.618法搜索最优参数
    主要用于搜索与训练过程无关的超参数
    :param function:    待调用的函数
    :param args:        参数
    :return:            无
    """
    work_dir = os.path.join(os.path.join(args.output_dir, 'exps'), args_to_str(args))
    logger = get_logger(work_dir)
    delta = args.search_delta
    k = args.search_start
    arrange = args.search_arrange
    end_value = args.end_threshold
    k_list = []
    f_list = []
    if arrange is None:
        logger.info('首先确定初始搜索区间:\n')
        logger.info('- - - - - - - - - - - - - - - - - -')
        logger.info('{} = {}, 开始验证'.format(args.search_target, k))
        logger.info('- - - - - - - - - - - - - - - - - -')
        k_list.append(k)
        args.__dict__[args.search_target] = k
        f_list.append(function(args))
        while True:
            k = k + delta
            logger.info('- - - - - - - - - - - - - - - - - -')
            logger.info('{} = {}, 开始验证'.format(args.search_target, k))
            logger.info('- - - - - - - - - - - - - - - - - -')
            k_list.append(k)
            args.__dict__[args.search_target] = k
            f_list.append(function(args))
            if f_list[-1] < f_list[-2]:
                break
        a = k_list[-3]
        b = k_list[-1]
    else:
        a = arrange[0]
        b = arrange[1]

    logger.info('使用0.618法开始搜索:\n')
    logger.info('在区间{}~{}内进行搜索'.format(a, b))
    k1 = b - 0.618 * (b - a)
    k2 = a + 0.618 * (b - a)
    k = k1
    logger.info('- - - - - - - - - - - - - - - - - -')
    logger.info('{} = {}, 开始验证'.format(args.search_target, k))
    logger.info('- - - - - - - - - - - - - - - - - -')
    args.__dict__[args.search_target] = k
    k1_value = function(args)
    k_list.append(k)
    f_list.append(k1_value)
    k = k2
    logger.info('- - - - - - - - - - - - - - - - - -')
    logger.info('{} = {}, 开始验证'.format(args.search_target, k))
    logger.info('- - - - - - - - - - - - - - - - - -')
    args.__dict__[args.search_target] = k
    k2_value = function(args)
    k_list.append(k)
    f_list.append(k2_value)
    while True:
        if b - a < end_value:
            break
        if k1_value > k2_value:
            b = k2
            k2 = k1
            k2_value = k1_value
            k1 = b - 0.618 * (b - a)
            k = k1
            logger.info('- - - - - - - - - - - - - - - - - -')
            logger.info('{} = {}, 开始验证'.format(args.search_target, k))
            logger.info('- - - - - - - - - - - - - - - - - -')
            args.__dict__[args.search_target] = k
            k1_value = function(args)
            k_list.append(k)
            f_list.append(k1_value)
        else:
            a = k1
            k1 = k2
            k1_value = k2_value
            k2 = a + 0.618 * (b - a)
            k = k2
            logger.info('- - - - - - - - - - - - - - - - - -')
            logger.info('{} = {}, 开始验证'.format(args.search_target, k))
            logger.info('- - - - - - - - - - - - - - - - - -')
            args.__dict__[args.search_target] = k
            k2_value = function(args)
            k_list.append(k)
            f_list.append(k2_value)
    logger.info('最优值在区间{}~{}内'.format(a, b))
    logger.info('k = {}'.format(k_list))
    logger.info('f = {}'.format(f_list))
