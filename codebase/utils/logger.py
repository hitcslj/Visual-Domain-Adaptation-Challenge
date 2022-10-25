import logging
import os
import time


def get_logger(work_dir):
    """
    构造一个logger
    :param work_dir:        logger记录的路径
    :return:                logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s]  %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    if not logger.handlers:
        s_handler = logging.StreamHandler()
        s_handler.setFormatter(formatter)
        logger.addHandler(s_handler)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        file_name = 'log_' + time.strftime("%m_%d_%H_%M", time.localtime()) + '.txt'
        f_handler = logging.FileHandler(os.path.join(work_dir, file_name), mode='w')
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
    return logger


def remove_formatter(logger):
    """
    去除格式
    :param logger:  logger
    :return:        logger
    """
    formatter = logging.Formatter(fmt="%(message)s")
    for i in range(len(logger.handlers)):
        logger.handlers[i].setFormatter(formatter)
    return logger


def add_formatter(logger):
    """
    添加格式
    :param logger:  logger
    :return:        logger
    """
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s]  %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    for i in range(len(logger.handlers)):
        logger.handlers[i].setFormatter(formatter)
    return logger

