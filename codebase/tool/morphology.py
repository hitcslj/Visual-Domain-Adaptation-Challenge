from utils.utils import label_to_one_hot
from utils.convert import visualize_gt
from copy import deepcopy
from PIL import Image
import numpy as np
import torch
import cv2

cv2.setNumThreads(0)

class Morphology:
    """
    形态学操作

    使用：
        morphology = Morphology(k_object, k_back, kernel_shape)
        outputs = morphology.morphology_process(outputs)
        results = outputs.argmax(dim=1)
        results = morphology.seg_fill(results)

    """

    def __init__(self, k_object=0.03, k_back=0.01, kernel_shape='circle'):
        self.kernel_object = None
        self.kernel_back = None
        self.k_object = k_object
        self.k_back = k_back
        self.kernel_shape = kernel_shape
        self.image_shape = None

    def morphology_process(self, x):
        """
        形态学处理

        思路：
            主要使用了灰度开运算和灰度闭运算
            用于分割概率的后处理，能够使得分割更加连续

        :param x:               各个像素属于各类别的预测概率，tensor4d，eg. (1(bs), 5, 1080, 1920)
        :return:                处理后的概率
        """
        after_softmax = (x.sum(dim=1) == 1).sum() == x.shape[2] * x.shape[3]
        if not after_softmax:
            x = x.softmax(dim=1)
        if self.image_shape != x.shape:
            self.image_shape = x.shape
            self.kernel_object = self.get_kernel(int(self.k_object * (x.shape[2] * x.shape[3]) ** 0.5),
                                                 shape=self.kernel_shape)
            self.kernel_back = self.get_kernel(int(self.k_back * (x.shape[2] * x.shape[3]) ** 0.5),
                                               shape=self.kernel_shape)
        y = x.cpu().numpy()
        y = self.object_close(y, kernel=self.kernel_object)
        y = self.object_open(y, kernel=self.kernel_object)
        y = self.background_open(y, kernel=self.kernel_back)
        y = self.background_close(y, kernel=self.kernel_back)
        y = torch.tensor(y).to(x.device)
        return y

    @staticmethod
    def object_close(x, kernel):
        """
        对物体类别做灰度闭运算

        使用：
            在morphology_process函数中调用，通常不在外部调用

        :param x:       预测概率，np.array4d，eg. (1(bs), 5, 1080, 1920)
        :param kernel:  核，np.array2d，eg. (11, 11)
        :return:        处理后的结果，np.array4d，eg. (1(bs), 5, 1080, 1920)
        """
        for i in range(x.shape[0]):
            x[i, 1, :, :] = cv2.morphologyEx((x[i, 1, :, :] * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel) / 255
            x[i, 2, :, :] = cv2.morphologyEx((x[i, 2, :, :] * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel) / 255
            x[i, 3, :, :] = cv2.morphologyEx((x[i, 3, :, :] * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel) / 255
            x[i, 4, :, :] = cv2.morphologyEx((x[i, 4, :, :] * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel) / 255
        return x

    @staticmethod
    def object_open(x, kernel):
        """
        对物体类别做灰度开运算

        使用：
            在morphology_process函数中调用，通常不在外部调用

        :param x:       预测概率，np.array4d，eg. (1(bs), 5, 1080, 1920)
        :param kernel:  核，np.array2d，eg. (11, 11)
        :return:        处理后的结果，np.array4d，eg. (1(bs), 5, 1080, 1920)
        """
        for i in range(x.shape[0]):
            x[i, 1, :, :] = cv2.morphologyEx((x[i, 1, :, :] * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel) / 255
            x[i, 2, :, :] = cv2.morphologyEx((x[i, 2, :, :] * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel) / 255
            x[i, 3, :, :] = cv2.morphologyEx((x[i, 3, :, :] * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel) / 255
            x[i, 4, :, :] = cv2.morphologyEx((x[i, 4, :, :] * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel) / 255
        return x

    @staticmethod
    def background_close(x, kernel):
        """
        对背景做灰度闭运算

        使用：
            在morphology_process函数中调用，通常不在外部调用

        :param x:       预测概率，np.array4d，eg. (1(bs), 5, 1080, 1920)
        :param kernel:  核，np.array2d，eg. (11, 11)
        :return:        处理后的结果，np.array4d，eg. (1(bs), 5, 1080, 1920)
        """
        for i in range(x.shape[0]):
            x[i, 0, :, :] = cv2.morphologyEx((x[i, 0, :, :] * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel) / 255
        return x

    @staticmethod
    def background_open(x, kernel):
        """
        对背景做灰度开运算

        使用：
            在morphology_process函数中调用，通常不在外部调用

        :param x:       预测概率，np.array4d，eg. (1(bs), 5, 1080, 1920)
        :param kernel:  核，np.array2d，eg. (11, 11)
        :return:        处理后的结果，np.array4d，eg. (1(bs), 5, 1080, 1920)
        """
        for i in range(x.shape[0]):
            x[i, 0, :, :] = cv2.morphologyEx((x[i, 0, :, :] * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel) / 255
        return x

    @staticmethod
    def get_kernel(size, shape='circle'):
        """
        构造一个核

        使用：
            在morphology_process函数中调用，通常不在外部调用

        :param size:    核的大小，int
        :param shape:   核的形状，目前实现了圆形核与方形核，str，['circle', 'square']
        :return:        核
        """
        assert shape in ['circle', 'square']
        if shape == 'circle':
            kernel = np.zeros((size, size), np.uint8)
            r = (size - 1) / 2
            for i in range(size):
                for j in range(size):
                    if (i - r) ** 2 + (j - r) ** 2 <= r ** 2:
                        kernel[i, j] = 1
            return kernel
        elif shape == 'square':
            kernel = np.ones((size, size), np.uint8)
            return kernel

    @staticmethod
    def image_fill(image):
        """
        填充二值图片中的空洞

        使用：
            在seg_fill函数中调用，通常不在外部调用

        :param image:   二值图片，np.array2d，eg，[]1080, 1920]
        :return:        处理结果，np.array2d，eg，[]1080, 1920]
        """
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len_contour = len(contours)
        contour_list = []
        for i in range(len_contour):
            drawing = np.zeros_like(image, np.uint8)
            img_contour = cv2.drawContours(drawing, contours, i, 1, -1)
            contour_list.append(img_contour)
        result = sum(contour_list)
        if len_contour > 0:
            result = (result > 0).astype(np.int32)
        return result

    @staticmethod
    def seg_fill(seg):
        """
        填充分割结果中的空洞

        思路：
            将属于各类垃圾的mask分别取出，填充空洞
            填充空洞后，对于不属于任何垃圾类别的像素，将其类别设为背景
            对于同时属于两种及更多类别的像素，保持其操作之前的类别不变

        :param seg:     分割结果，tensor3d，eg. (1(bs), 1080, 1920)，
                            or  tensor4d，eg. (1(bs), 3, 1080, 1920)
        :return:        处理后的分割结果，tensor3d，eg. (1(bs), 1080, 1920)
        """
        if len(seg.shape) == 2:
            seg = seg.unsqueeze(dim=0)
            seg = label_to_one_hot(seg)
        elif len(seg.shape) == 3:
            seg = label_to_one_hot(seg)
        result = seg.argmax(dim=1).cpu().numpy()
        to_fill = seg.cpu().numpy()
        for i in range(seg.shape[0]):
            for j in range(1, seg.shape[1]):
                to_fill[i, j, :, :] = Morphology.image_fill(to_fill[i, j, :, :].astype(np.uint8))
            to_fill[i, 0, :, :] = 0
            temp = to_fill[i, :, :, :].sum(axis=0)
            mask1 = temp > 1
            mask2 = temp < 1
            result[i, ~mask1] = np.argmax(to_fill[i, :, :, :], axis=0)[~mask1]
            result[i, mask2] = 0
        result = torch.tensor(result).to(seg.device).type(seg.dtype)
        return result


if __name__ == '__main__':
    process = Morphology(k_object=0.0375, k_back=0.015, kernel_shape='circle')
    prob = torch.load('../data/01_frame_054100.PNG.pt')
    pred0 = prob.argmax(dim=0).cpu().numpy()
    prob1 = process.morphology_process(prob.unsqueeze(dim=0)).squeeze(dim=0)
    pred1 = prob1.argmax(dim=0)
    # pred1 = process.seg_fill(pred1).squeeze(dim=0)
    pred1 = pred1.cpu().numpy()
    gt = Image.open('../data/01_frame_054100.PNG')
    visualize_gt(np.array(gt), is_tensor=False)
    visualize_gt(pred0, is_tensor=False)
    visualize_gt(pred1, is_tensor=False)

