from utils.utils import label_to_one_hot
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import os
import nori2 as nori
import ujson as json
import cv2
import io

cv2.setNumThreads(0)

class WasteDataset(Dataset):
    """
    垃圾分类数据集
    """
    def __init__(self, root, transform=None, ground_truth=True, num_class=5, balance=False,
                 seg_augment=None, mix_up=None, rank=0, target_set=False):
        """
        初始化函数

        :param root:            数据集根目录，写到包含data和sem_seg的父文件夹，
                                str or list(str)，eg. 'data/source'
        :param transform:       torchvision.transforms
        :param ground_truth:    数据集是否包含gt，bool
        :param num_class:       类别数，int
        :param balance:         是否启用数据均衡，bool
        :param seg_augment:     分割数据增强，自定义类，见tool.transform.seg_augment
        :param mix_up:          数据mix up，自定义类，见tool.transform.mixup
        """
        self.root = root
        self.transform = (transform if transform is not None else transforms.Compose([transforms.ToTensor()]))

        self.data = []
        self.ground_truth = ground_truth
        self.num_class = num_class
        self.seg_augment = seg_augment
        self.mix_up = mix_up
        self.balance_freq = np.array([1, 4, 1, 9, 2, 1]) # [bg, rigid plastic, cardboard, metal, soft plastic]
        self.rank = rank
        self.target_set = target_set
        self.nf = nori.Fetcher()
        # self.balance = balance
        #self.get_path()

    def get_path(self):
        """
        遍历所有图片及标签的路径，以便后续读入数据

        思路：
            遍历所有图片和标签的路径，存储在self.root变量中，数据类别均衡操作在此实现。
            self.root变量是一个二位列表，举例：
                path = self.root[0]
                则此时path = [img_path, gt_path]
        使用：
            在__init__函数中调用，无需外部调用
        说明：
            如果启用数据均衡，这里的for循环会比较耗时

        :return:    无
        """

        for one_root in self.root:
            with nori.smart_open(one_root) as f:
                for data in json.load(f):
                    self.data.append(data) # data:索引
    
    def balance(self):
        
        if self.ground_truth: # 启用数据均衡
            length = len(self.data) # self.data[i] : [2] , [data, gt]
            for i in range(length):
                sem_seg = self.__get_from_nori__(self.data[i][1], gray=True) 
                sem_seg = np.array(sem_seg)
                #sem_seg[sem_seg == 5] = 0 # paper class to bg
                sem_seg = torch.tensor(sem_seg)
                has = (sem_seg.unsqueeze(dim=0) == torch.tensor(list(range(self.num_class + 1))).reshape(-1, 1, 1))
                has = (has.sum(dim=2).sum(dim=1) > 0)
                freq = int(self.balance_freq[has].max())
                for j in range(freq):
                    self.data.append(self.data[i])
        print("final_length_after_balance:", self.__len__())
    
    def pass_it(self, sem_seg):
        sem_seg = np.array(sem_seg)
        #sem_seg[sem_seg == 5] = 0 # paper class to bg
        #sem_seg = torch.tensor(sem_seg)
        has = (np.expand_dims(sem_seg, axis=0) == np.array(list(range(self.num_class + 1))).reshape(-1, 1, 1))
        has = (has.sum(axis=2).sum(axis=1) > 0)
        freq = int(self.balance_freq[has].max())
        if np.random.uniform(0, 1) < 1.0 * freq / 9:
            return False
        else:
            return True

    def __len__(self):
        """
        数据集大小

        使用：
            len(WasteDatasetItem)

        :return:    数据集大小
        """
        return len(self.data) 

    def __get_from_nori__(self, path, gray = False):

        assert isinstance(path, str)
        cv2.setNumThreads(0)
        
        im = cv2.imdecode(np.frombuffer(io.BytesIO(self.nf.get(path)).getbuffer(), np.uint8),
            cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if gray == True:
            im[im == 5] = 0
        im = im[:1080,:] # 有些图是1920x1920的，不知道从哪来的毒瘤数据
        assert im.shape[1] == 1920
        im = Image.fromarray(im if gray else cv2.cvtColor(im,cv2.COLOR_BGR2RGB)) 
        return im

    def __getitem__(self, idx):
        """
        读取数据集中的一组数据

        思路：
            首先读取数据，之后进行augment、mix up等操作
        使用：
            通常无需直接调用，data_loader会调用
        说明：
            注意操作过程中数据类型的转变，刚开始都进来是PIL.Image，
            augment、mix up都是对PIL.Image类型的数据进行的，
            最后经过transform才会变成tensor

        :param idx: 序号，int
        :return:    image: tensor3d, eg. (3, 1080, 1920),
                    label: tensor3d or None, eg. (1, 1080, 1920) or None
                    path: str
        """
        the_path = self.data[idx]
        image = self.__get_from_nori__(the_path[0]).convert("RGB")
        gt, label = None, 0
        if self.ground_truth or self.target_set: # not in test mode
            gt = self.__get_from_nori__(the_path[1], gray = True)
            if self.balance:
                while self.pass_it(gt):
                    idx = np.random.randint(self.__len__())
                    the_path = self.data[idx]
                    image = self.__get_from_nori__(the_path[0]).convert("RGB")
                    gt = self.__get_from_nori__(the_path[1], gray = True)

        else: # in test mode, gt is name
            gt = the_path[1]
        if self.seg_augment is not None:
            image, gt = self.seg_augment(image, gt)
        if self.mix_up is not None: # in training mode
            idx_target = np.random.randint(self.__len__())
            the_path = self.data[idx_target]  
            image_target = self.__get_from_nori__(the_path[0]).convert("RGB")
            gt_target = None
            if self.ground_truth or self.target_set: # not in test mode, need to read img
                gt_target = self.__get_from_nori__(the_path[1], gray = True)
            if self.seg_augment is not None:
                image_target, gt_target = self.seg_augment(image_target, gt_target)
            image, label = self.mix_up(image, gt, image_target, gt_target)
        else:
            if self.ground_truth or self.target_set: # not in test mode
                gt = torch.tensor(np.array(gt))
                label = label_to_one_hot(gt).type(torch.float32)
            else: 
                # in test mode , return file name as the second
                label = gt
        image = self.transform(image)
        return image, label

