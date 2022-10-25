from core.dataset.waste_dataset import WasteDataset
from torch.utils.data import DataLoader
from utils.averager import AverageMeter
from tqdm import tqdm
import sys


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("请指定数据集路径!")
    else:
        count_pixel = [0, 0, 0, 0, 0]
        count_img = [0, 0, 0, 0, 0]
        data_path = sys.argv[1]
        data_set = WasteDataset(data_path)
        data_loader = DataLoader(data_set, batch_size=1, shuffle=True)
        bar = tqdm(range(len(data_loader)))
        for i, (inputs, labels, gt, name) in enumerate(data_loader):
            for j in range(5):
                temp = (gt == j).sum()
                count_pixel[j] += temp
                if temp > 0:
                    count_img[j] += 1
            bar.update(1)
        print(count_pixel)
        print(count_img)
