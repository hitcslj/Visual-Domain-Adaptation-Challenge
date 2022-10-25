from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.nn.parallel import DistributedDataParallel
from core.dataset.waste_dataset import WasteDataset
from tool.transform.rand_augment import RandAugment
from tool.transform.seg_augment import SegAugment
from tool.transform.mixup import MixUp
from torch.utils.data import DataLoader
from tool.loop.train import train, train_uda
from tool.loop.search import search
from torchvision import transforms
from tool.loop.validate import validate
from core.model.core_model import CoreModel
from utils.utils import setup_seed
import torch.optim as optim
from tool.loop.test import test
from core.uda.uda import UDA
from utils.parser import *
import torch
import sys
import os


# 本地调试时设为True以便调试
local_debugging = False


def main(args):
    """
    主函数
    :param args:    参数
    :return:        无
    """

    input_output = (not args.distributed or (args.distributed and args.local_rank == 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device(device, args.local_rank)
        print('GPU {} is READY! :P'.format(args.local_rank))
        torch.distributed.barrier() # add if bug happens

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    if args.augment:
        transform.transforms.insert(0, RandAugment(2, 20))
        seg_augment = SegAugment(1, 20)
    else:
        seg_augment = None
    if args.mixup:
        mix_up = MixUp(p=args.mix_p, mode=args.mix_mode, k_max=args.mix_k_max, k_min=args.mix_k_min)
    else:
        mix_up = None

    #--- definition for all datasets ---#
    # for training
    data_set = WasteDataset(root=args.data, transform=transform, seg_augment=seg_augment, mix_up=mix_up,
                            ground_truth=(False if args.order == 'test' else True), balance=args.balance, rank=args.local_rank)
    data_set.get_path()
    if args.balance:
        data_set.balance()
    # print(len(data_set))
    data_loader = DataLoader(data_set, batch_size=args.bs, shuffle=(False if args.distributed else True),
                             sampler=(DistributedSampler(data_set) if args.distributed else None), num_workers=4, 
                             pin_memory=False, drop_last=True)
                             
    if args.order == 'train' or args.order == 'uda':
       
        data_set_source_eval = WasteDataset(root=args.data_val, transform=transform_val, seg_augment=None, mix_up=None,
                                ground_truth=True, balance=False, rank=args.local_rank)
        data_set_source_eval.get_path()
        data_loader_source_eval = DataLoader(data_set_source_eval, batch_size=1, shuffle=False,
                                sampler=None)

        
        data_set_target_eval = WasteDataset(root=args.data_target_val, transform=transform_val, seg_augment=None, mix_up=None,
                                ground_truth=True, balance=False, rank=args.local_rank)
        data_set_target_eval.get_path()
        data_loader_target_eval = DataLoader(data_set_target_eval, batch_size=1, shuffle=False,
                                sampler=None)


    model = UDA(core_model=CoreModel(device=device, args=args).get(), args=args).to(device)
    if args.model_path != '' and input_output:
        model.load_state_dict(torch.load(args.model_path))
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.order == 'train':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        train(model, data_loader, data_loader_source_eval, data_loader_target_eval, 
              device, optimizer, args, log=input_output)  
    elif args.order == 'uda':
        target_set = WasteDataset(root=args.data_target, transform=transform, ground_truth=False,
                                  seg_augment=seg_augment, rank=args.local_rank, target_set=True)
        target_set.get_path()
        target_loader = DataLoader(target_set, batch_size=args.bs, shuffle=(False if args.distributed else True),
                                   sampler=(DistributedSampler(target_set) if args.distributed else None), num_workers=6,
                                   pin_memory=False, drop_last=True)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        train_uda(model, data_loader, target_loader, data_loader_source_eval, data_loader_target_eval,
                  device, optimizer, args, log=input_output)  
    elif args.order == 'test':
        test(model, data_loader, device, args)
    elif args.order == 'val':
        return validate(model, data_loader, device, args)


if __name__ == "__main__":

    setup_seed(666)

    args = parse_args(sys.argv[1:])
    # print(sys.argv[1:])
    if local_debugging:
        args.order = 'uda'
        args.model = 'fcn'
        # args.model_path = './weight.pth'
        args.data = './data/source'
        args.data_target = './data/target'
        args.crop = True
        args.crop_size = [32, 32]
        args.resize = True
        args.augment = True
        args.mixup = True
        args.mix_mode = 'object'
        args.mix_p = 1.0
        args.mix_k_min = 1.0
        args.mix_k_max = 1.0
        args.resize_ratio = 0.5
        args.drop = 0.2
        args.output = True
        args.morphology = False
        args.rand_resize = True
        args.output_dir = './'
        args.uda_version = 7

    if not args.search:
        main(args)
    else:
        search(main, args)
