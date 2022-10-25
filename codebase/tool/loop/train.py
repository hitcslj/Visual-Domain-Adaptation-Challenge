from numpy import Inf, block
from utils.convert import visualize_data, visualize_gt
from tool.transform.crop import random_crop, cover_crop, recover_from_crop
from tool.transform.resize import resize, rand_resize_ratio
from utils.averager import AverageMeter
from tool.metrics import Metrics
from tool.morphology import Morphology
from utils.parser import *
from utils.utils import label_to_one_hot
from utils.logger import *
from tqdm import tqdm
import torch
import time
import math
import os
import numpy as np
from tool.transform.denormalize import denormalize
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

palette = np.array([[0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0], [255, 255, 255]])


def train(model, data_loader, source_val_loader, target_val_loader, device, optimizer, args, log=True):
    """
    有监督预训练
    :param model:       模型
    :param data_loader: 数据集
    :param device:      设备
    :param optimizer:   优化器
    :param args:        参数
    :param log:         是否输出Log和权重文件
    :return:            无
    """
    epoch = 0
    iter_count = 1
    start = time.time()
    model.train()
    model.to(device)
    metrics = Metrics()
    work_dir = os.path.join(args.output_dir, 'exps' + args.tag) #os.path.join(os.path.join(args.output_dir, 'exps'), args_to_str(args))
    if log:
        writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard_log' + args.tag))
        logger = remove_formatter(get_logger(work_dir))
        for item in args_to_str_list(args):
            logger.info(item)
        logger.info('\nStart Training ...')
        logger = add_formatter(logger)
        logger.info('len(data_loader) = {}'.format(len(data_loader)))
    average_loss = AverageMeter('loss', ':.4f')
    while True:
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        # time_start = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            # time_end = time.time()
            # if args.local_rank == 0:
            #     print("Time for dataloader for a batch:", time_end - time_start)

            lr = adjust_learning_rate(optimizer, iter_count, args.lr, args.iters, int(0.04 * args.iters))
            
            inputs_gpu = inputs.to(device, non_blocking=True)
            labels_gpu = labels.to(device, non_blocking=True)
            
            if args.resize:
                inputs, labels = resize(inputs, labels, ratio=args.resize_ratio)
            dist.barrier()
            for j in range(10):
                if args.crop:
                    inputs, labels = random_crop(args.crop_size, inputs_gpu, labels_gpu)
                optimizer.zero_grad()
                if args.distributed:
                    loss, outputs = model.module.gt_loss_backward(inputs, labels)
                else:
                    loss, outputs = model.gt_loss_backward(inputs, labels)
                optimizer.step()
                average_loss.update(loss)
                metrics.update(outputs.argmax(dim=1), labels.argmax(dim=1))
            if iter_count % args.log_interval == 0 and log:
                used = time.time() - start
                remain = (args.iters - iter_count - 1) / (iter_count + 1) * used
                lr_string = '%.4g' % lr
                logger.info('iter:{}/{}, {}, lr:{}  --  {} mins used, {} mins remain'
                            .format(iter_count, args.iters, average_loss, lr_string, int(used / 60), int(remain / 60)))
                writer.add_scalar('basic_learning_metrics/average_loss', average_loss.average, global_step = iter_count / args.log_interval)
                writer.add_scalar('basic_learning_metrics/learning_rate', lr, global_step = iter_count / args.log_interval)
                average_loss.reset()
            if iter_count % (args.log_interval * 5) == 0 and log:
                
                step = iter_count / (args.log_interval * 5)
                logger = remove_formatter(logger)
                logger.info(str(metrics))
                metrics.write_to_tensorboard(writer, step = step)  
                metrics.reset()
                logger = add_formatter(logger)

                # show inputs of model
                input_tmp = (denormalize(inputs[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])*255)[0].cpu().numpy().astype('uint8')
                label_tmp = palette[labels[0].argmax(0).cpu().numpy()].astype('uint8').transpose(2, 0, 1)
                output_tmp = palette[outputs[0].argmax(0).cpu().numpy()].astype('uint8').transpose(2, 0, 1)
                img_tmp = np.concatenate([input_tmp, label_tmp, output_tmp], axis=2)
                writer.add_image('imgs/data,label,output', img_tmp, global_step = step, dataformats='CHW')

            if iter_count % args.save_interval == 0 and log:
                torch.save(model.module.state_dict() if args.distributed else model.state_dict(),
                           os.path.join(work_dir, 'weight.pth'))                

           
            if iter_count % (args.log_interval * 100) == 0 and log and args.order != 'test':
                eval(model, source_val_loader, device, args, writer, iter_count / (args.log_interval * 100), 'SOURCE_')
                eval(model, target_val_loader, device, args, writer, iter_count / (args.log_interval * 100), 'TARGET_')
        
            iter_count = iter_count + 1
            if iter_count >= args.iters and log:
                logger.info("\nSaving Final Weights.\n")
                torch.save(model.module.state_dict() if args.distributed else model.state_dict(),
                           os.path.join(work_dir, 'weight.pth'))
                logger = remove_formatter(logger)
                logger.info('\nTrain Finished.\n\n')
            if iter_count >= args.iters:
                return
            # time_start = time.time()
        epoch = epoch + 1


def train_uda(model, source_loader, target_loader, source_val_loader, target_val_loader, device, optimizer, args, log=True):
    """
    无监督域迁移训练
    :param model:           模型
    :param source_loader:   source domain数据集
    :param target_loader:   target domain数据集
    :param device:          设备
    :param optimizer:       优化器
    :param args:            参数
    :param log:         是否输出Log和权重文件
    :return:                无
    """
    epoch = 0
    iter_count = 1
    start = time.time()
    model.train()
    model.to(device)
    metrics = Metrics()
    work_dir = os.path.join(args.output_dir, 'exps'+args.tag)#os.path.join(os.path.join(args.output_dir, 'exps'), args_to_str(args))
    if log:
        writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard_log' + args.tag))
        logger = remove_formatter(get_logger(work_dir))
        for item in args_to_str_list(args):
            logger.info(item)
        logger.info('\nStart UDA Training ...')
        logger = add_formatter(logger)
    average_loss_gt = AverageMeter('loss_gt', ':.4f')
    average_loss_uda = AverageMeter('loss_uda', ':.4f')
    if args.distributed:
        model.module.init_teacher()
    else:
        model.init_teacher()
    while True:
        if args.distributed:
            source_loader.sampler.set_epoch(epoch)
            target_loader.sampler.set_epoch(epoch)
            #source_val_loader.sampler.set_epoch(epoch)
            #target_val_loader.sampler.set_epoch(epoch)
        for i, (source, target) in enumerate(zip(source_loader, target_loader)):
            (inputs, labels) = source
            (target_data, _) = target
            lr = adjust_learning_rate(optimizer, iter_count, args.lr, args.iters, int(0.04 * args.iters))
            inputs_gpu = inputs.to(device)
            labels_gpu = labels.to(device)
            target_data_gpu = target_data.to(device)
            assert inputs.shape[0] == target_data.shape[0] # same batch size
            if args.resize:
                inputs, labels = resize(inputs_gpu, labels_gpu, ratio=args.resize_ratio)
                target_data, _ = resize(target_data_gpu, ratio=args.resize_ratio)
            for j in range(10):
                if args.crop:
                    inputs, labels = random_crop(args.crop_size, inputs, labels)
                    target_data, _ = random_crop(args.crop_size, target_data)
                k = (1 - iter_count / args.iters) if args.source_loose else 1.0
                optimizer.zero_grad()
                if args.distributed:
                    loss_gt, outputs = model.module.gt_loss_backward(inputs, labels, k=k)
                else:
                    loss_gt, outputs = model.gt_loss_backward(inputs, labels, k=k)
                optimizer.step()
                optimizer.zero_grad()
                if args.distributed:  
                    loss_uda = model.module.uda_loss_backward(inputs, labels, target_data, bs = args.bs)
                else:
                    loss_uda = model.uda_loss_backward(inputs, labels, target_data, bs = args.bs)
                optimizer.step()
                if args.distributed:
                    model.module.update_teacher()
                else:
                    model.update_teacher()
                if j == 0:
                    average_loss_gt.update(loss_gt)
                    average_loss_uda.update(loss_uda)
                    metrics.update(outputs.argmax(dim=1), labels.argmax(dim=1))

            if iter_count % args.log_interval == 0 and log:
                used = time.time() - start
                remain = (args.iters - iter_count - 1) / (iter_count + 1) * used
                lr_string = '%.4g' % lr
                logger.info('iter:{}/{}, {}, {}, lr:{}  --  {} mins used, {} mins remain'
                            .format(iter_count, args.iters, average_loss_gt, average_loss_uda,
                                    lr_string, int(used / 60), int(remain / 60)))
                writer.add_scalar('basic_learning_metrics/average_loss_gt', average_loss_gt.average, global_step = iter_count / args.log_interval)
                writer.add_scalar('basic_learning_metrics/average_loss_uda', average_loss_uda.average, global_step = iter_count / args.log_interval)
                writer.add_scalar('basic_learning_metrics/learning_rate', lr, global_step = iter_count / args.log_interval)
                
                average_loss_gt.reset()
                average_loss_uda.reset()
                if args.distributed:
                    uda_log = model.module.uda_log()
                    if len(uda_log) > 0:
                        logger.info(uda_log)
                else:
                    uda_log = model.uda_log()
                    if len(uda_log) > 0:
                        logger.info(uda_log)
            if iter_count % (args.log_interval * 5) == 0 and log:
                step = iter_count / (args.log_interval * 5)
                logger = remove_formatter(logger)
                logger.info(str(metrics))
                metrics.write_to_tensorboard(writer, step = step)  
                metrics.reset()
                logger = add_formatter(logger)
                # show inputs of model
                input_tmp = (denormalize(inputs[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])*255)[0].cpu().numpy().astype('uint8')
                label_tmp = palette[labels[0].argmax(0).cpu().numpy()].astype('uint8').transpose(2, 0, 1)
                output_tmp = palette[outputs[0].argmax(0).cpu().numpy()].astype('uint8').transpose(2, 0, 1)
                img_tmp = np.concatenate([input_tmp, label_tmp, output_tmp], axis=2)
                writer.add_image('train/data,label,output', img_tmp, global_step = step, dataformats='CHW')
            
            if iter_count % args.save_interval == 0 and log:
                weight_dict = model.module.state_dict() if args.distributed else model.state_dict()
                weight_dict = {n: weight_dict[n] for n in weight_dict.keys() if not n.startswith('teacher_model')}
                torch.save(weight_dict, os.path.join(work_dir, 'weight.pth'))
            
         
            if iter_count % (args.log_interval * 100) == 0 and log and args.order != 'test':
                eval(model, source_val_loader, device, args, writer, iter_count / (args.log_interval * 100), 'SOURCE_')
                eval(model, target_val_loader, device, args, writer, iter_count / (args.log_interval * 100), 'TARGET_')

            iter_count = iter_count + 1
            if iter_count >= args.iters and log:
                weight_dict = model.module.state_dict() if args.distributed else model.state_dict()
                weight_dict = {n: weight_dict[n] for n in weight_dict.keys() if not n.startswith('teacher_model')}
                torch.save(weight_dict, os.path.join(work_dir, 'weight.pth'))
                logger = remove_formatter(logger)
                logger.info('\nTrain Finished .\n\n')
            if iter_count >= args.iters:
                return
        epoch = epoch + 1

def eval(model, val_loader, device, args, writer, step, mark):
    """  
    在训练时，每隔一段时间eval
    :param model:           模型
    :param val_loader:      训练集的loader
    :param device:          设备
    :param args:            设定
    :param writer:          tensorboard的summary writer
    :param step:            tensorboard的step
    :param mark:            tensorboard的mark
    :return:                无
    """
    metrics_tmp = Metrics()
    if args.morphology:
        morphology = Morphology(k_object=args.k_object, k_back=args.k_back, kernel_shape=args.kernel_shape)
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(val_loader), desc=mark+'val', total=min(len(val_loader), args.max_eval_num)):
            if i >= args.max_eval_num:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            shape = inputs.shape
            if args.resize:
                inputs, _ = resize(inputs, ratio=args.resize_ratio)
            if args.crop:
                recover_shape = inputs.shape
                inputs, _ = cover_crop(args.crop_size, inputs)
                for j in range(inputs.shape[0]):
                    if j == 0:
                        outputs = model(inputs[j, :, :, :].unsqueeze(dim=0))
                    else:
                        outputs = torch.cat([outputs, model(inputs[j, :, :, :].unsqueeze(dim=0))], dim=0)
                outputs = recover_from_crop(outputs, recover_shape)
            else:
                outputs = model(inputs)
            if args.resize:
                outputs, _ = resize(outputs, mode='bilinear', size=[shape[2], shape[3]])
            if args.morphology:
                outputs = morphology.morphology_process(outputs)
                results = outputs.argmax(dim=1)
                results = morphology.seg_fill(results)
            else:
                results = outputs.argmax(dim=1)
            metrics_tmp.update(results, labels.argmax(dim=1))
        metrics_tmp.write_to_tensorboard(writer, step=step, mark=mark) 
        
        input_tmp = (denormalize(inputs[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])*255)[0].cpu().numpy().astype('uint8')
        label_tmp = palette[labels[0].argmax(0).cpu().numpy()].astype('uint8').transpose(2, 0, 1)
        output_tmp = palette[outputs[0].argmax(0).cpu().numpy()].astype('uint8').transpose(2, 0, 1)
        img_tmp = np.concatenate([input_tmp, label_tmp, output_tmp], axis=2)
        writer.add_image('eval/data,label,output', img_tmp, global_step = step, dataformats='CHW')
    return




def adjust_learning_rate(optimizer, iter_count, init_lr, iters, warm_up_iters):
    """
    调整学习率，方案：Warm-up + cosine
    :param optimizer:       优化器
    :param iter_count:      当前epoch
    :param init_lr:         初始学习率
    :param iters:           总epoch数
    :param warm_up_iters:   warm up的iter数目
    :return:                lr
    """
    intersection_value = 0.5 * init_lr * (1 + math.cos(math.pi * (warm_up_iters - 1) / float(iters)))
    if iter_count < warm_up_iters:
        lr = (init_lr * 0.1 + iter_count * (intersection_value - init_lr * 0.1) / (warm_up_iters - 1))
    else:
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * iter_count / float(iters)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == "__main__":
    print(1)
