from tool.transform.crop import cover_crop, recover_from_crop
from tool.transform.restore import denormalize
from tool.morphology import Morphology
from tool.transform.resize import resize
from tool.metrics import Metrics
from utils.parser import *
from utils.logger import *
import numpy as np
import torch
import time
import cv2
import os

cv2.setNumThreads(0)


palette = np.array([[0, 0, 0], [128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102], [255, 255, 255]])


def validate(model, data_loader, device, args):
    """
    验证
    :param model:           模型
    :param data_loader:     数据集
    :param device:          设备
    :param args:            参数
    :return:                无
    """
    start = time.time()
    model.eval()
    model.to(device)
    metrics = Metrics()
    args.iters = len(data_loader)
    work_dir = os.path.join(os.path.join(args.output_dir, 'exps'), args_to_str(args))
    output_result = args.output
    if output_result:
        work_dir_visual = os.path.join(work_dir, 'visual')
        work_dir_label = os.path.join(work_dir, 'label')
        work_dir_blend = os.path.join(work_dir, 'blend')
        if not os.path.exists(work_dir_visual):
            os.makedirs(work_dir_visual)
        if not os.path.exists(work_dir_label):
            os.makedirs(work_dir_label)
        if not os.path.exists(work_dir_blend):
            os.makedirs(work_dir_blend)
    logger = remove_formatter(get_logger(work_dir))
    for item in args_to_str_list(args):
        logger.info(item)
    logger.info('\nStart Validating ...')
    logger = add_formatter(logger)
    if args.morphology:
        morphology = Morphology(k_object=args.k_object, k_back=args.k_back, kernel_shape=args.kernel_shape)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            # print(inputs.shape)
            # print(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            if output_result:
                images = denormalize(inputs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if args.resize:
                inputs, _ = resize(inputs, ratio=args.resize_ratio)
            if args.crop:
                shape = inputs.shape
                inputs, _ = cover_crop(args.crop_size, inputs)
                for j in range(inputs.shape[0]):
                    x = inputs[j, :, :, :].unsqueeze(dim=0)
                    x_1 = x.flip(3)
                    if j == 0:
                        y = model(x)
                        y1 = model(x_1).flip(3)
                        outputs = (y1+y)/2
                    else:
                        outputs = torch.cat([outputs, model(inputs[j, :, :, :].unsqueeze(dim=0))], dim=0)
                outputs = recover_from_crop(outputs, shape)
            else:
                outputs = model(inputs)
            if args.resize:
                outputs, _ = resize(outputs, mode='bilinear', size=[labels.shape[2], labels.shape[3]])
            if args.morphology:
                outputs = morphology.morphology_process(outputs)
                results = outputs.argmax(dim=1)
                results = morphology.seg_fill(results)
            else:
                results = outputs.argmax(dim=1)
            metrics.update(results, labels.argmax(dim=1))
            # if output_result:
            #     results = results.cpu().numpy()
            #     for j, item in enumerate(list(name)):
            #         seg = results[j, :, :]
            #         cv2.imwrite(os.path.join(work_dir_label, item), seg)
            #         cv2.imwrite(os.path.join(work_dir_visual, item), palette[seg])
            #         img = (images[j, :, :, :] * 255).permute(1, 2, 0)[:, :, [2, 1, 0]].type(torch.uint8).cpu().numpy()
            #         cv2.imwrite(os.path.join(work_dir_blend, item), (palette[seg] * 0.25 + img * 0.75).astype(np.uint8))
            if i % args.log_interval == 0:
                used = time.time() - start
                remain = (args.iters - i - 1) / (i + 1) * used
                logger.info('iter:{}/{}  --  {} mins used, {} mins remain'
                            .format(i, args.iters, int(used / 60), int(remain / 60)))
    logger = remove_formatter(logger)
    logger.info(str(metrics))
    logger.info('\nValidate Finished .\n\n')
    logger = add_formatter(logger)
    return metrics.m_iou()


if __name__ == "__main__":
    print(1)
