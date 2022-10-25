from tool.transform.restore import denormalize
from PIL import Image
import numpy as np
import argparse
import imageio
import tqdm
import os


PALETTE = np.array([[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [255, 255, 255]])


def convert_seg(gt):
    """
    将0-4的数字标签转化为可视化的彩色图片

    使用：
        通常不在外部调用

    :param gt:  标注，np.array
    :return:    可视化结果
    """
    return PALETTE[gt]


def visualize_gt(gt, is_tensor=True):
    """
    将0-4的数字标签转化为可视化的彩色图片并显示
    主要用于在本地调试各种操作，直观地查看数据是否正确

    使用：
        主要用于本地调试，直接调用
        visualize_gt(gt) or visualize_gt(labels.argmax(dim=1))
        即可显示标注

    :param gt:          标注，tensor3d，eg. (1(bs), 1080, 1920)
    :param is_tensor:   是否是tensor，bool
    :return:            无
    """
    if is_tensor:
        gt = gt[0, :, :].cpu().numpy()
    gt = PALETTE[gt]
    Image.fromarray(np.uint8(gt)).show()


def visualize_data(data, is_tensor=True, denorm=True):
    """
    将数据图片显示出来
    主要用于在本地调试各种操作，直观地查看数据是否正确

    使用：
        主要用于本地调试，直接调用
        visualize_data(data)
        即可显示数据图片

    :param data:        数据，tensor4d，eg. (1(bs), 3， 1080, 1920),
    :param is_tensor:   是否是tensor，bool
    :param denorm:      是否需要取消归一化，bool
    :return:            无
    """
    if is_tensor:
        if denorm:
            data = denormalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        data = data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
    Image.fromarray(np.uint8(data * 255)).show()


def main():
    """
    将某个文件夹下的所有gt图片可视化，并保存结果于另一个文件夹中
    :return:    无
    """
    parser = argparse.ArgumentParser(description='Convert ZeroWaste visuals to labels.')
    parser.add_argument('vis_folder', type=str,
                        help='path to the folder with predicted labels.')
    parser.add_argument('out_folder', type=str,
                        help='output path with predicted visuals.')
    args = parser.parse_args()
    os.makedirs(args.out_folder, exist_ok=True)
    img_list = os.listdir(args.vis_folder)
    for img_name in tqdm.tqdm(img_list):
        pred_img = imageio.imread(os.path.join(args.vis_folder, img_name))
        pred_lbl_img = convert_seg(pred_img)
        imageio.imsave(
            os.path.join(args.out_folder, img_name),
            pred_lbl_img.astype(np.uint8))


if __name__ == "__main__":
    main()