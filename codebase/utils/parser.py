import argparse

from numpy import Inf, True_


def parse_args(args):
    """
    转译参数
    :param args:    参数
    :return:        args
    """
    parser = argparse.ArgumentParser(description='Train UDA')
    # 基础配置信息
    parser.add_argument('--order', choices=['train', 'uda', 'val', 'test'], default='test')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--model', choices=['fcn', 'daformer', 'segclip'], default='fcn')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--w_decay', type=float, default=1e-6)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default='../')
    # 数据相关信息
    parser.add_argument('--data', nargs='+', default='', help='train set for source')
    parser.add_argument('--data_val', nargs='+', default='', help='eval set for source')          
    parser.add_argument('--data_target', nargs='+', default='', help='train set for target')
    parser.add_argument('--data_target_val', nargs='+', default='', help='eval set for target')  
    parser.add_argument('--max_eval_num', type=int, default=1000000, help='max num of eval set')  

    # crop相关配置信息
    parser.add_argument('--crop', action='store_true', default=False)
    parser.add_argument('--crop_size', type=int, nargs='+', default=600)
    # resize相关配置信息
    parser.add_argument('--resize', action='store_true', default=False)
    parser.add_argument('--resize_ratio', type=float, default=0.5)
    parser.add_argument('--rand_resize', action='store_true', default=False)
    # 数据增强相关配置信息
    parser.add_argument('--augment', action='store_true', default=False)
    # MixUp相关配置信息
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--mix_mode', choices=['hard', 'object', 'soft', 'random'], default='hard')
    parser.add_argument('--mix_k_min', type=float, default=0.6)
    parser.add_argument('--mix_k_max', type=float, default=0.8)
    parser.add_argument('--mix_p', type=float, default=0.7)
    # progressive learning相关配置信息
    parser.add_argument('--pl', action='store_true', default=False)
    # 数据均衡化采样配置信息
    parser.add_argument('--balance', action='store_true', default=False)
    # 形态学后处理相关配置信息
    parser.add_argument('--morphology', action='store_true', default=False)
    parser.add_argument('--k_object', type=float, default=0.05)
    parser.add_argument('--k_back', type=float, default=0.02)
    parser.add_argument('--kernel_shape', choices=['circle', 'square'], default='square')
    # 搜索相关配置信息
    parser.add_argument('--search', action='store_true', default=False)
    parser.add_argument('--search_target', type=str, default='k_object')
    parser.add_argument('--search_start', type=float, default=0.0)
    parser.add_argument('--search_delta', type=float, default=1.0)
    parser.add_argument('--search_arrange', nargs='+', type=float, default=-1.0)
    parser.add_argument('--end_threshold', type=float, default=0.1)
    # UDA相关配置信息
    parser.add_argument('--uda_version', type=int, default=0)
    parser.add_argument('--source_loose', action='store_true', default=False)
    # 其他
    parser.add_argument('--output', action='store_true', default=False)
    # 多卡并行
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--world_size', type=int, default=4, help='distributed world size')
    parser.add_argument('--local_rank', type=int, default=-1)
    # CLIPSeg训练相关
    parser.add_argument('--freeze_clip', action='store_true', default=False)
    
    parser.add_argument('--tag', type=str, default='1.1', help='add to dir while saving')
    args = parser.parse_args(args)
    
    if not args.crop:
        args.crop_size = -1
    else:
        if not isinstance(args.crop_size, list):
            args.crop_size = [args.crop_size, args.crop_size]
        elif len(args.crop_size) == 1:
            args.crop_size = [args.crop_size[0], args.crop_size[0]]
    if not args.resize:
        args.resize_ratio = -1
    if args.search_arrange == -1 or args.search_arrange == [-1]:
        args.search_arrange = None
    assert args.mix_k_min <= args.mix_k_max
    return args


def args_to_str(args):
    """
    核心参数转化为字符串，主要用于构造不同实验的文件夹名称
    :param args:    args
    :return:        字符串
    """
    string = args.order
    if args.search:
        string = 'search'
        string += ('_' + args.search_target)
        string += ('_' + args.order)
    if 'uda' in args.order:
        string += str(args.uda_version)
    string += ('_' + args.model)
    string += (('_' + args.name) if args.name != '' else '')
    string += ('_iters' + str(args.iters))
    if 'train' in args.order or 'uda' in args.order:
        if 'segclip' in args.model and args.freeze_clip:
            string += '_freeze'
        string += ('_bs' + str(args.bs))
        string += ('_lr' + str(args.lr))
        string += (('_drop' + str(args.drop)) if args.drop > 0 else '')
        string += ('_augment' if args.augment else '')
        string += (('_mixup' + str(args.mix_k_min) + '-' + str(args.mix_k_max)
                    + 'p' + str(args.mix_p) + args.mix_mode) if args.mixup else '')
        string += ('_pl' if args.pl else '')
        string += ('_balance' if args.balance else '')
        if args.resize:
            if not args.rand_resize:
                string += ('_resize' + str(args.resize_ratio))
            else:
                string += ('_randResize' + str(args.resize_ratio))
        if args.crop:
            string += ('_crop' + str(args.crop_size[0]) + '.' + str(args.crop_size[1]))
        if args.source_loose:
            string += '_loose'
    return string


def args_to_str_list(args):
    """
    所有参数转换为字符串列表
    用于保存完整实验设置
    :param args:    args
    :return:        字符串列表
    """
    arg_list = list()
    arg_list.append('order:\t\t\t' + args.order)
    if 'uda' in args.order:
        arg_list.append('uda_version:\t\t' + str(args.uda_version))
        arg_list.append('source_loose:\t\t' + str(args.source_loose))
    arg_list.append('name:\t\t\t' + args.name)
    arg_list.append('model:\t\t\t' + args.model)
    arg_list.append('model_path:\t\t\t' + args.model_path)
    arg_list.append('iters:\t\t\t' + str(args.iters))
    if 'train' in args.order or 'uda' in args.order:
        if 'segclip' in args.model:
            arg_list.append('freeze_clip:\t\t' + str(args.freeze_clip))
        arg_list.append('bs:\t\t\t\t' + str(args.bs))
        arg_list.append('lr:\t\t\t\t' + str(args.lr))
        arg_list.append('w_decay:\t\t' + str(args.w_decay))
        arg_list.append('drop:\t\t\t' + str(args.drop))
    arg_list.append('data:\t\t\t' + str(args.data))
    arg_list.append('data_val:\t\t\t' + str(args.data_val))              
    arg_list.append('data_target:\t\t' + str(args.data_target))         
    arg_list.append('data_target_val:\t\t' + str(args.data_target_val))  
    arg_list.append('max_eval_num:\t\t' + str(args.max_eval_num))        
    arg_list.append('log_interval:\t' + str(args.log_interval))
    arg_list.append('save_interval:\t' + str(args.save_interval))
    arg_list.append('output_dir:\t\t' + str(args.output_dir))
    arg_list.append('crop:\t\t\t' + str(args.crop))
    arg_list.append('crop_size:\t\t' + str(args.crop_size))
    arg_list.append('resize:\t\t\t' + str(args.resize))
    arg_list.append('resize_ratio:\t' + str(args.resize_ratio))
    arg_list.append('rand_resize:\t\t' + str(args.rand_resize))
    arg_list.append('augment:\t\t' + str(args.augment))
    arg_list.append('mixup:\t\t\t' + str(args.mixup))
    if args.mixup:
        arg_list.append('mix_mode\t\t' + args.mix_mode)
        arg_list.append('mix_k_min:\t\t' + str(args.mix_k_min))
        arg_list.append('mix_k_max:\t\t' + str(args.mix_k_max))
        arg_list.append('mix_p:\t\t\t' + str(args.mix_p))
    arg_list.append('pl:\t\t\t' + str(args.pl))
    arg_list.append('balance:\t\t\t' + str(args.balance))
    arg_list.append('output:\t\t\t' + str(args.output))
    if args.morphology:
        arg_list.append('morphology:\t\t' + str(args.morphology))
        arg_list.append('k_object:\t\t' + str(args.k_object))
        arg_list.append('k_back:\t\t\t' + str(args.k_back))
        arg_list.append('kernel_shape:\t\t' + args.kernel_shape)
    if args.search:
        arg_list.append('search:\t\t\t' + str(args.search))
        arg_list.append('search_target:\t\t' + args.search_target)
        arg_list.append('search_start:\t\t' + str(args.search_start))
        arg_list.append('search_delta:\t\t' + str(args.search_delta))
        arg_list.append('search_arrange:\t\t' + str(args.search_arrange))
        arg_list.append('end_threshold:\t\t' + str(args.end_threshold))
    return arg_list
