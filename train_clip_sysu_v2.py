import argparse
import os
import os.path as osp
from datetime import timedelta
import time
import sys
import random

import easydict
import numpy as np
import yaml

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.backends import cudnn

from train.train_2stage_v3 import do_train_stage2_v3
from train.train_2stage import do_train_stage2
from util.loss.make_loss import make_loss
from train.train_1stage_v2 import do_train_stage1_v2
from train.train_1stage import do_train_stage1
from util.make_optimizer import make_optimizer_1stage, make_optimizer_2stage
from util.optim.lr_scheduler import WarmupMultiStepLR
from util.optim.scheduler_factory import create_scheduler
from data.dataloader_v2 import sysu, ImageDataset
from model.make_model_clip import build_model
from util.utils import Logger
from model.make_model_clip import load_clip_to_cpu
from util.transforms import RandomColoring


start_epoch = best_mAP = 0


def main(args):
    # args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()
    cudnn.benchmark = True

    logs_time = args.logs_time
    logs_file = str(args.logs_file)

    sys.stdout = Logger(osp.join(args.logs_dir, logs_time, logs_file + '.txt'))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("==========\nArgs:{}\n==========".format(args))
    # Load datasets
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transform_test = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((args.img_h, args.img_w)),
    #     transforms.ToTensor(),
    #     normalizer,
    # ])
    transform_test_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((288, 144)),
        transforms.ToTensor(),
        RandomColoring(p=0.5,is_rgb=True),
        normalizer,
    ])
    transform_test_ir = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((288, 144)),
        transforms.ToTensor(),
        RandomColoring(p=0.5,is_rgb=False),
        normalizer,
    ])

    end = time.time()
    print("==> Load unlabeled dataset")
    # data_path = '/home/cz/dataset/SYSU-MM01/'
    data_path = args.data_path
    dataset = ImageDataset(sysu(data_path).train, transform=transform_test_rgb)

    n_color_class = len(np.unique(dataset.train_color_label))
    n_thermal_class = len(np.unique(dataset.train_thermal_label))
    # num_classes = n_color_class + n_thermal_class

    print("Dataset {} Statistics:".format(args.dataset))
    print("  ----------------------------")
    print("  subset   | # ids | # images")
    print("  ----------------------------")
    print("  visible  | {:5d} | {:8d}".format(n_color_class, len(dataset.train_color_image)))
    print("  thermal  | {:5d} | {:8d}".format(n_thermal_class, len(dataset.train_thermal_image)))
    print("  ----------------------------")
    # print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), len(query_label)))
    # print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), len(gall_label)))
    print("  ----------------------------")
    print("Data loading time:\t {:.3f}".format(time.time() - end))

    # Create model
    model = build_model(args, n_color_class, n_thermal_class)


    # checkpoint = torch.load(args.resume_path)
    # model.load_state_dict(checkpoint['state_dict'])
    model.to("cuda")


    # Optimizer
    # optimizer_1stage = make_optimizer_1stage(args, model)
    # scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs=args.stage1_maxepochs, lr_min=args.stage1_lrmin,
    #                                        warmup_lr_init=args.stage1_warmup_lrinit, warmup_t=args.stage1_warmup_epoch, noise_range=None)
    #
    # print("开始一阶段训练")
    # do_train_stage1_v2(args, dataset, model, optimizer_1stage, scheduler_1stage)

    optimizer_2stage = make_optimizer_2stage(args, model)
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, args.stage2_steps, args.stage2_gamma, args.stage2_warmup_factor,
                                         args.stage2_warmup_iters, args.stage2_warmup_method)

    loss_func = make_loss(args, num_classes=n_color_class)

    do_train_stage2_v3(args, model, optimizer_2stage, scheduler_2stage, loss_func)


    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Contrastive learning on unsupervised Cross re-ID")
    args_main = parser.parse_args()

    args = yaml.load(open('config/config_sysu.yaml'), Loader=yaml.FullLoader)
    args = easydict.EasyDict(args)
    main(args)
