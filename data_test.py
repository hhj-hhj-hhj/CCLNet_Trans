import argparse
import collections
import os
import os.path as osp
import shutil
from datetime import timedelta
import time
import sys
import random

import easydict
import numpy as np
import yaml
from sklearn.cluster import DBSCAN
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from torch.cuda import amp

from util.eval_metrics import extract_features_clip
from util.faiss_rerank import compute_jaccard_distance
from util.loss.supcontrast import SupConLoss
from util.utils import AverageMeter

from ClusterContrast.cm import ClusterMemory
from data.data_manager import process_query_sysu, process_gallery_sysu
from data.data_manager import process_test_regdb
from data.dataloader import SYSUData_Stage2, RegDBData_Stage2, IterLoader, TestData
from util.eval import tester
from util.utils import IdentitySampler_nosk, GenIdx, IdentitySampler_nosk_stage4
import matplotlib.pyplot as plt

from util.make_optimizer import make_optimizer_2stage, make_optimizer_2stage_later
from util.optim.lr_scheduler import WarmupMultiStepLR
from util.loss.softmax_loss import CrossEntropyLabelSmooth

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Pad(10),
    transforms.RandomCrop((288, 144)),
    transforms.ToTensor(),
    normalizer,
    transforms.RandomErasing(p=0.5),
])

trainset = SYSUData_Stage2('E:/hhj/SYSU-MM01/')

color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

sampler = IdentitySampler_nosk_stage4(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos,
                               16, 4)

trainset.cIndex = sampler.index1
trainset.tIndex = sampler.index2

trainloader = data.DataLoader(trainset, batch_size=64, sampler=sampler,
                              num_workers=0,
                              drop_last=True)

def trans_color(original_imgs, parsing_imgs):
    trans_imgs = original_imgs.clone()
    clothes_color = torch.tensor([128, 0, 128], device=original_imgs.device, dtype=torch.uint8)
    new_color = torch.tensor([0, 255, 0], device=original_imgs.device, dtype=torch.uint8)

    clothes_pixels = torch.all(parsing_imgs == clothes_color, dim=-1)
    trans_imgs[clothes_pixels] = new_color
    return trans_imgs

device = torch.device('cuda')

for n_iter, (img_rgb, img_ir, label_rgb, label_ir, img_parsing) in enumerate(trainloader):
    img_rgb = img_rgb.to(device)
    img_ir = img_ir.to(device)
    img_parsing = img_parsing.to(device)
    print(img_rgb.shape, img_parsing.shape)
    trans = trans_color(img_rgb, img_parsing)
    for rgb, parsing, train_c in zip(img_rgb, img_parsing, trans):

        print(rgb.shape, parsing.shape)

        # OpenCV需要将图像从RGB转换为BGR
        rgb_bgr = cv2.cvtColor(rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)
        parsing_bgr = cv2.cvtColor(parsing.cpu().numpy(), cv2.COLOR_RGB2BGR)
        train_c_bgr = cv2.cvtColor(train_c.cpu().numpy(), cv2.COLOR_RGB2BGR)

        cv2.imshow('RGB', rgb_bgr)
        cv2.imshow('Parsing', parsing_bgr)
        cv2.imshow('Train_C', train_c_bgr)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.imshow(rgb.cpu().numpy())
        # plt.subplot(1, 3, 2)
        # plt.imshow(parsing.cpu().numpy())
        # plt.subplot(1, 3, 3)
        # plt.imshow(train_c.cpu().numpy())

        # plt.show()
