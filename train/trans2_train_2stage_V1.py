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


from util.make_optimizer import make_optimizer_2stage, make_optimizer_2stage_later
from util.optim.lr_scheduler import WarmupMultiStepLR
from util.loss.softmax_loss import CrossEntropyLabelSmooth
import cv2
from util.transforms import RandomColoring

# torch.backends.cuda.max_split_size_mb = 2750

def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False
        # , pin_memory=True
    )
    return cluster_loader

def do_train_stage2_v5(args,
                    model,
                    optimizer,
                    scheduler,
                    loss_fn
                    ):
    best_acc = 0
    device = 'cuda'
    epochs = args.stage2_maxepochs
    start_time = time.monotonic()

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.dataset == 'sysu':
        transform_train_rgb = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.ToTensor(),
            RandomColoring(p=0.5, is_rgb=True),
            normalizer,
            transforms.RandomErasing(p=0.5)
        ])
        transform_train_ir = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.ToTensor(),
            RandomColoring(p=0.5, is_rgb=False),
            normalizer,
            transforms.RandomErasing(p=0.5),
        ])
    else:
        transform_train_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomGrayscale(p=0.5),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
        transforms.RandomErasing(p=0.5),
        ])
        transform_train_ir = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor(),
            normalizer,
            transforms.RandomErasing(p=0.5),
        ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
    ])


    batch = args.stage2_ims_per_batch
    num_classes_rgb = model.num_classes_rgb
    num_classes_ir = model.num_classes_ir
    i_ter_rgb = num_classes_rgb // batch
    i_ter_ir = num_classes_ir // batch
    left_rgb = num_classes_rgb-batch* (num_classes_rgb//batch)
    left_ir = num_classes_ir-batch* (num_classes_ir//batch)
    if left_rgb != 0 :
        i_ter_rgb = i_ter_rgb+1
    if left_ir != 0 :
        i_ter_ir = i_ter_ir+1
    text_features_rgb = []
    text_features_ir = []
    with torch.no_grad():
        for i in range(i_ter_rgb):
            if i+1 != i_ter_rgb:
                l_list_rgb = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list_rgb = torch.arange(i*batch, num_classes_rgb)
            # with amp.autocast(enabled=True):
            text_feature_rgb = model(get_text = True, label = l_list_rgb, modal=1)
            text_features_rgb.append(text_feature_rgb.cpu())
        text_features_rgb = torch.cat(text_features_rgb, 0).cuda()
    with torch.no_grad():
        for i in range(i_ter_ir):
            if i+1 != i_ter_ir:
                l_list_ir = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list_ir = torch.arange(i*batch, num_classes_ir)
            # with amp.autocast(enabled=True):
            text_feature_ir = model(get_text = True, label = l_list_ir, modal=2)
            text_features_ir.append(text_feature_ir.cpu())
        text_features_ir = torch.cat(text_features_ir, 0).cuda()

    del text_feature_rgb,text_feature_ir

    scaler = amp.GradScaler()
    losses = AverageMeter()
    losses_all = AverageMeter()
    losses_i2t = AverageMeter()
    losses_id = AverageMeter()
    losses_tri = AverageMeter()

    xent = CrossEntropyLabelSmooth(num_classes=num_classes_rgb)


    for epoch in range(1, epochs+1):
        end = time.time()
        if args.dataset == 'sysu':
            trainset = SYSUData_Stage2(args.data_path, transform_train_rgb, transform_train_ir)
        else:
            trainset = RegDBData_Stage2(args.data_path, args.trial, transform_train_rgb, transform_train_ir)
        print("New Dataset Information---- ")
        print("  ----------------------------")
        print("  subset   | # ids | # images")
        print("  ----------------------------")
        print("  visible  | {:5d} | {:8d}".format(len(np.unique(trainset.train_color_label)),
                                                  len(trainset.train_color_image)))
        print("  thermal  | {:5d} | {:8d}".format(len(np.unique(trainset.train_thermal_label)),
                                                  len(trainset.train_thermal_image)))
        print("  ----------------------------")
        print("Data loading time:\t {:.3f}".format(time.time() - end))

        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        sampler = IdentitySampler_nosk_stage4(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos,
                                       args.num_instances, args.batch_size)

        trainset.cIndex = sampler.index1
        trainset.tIndex = sampler.index2

        trainloader = data.DataLoader(trainset, batch_size=args.batch_size * args.num_instances, sampler=sampler,
                                      num_workers=args.workers,
                                      drop_last=True)

        losses.reset()
        losses_all.reset()
        losses_i2t.reset()
        losses_id.reset()
        losses_tri.reset()


        scheduler.step()
        # model.train()
        for n_iter, (img_rgb, img_ir, label_rgb, label_ir) in enumerate(trainloader):
            model.train()

            optimizer.zero_grad()
            img_rgb = img_rgb.to(device)
            label_rgb = label_rgb.to(device, dtype=torch.int64)

            img_ir = img_ir.to(device)
            label_ir = label_ir.to(device, dtype=torch.int64)

            # img_rgb = img_rgb.permute(0,3,1,2)
            # img_ir = img_ir.permute(0,3,1,2)

            # 开始提取特征计算损失
            with amp.autocast(enabled=True):
                # res_rgb, res_ir = model(x1=img_rgb, x2=img_ir, modal=0)

                # score_rgb, feat_rgb, image_features_rgb, score_ir, feat_ir, image_features_ir = model(x1=img_rgb, x2=img_ir, modal=0)
                score_all, feat_all, image_features_all = model(x1=img_rgb, x2=img_ir, modal=0)

                # print(score_ir[0].shape, score_rgb[0].shape)

                logits_rgb = image_features_all[:img_rgb.size(0)] @ text_features_rgb.t()
                logits_ir = image_features_all[img_rgb.size(0):] @ text_features_ir.t()

                I2TLOSS = xent(logits_rgb, label_ir) + xent(logits_ir, label_rgb)

                loss_all = loss_fn(score_all, feat_all, torch.cat((label_rgb,label_ir),dim=0))

                ID_LOSS, TRI_LOSS = loss_all

                loss_all = ID_LOSS + TRI_LOSS + I2TLOSS


            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()

            losses_all.update(loss_all.item())
            losses_i2t.update(I2TLOSS.item())
            losses_id.update(ID_LOSS.item())
            losses_tri.update(TRI_LOSS.item())

            losses.update(loss_all.item())
            torch.cuda.synchronize()
            if n_iter % args.print_freq == 0:
                print("Epoch[{}] Iteration[{}/{}], Loss_rgb_ir_i2t_id_tri: ({:.3f}) ({:.3f}) ({:.3f}) ({:.3f}), Base Lr: {:.2e}"
                 .format(epoch, (n_iter + 1), len(trainloader), losses_all.avg,
                         losses_i2t.avg, losses_id.avg, losses_tri.avg, scheduler.get_lr()[0]))


        if epoch % args.eval_step == 0 or (epoch == args.stage2_maxepochs):
            print("start test")
        # if True:
            if args.dataset == 'sysu':
                print('Test Epoch: {}'.format(epoch))
                test_mode = [1, 2]
                query_img, query_label, query_cam = process_query_sysu(args.data_path, mode=args.mode)
                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

                for trial in range(10):
                    # print('-------test trial {}-------'.format(trial))
                    gall_img, gall_label, gall_cam = process_gallery_sysu(args.data_path, mode=args.mode, trial=trial)
                    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

                    cmc, mAP, mINP = tester(args, epoch, model, test_mode, gall_label, gall_loader, query_label, query_loader,
                                            feat_dim=3072,
                                            query_cam=query_cam, gall_cam=gall_cam)

                    if trial == 0:
                        all_cmc = cmc
                        all_mAP = mAP
                        all_mINP = mINP
                    else:
                        all_cmc = all_cmc + cmc
                        all_mAP = all_mAP + mAP
                        all_mINP = all_mINP + mINP

                cmc = all_cmc / 10
                mAP = all_mAP / 10
                mINP = all_mINP / 10
                print(
                    "Performance[ALL]: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
                        cmc[0], cmc[4],
                        cmc[9], cmc[19],
                        mAP, mINP))

                if cmc[0] > best_acc:
                    best_acc = cmc[0]
                    best_epoch = epoch
                    best_mAP = mAP
                    best_mINP = mINP
                    state = {
                        "state_dict": model.state_dict(),
                        "cmc": cmc,
                        "mAP": mAP,
                        "mINP": mINP,
                        "epoch": epoch,
                    }
                    torch.save(state, os.path.join(args.model_path, args.logs_file + "V5_stage2_v1.pth"))
                print("Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}".format(best_epoch, best_acc, best_mAP, best_mINP))
            elif args.dataset == 'regdb':
                print('Test Epoch: {}'.format(epoch))

                query_img, query_label = process_test_regdb(img_dir=args.data_path, trial=args.trial, modal='visible')
                gall_img, gall_label = process_test_regdb(img_dir=args.data_path, trial=args.trial, modal='thermal')

                test_mode = [2, 1]
                gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

                # testing data loader
                gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

                cmc, mAP, mINP = tester(args, epoch, model, test_mode, gall_label, gall_loader, query_label, query_loader,
                                        feat_dim=3072)

                print(
                    "Performance[ALL]: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
                        cmc[0], cmc[4],
                        cmc[9], cmc[19],
                        mAP, mINP))
                if cmc[0] > best_acc:
                    best_acc = cmc[0]
                    best_epoch = epoch
                    best_mAP = mAP
                    best_mINP = mINP
                    state = {
                        "state_dict": model.state_dict(),
                        "cmc": cmc,
                        "mAP": mAP,
                        "mINP": mINP,
                        "epoch": epoch,
                    }
                    torch.save(state, os.path.join(args.model_path, args.logs_file + "_stage2_regdb.pth"))
                print("Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}".format(best_epoch, best_acc, best_mAP, best_mINP))
            else:
                print('please input correct dataset!!')

        torch.cuda.empty_cache()

    end_time = time.monotonic()
    print('Stage2 running time: ', timedelta(seconds=end_time - start_time))

