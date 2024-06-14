import os
import time
from datetime import timedelta
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.cuda import amp

from util.eval_metrics import extract_features_clip, extract_features_clip_ir
from util.loss.supcontrast import SupConLoss
from util.utils import AverageMeter

from data.data_manager import process_query_sysu, process_gallery_sysu
from data.dataloader import TestData
from data.data_manager import process_test_regdb
from util.eval import tester
import cv2


def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False
        # , pin_memory=True
    )
    return cluster_loader

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

trans_idx = [1,3,5,6,7,8,9,10,11,12,18,19]
palette = get_palette(20)
trans_color_list = [np.array(palette[3*i:3*i + 3]) for i in trans_idx]


def adjust_clothes_color(image_numpy, mask_numpy):
    """
    根据提供的mask随机改变衣服颜色并保持亮度不变。
    """
    # 将图像从RGB转换到HSV颜色空间
    image_hsv = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2HSV)

    # 生成一个随机色调
    random_hue = np.random.randint(0, 180)  # HSV色调范围是[0, 179]

    # 获得mask的布尔索引，扩展维度以适应HSV图像
    mask_boolean = np.squeeze(mask_numpy.astype(bool))
    mask_hsv = np.zeros_like(image_hsv, dtype=bool)
    mask_hsv[:, :, 0] = mask_boolean

    # 只改变衣服区域的色调
    image_hsv[mask_hsv] = random_hue
    # 只改变衣服区域的色调
    image_hsv[mask_hsv[:, :, 0]] = (image_hsv[mask_hsv[:, :, 0]] + random_hue) % 180

    # HSV颜色空间转换回RGB
    adjusted_image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    return adjusted_image_rgb


def trans_color2(trans_img, parsing_img, clothe_color_list):
    parsing_img = parsing_img.cpu().numpy()

    trans_img = trans_img.cpu().numpy()
    for clothe_color in clothe_color_list:
        clothes_pixels = np.all(parsing_img == clothe_color, axis=-1, keepdims=True)
        trans_img = adjust_clothes_color(trans_img, clothes_pixels)

    return trans_img

def do_train_stage1_v2(args,
                    dataset,
                    model,
                    optimizer,
                    scheduler
                    ):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalizer,
    ])

    device = "cuda"
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    #
    with torch.no_grad():

        print("==> Extract IR features")
        dataset.ir_cluster = True
        dataset.rgb_cluster = False
        cluster_loader_ir = get_cluster_loader(dataset, args.test_batch_size, args.workers)
        features_ir, labels_ir = extract_features_clip_ir(model, cluster_loader_ir, modal=2, get_image=True)
        features_ir = torch.cat([features_ir[path].unsqueeze(0) for path in dataset.train_thermal_path], 0).cuda()
        labels_ir = torch.cat([labels_ir[path].unsqueeze(0) for path in dataset.train_thermal_path], 0)

    del  cluster_loader_ir
    # labels_ir = [torch.tensor([1]) for i in range(40000)]
    # features_ir = [torch.tensor([1]) for i in range(40000)]
    # labels_ir = torch.cat(labels_ir, dim=-1)
    # features_ir = torch.cat(features_ir, dim=-1)

    nums_rgb = len(dataset.train_color_label)
    nums_ir = len(dataset.train_thermal_label)

    start_time = time.monotonic()

    dataset.ir_cluster = False
    dataset.rgb_cluster = True
    dataloader_rgb = get_cluster_loader(dataset, args.stage1_batch_size, args.workers)

    for epoch in range(1, args.stage1_maxepochs + 1):
        scheduler.step(epoch)
        model.train()

        iter_list_ir = torch.cat([torch.randperm(nums_ir),torch.randint(0,nums_ir,(nums_rgb-nums_ir,))],dim=0).to(device)

        batch = args.stage1_batch_size
        i_ter = len(iter_list_ir) // batch

        print('-----len of rgb and ir iter_list------', len(iter_list_ir), len(iter_list_ir))
        print('---------------------------------------------------------------------')
        print("the learning rate is ", optimizer.state_dict()['param_groups'][0]['lr'])
        print('---------------------------------------------------------------------')

        loss_meter = AverageMeter()

        for i, (images, labels, parsing) in enumerate(dataloader_rgb):
            optimizer.zero_grad()
            if i != i_ter:
                b_list_ir = iter_list_ir[i * batch:(i + 1) * batch]
            else:
                b_list_ir = iter_list_ir[i * batch:]

            target_ir = labels_ir[b_list_ir]
            image_features_ir = features_ir[b_list_ir]


            # image_features_rgb = features_rgb[b_list_rgb]
            target_rgb = labels.to(device)
            images_rgb = images.to(device)
            image_parsing = parsing.to(device)

            trans_imgs = []
            imgs_rgb = []
            # images_rgb = images_rgb.permute(0,2,3,1)
            for idx, (trans_img, parse) in enumerate(zip(images_rgb, image_parsing)):
                init_img = trans_img.clone()
                imgs_rgb.append(transform_test(init_img.permute(2,0,1)))
                trans_img = trans_color2(trans_img, parse, trans_color_list)
                trans_img = transform_test(trans_img)
                trans_imgs.append(trans_img)

            images_rgb = imgs_rgb
            images_rgb = torch.stack(images_rgb)  # 将tensor堆叠成一个4D tensor
            images_rgb = images_rgb.to(device)


            trans_imgs = torch.stack(trans_imgs)  # 将tensor堆叠成一个4D tensor
            trans_imgs = trans_imgs.to(device)


            # images_rgb = transform_test(images_rgb)
            # trans_imgs = transform_test(trans_imgs)

            # images_rgb = images_rgb.permute(0, 3, 1, 2)
            # trans_imgs = images_rgb.permute(0, 3, 1, 2)

            with torch.no_grad():
                trans_feature = model(trans_imgs, trans_imgs, modal=1, get_image=True)
                image_features_rgb = model(images_rgb, images_rgb, modal=1, get_image=True)

            text_features_rgb = model(get_text=True, label=target_rgb, modal=1)
            text_features_ir = model(get_text=True, label=target_ir, modal=2)
            loss_i2t_rgb = (xent(image_features_rgb, text_features_rgb, target_rgb, target_rgb) + xent(trans_feature, text_features_rgb, target_rgb, target_rgb)) / 2
            loss_t2i_rgb = (xent(text_features_rgb, image_features_rgb, target_rgb, target_rgb) + xent(text_features_rgb, trans_feature, target_rgb, target_rgb)) / 2

            loss_i2t_ir = xent(image_features_ir, text_features_ir, target_ir, target_ir)
            loss_t2i_ir = xent(text_features_ir, image_features_ir, target_ir, target_ir)

            loss = loss_i2t_rgb + loss_t2i_rgb + loss_i2t_ir + loss_t2i_ir

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item())

            torch.cuda.synchronize()
            if i % 100 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss_prompt: {:.3f}, Base Lr: {:.2e}"
                      .format(epoch, (i + 1), i_ter + 1,
                              loss_meter.avg, scheduler._get_lr(epoch)[0]))

            # if epoch % args.stage1_checkpoint == 0:
            #     torch.save(model.state_dict(), os.path.join(args.model_path, args.logs_file + '_stage1_{}.pth'.format(epoch)))

        if epoch == args.stage1_maxepochs:
            # if True:
            if args.dataset == 'sysu':
                torch.save(model.state_dict(), os.path.join(args.model_path, args.logs_file + "_stage1.pth"))
                print('Test Epoch: {}'.format(epoch))
                test_mode = [1, 2]
                query_img, query_label, query_cam = process_query_sysu(args.data_path, mode=args.mode)
                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False,
                                               num_workers=args.workers)

                for trial in range(10):
                    # print('-------test trial {}-------'.format(trial))
                    gall_img, gall_label, gall_cam = process_gallery_sysu(args.data_path, mode=args.mode, trial=trial)
                    gallset = TestData(gall_img, gall_label, transform=transform_test,
                                       img_size=(args.img_w, args.img_h))
                    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=args.workers)

                    cmc, mAP, mINP = tester(args, epoch, model, test_mode, gall_label, gall_loader, query_label,
                                            query_loader,
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

                state = {
                    "state_dict": model.state_dict(),
                    "cmc": cmc,
                    "mAP": mAP,
                    "mINP": mINP,
                    "epoch": epoch,
                }
                torch.save(state, os.path.join(args.model_path, args.logs_file + "_stage1_add.pth"))
            elif args.dataset == 'regdb':
                state = {
                    "state_dict": model.state_dict(),
                    "cmc": cmc,
                    "mAP": mAP,
                    "mINP": mINP,
                    "epoch": epoch,
                }
                torch.save(state, os.path.join(args.model_path, args.logs_file + "_stage1_regdb.pth"))
                print('Test Epoch: {}'.format(epoch))

                query_img, query_label = process_test_regdb(img_dir=args.data_path, trial=args.trial, modal='visible')
                gall_img, gall_label = process_test_regdb(img_dir=args.data_path, trial=args.trial, modal='thermal')

                test_mode = [2, 1]
                gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

                # testing data loader
                gall_loader = data.DataLoader(gallset, batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers)
                query_loader = data.DataLoader(queryset, batch_size=args.test_batch_size, shuffle=False,
                                               num_workers=args.workers)

                cmc, mAP, mINP = tester(args, epoch, model, test_mode, gall_label, gall_loader, query_label,
                                        query_loader,
                                        feat_dim=3072)

                print(
                    "Performance[ALL]: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
                        cmc[0], cmc[4],
                        cmc[9], cmc[19],
                        mAP, mINP))

                # state = {
                #     "state_dict": model.state_dict(),
                #     "cmc": cmc,
                #     "mAP": mAP,
                #     "mINP": mINP,
                #     "epoch": epoch,
                # }
                # torch.save(state, os.path.join(args.model_path, args.logs_file + "_stage1_regdb.pth"))

            else:
                print('please input correct dataset!!')

    end_time = time.monotonic()
    print('Stage1 running time: ', timedelta(seconds=end_time - start_time))









