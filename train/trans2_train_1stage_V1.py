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
from util.transforms import RGB_HSV, RandomColoring_tensor




def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False
        , pin_memory=True
    )
    return cluster_loader


def do_train_stage1_v3(args,
                    dataset,
                    model,
                    optimizer,
                    scheduler
                    ):

    device = "cuda"
    scaler = amp.GradScaler()
    xent = SupConLoss(device)


    start_time = time.monotonic()

    print("==> Start training")

    for epoch in range(1, args.stage1_maxepochs + 1):
        scheduler.step(epoch)
        model.train()

        dataset.resetIdx()
        dataset.rgb_cluster, dataset.ir_cluster = True, False
        rgb_dataloader = data.DataLoader(dataset,batch_size=args.test_batch_size, num_workers=args.workers, shuffle=True)

        dataset.rgb_cluster, dataset.ir_cluster = False, True
        ir_dataloader = data.DataLoader(dataset, batch_size=args.test_batch_size, num_workers=args.workers, shuffle=True)

        i_ter = len(rgb_dataloader)

        # print(len(ir_dataloader), len(rgb_dataloader))


        print('---------------------------------------------------------------------')
        print("the learning rate is ", optimizer.state_dict()['param_groups'][0]['lr'])
        print('---------------------------------------------------------------------')

        loss_meter_rgb = AverageMeter()
        loss_meter_ir = AverageMeter()

        for i, (rgb_batch, ir_batch) in enumerate(zip(rgb_dataloader,ir_dataloader)):
            optimizer.zero_grad()

            image_rgb, target_rgb = rgb_batch
            image_ir, target_ir = ir_batch

            image_rgb = image_rgb.to(device)
            image_ir = image_ir.to(device)

            target_rgb = target_rgb.to(device)
            target_ir = target_ir.to(device)

            with torch.no_grad():
                image_features_rgb = model(x1 = image_rgb, get_image = True, modal = 1)
                image_features_ir = model(x2 = image_ir, get_image = True, modal = 2)

            text_features_rgb = model(get_text=True, label=target_rgb, modal=1)
            text_features_ir = model(get_text=True, label=target_ir, modal=2)
            loss_i2t_rgb = xent(image_features_rgb, text_features_rgb, target_rgb, target_rgb)
            loss_t2i_rgb = xent(text_features_rgb, image_features_rgb, target_rgb, target_rgb)

            loss_i2t_ir = xent(image_features_ir, text_features_ir, target_ir, target_ir)
            loss_t2i_ir = xent(text_features_ir, image_features_ir, target_ir, target_ir)

            loss_rgb = loss_i2t_rgb + loss_t2i_rgb
            loss_ir = loss_i2t_ir + loss_t2i_ir
            loss = loss_rgb + loss_ir

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            loss_meter_rgb.update(loss_rgb.item())
            loss_meter_ir.update(loss_ir.item())


            torch.cuda.synchronize()
            if i % 100 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss_prompt_RGB_IR: {:.3f} {:.3f}, Base Lr: {:.2e}"
                      .format(epoch, (i + 1), i_ter + 1,
                              loss_meter_rgb.avg,loss_meter_rgb.avg, scheduler._get_lr(epoch)[0]))

            # if epoch % args.stage1_checkpoint == 0:
            #     torch.save(model.state_dict(), os.path.join(args.model_path, args.logs_file + '_stage1_{}.pth'.format(epoch)))

        if epoch == args.stage1_maxepochs:
            # if True:
            state = {
                "state_dict": model.state_dict(),
                "cmc": 0,
                "mAP": 0,
                "mINP": 0,
                "epoch": epoch,
            }
            if args.dataset == 'sysu':
                torch.save(state, os.path.join(args.model_path, args.logs_file + "_V3_stage1_V1.pth"))
            elif args.dataset == 'regdb':
                torch.save(state, os.path.join(args.model_path, args.logs_file + "_stage1_regdb.pth"))
            else:
                print('please input correct dataset!!')

    end_time = time.monotonic()
    print('Stage1 running time: ', timedelta(seconds=end_time - start_time))
