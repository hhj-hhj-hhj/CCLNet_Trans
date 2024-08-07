import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from data.data_manager import process_gallery_sysu
from data.dataloader import TestData

# torch.backends.cuda.max_split_size_mb = 2750

def get_cluster_loader(dataset, batch_size, workers):
    cluster_loader = data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False
        # , pin_memory=True
    )
    return cluster_loader


def t_SNE(args,
        model,
        ):


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

    text_features_rgb = text_features_rgb.cpu().numpy()
    text_features_ir = text_features_ir.cpu().numpy()

    len_features = len(text_features_rgb)

    draw_step = len_features // 50

    def draw_tsne(rgb, ir):
        tsne = TSNE(n_components=2, init='pca', random_state=42, n_iter=1000)
        all_features = np.concatenate((rgb, ir), axis=0)
        X_tsne = tsne.fit_transform(all_features)
        X_tsne_rgb = X_tsne[:len(rgb)]
        X_tsne_ir = X_tsne[len(rgb):]


        # 创建一个颜色列表，长度与特征点数量相同
        colors = plt.cm.rainbow(np.linspace(0, 1, len(rgb)))

        plt.figure(figsize=(10, 5))
        # 对于rgb特征，使用五角星形状，颜色从颜色列表中获取
        plt.scatter(X_tsne_rgb[:, 0], X_tsne_rgb[:, 1] + 0.5, c=colors, label='rgb', marker='*')

        # 对于ir特征，使用默认的圆形形状，颜色从颜色列表中获取
        plt.scatter(X_tsne_ir[:, 0], X_tsne_ir[:, 1], c=colors, label='ir', marker='s')

        plt.legend()
        plt.show()

        print('t-SNE finished!')

    draw_tsne(text_features_rgb, text_features_ir)
    # for i in range(0,draw_step + 1):
    #     if i == draw_step:
    #         draw_tsne(text_features_rgb[i * 50:], text_features_ir[i * 50:])
    #     else:
    #         draw_tsne(text_features_rgb[i * 50 :i * 50 + 50], text_features_ir[i * 50 :i * 50 + 50])
