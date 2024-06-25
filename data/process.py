import numpy as np
from PIL import Image
import pdb
import os
import cv2

data_path = 'E:/hhj/SYSU-MM01/'

rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cameras = ['cam3', 'cam6']

# load id info
file_path_train = os.path.join(data_path, 'exp/train_id.txt')
file_path_val = os.path.join(data_path, 'exp/val_id.txt')
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]

with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]

# combine train and val split
id_train.extend(id_val)

files_rgb = []
files_ir = []
files_parsing = []
for id in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)

    for cam in ir_cameras:
        img_dir = os.path.join(data_path, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)

files_all = files_rgb.copy()
files_all.extend(files_ir)

def path_trans(path):
    path = path.replace('\\', '/')
    path_list = path.split('/')
    new_path_list = path_list[:]
    new_path_list[2] = 'SYSU-MM01-output'
    new_path_list[-1] = 'rgb_' + new_path_list[-1]
    new_path_list[-1] = new_path_list[-1].replace('jpg', 'png')
    new_path = '/'.join(new_path_list)
    return new_path

files_parsing = [path_trans(path) for path in files_ir]

# relabel
pid_container = set()
for img_path in files_ir:
    pid = int(img_path[-13:-9])
    pid_container.add(pid)
pid2label = {pid: label for label, pid in enumerate(pid_container)}
fix_image_width = 144
fix_image_height = 288


def read_imgs(train_image):
    train_img = []
    train_label = []
    train_path = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.Resampling.LANCZOS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)

        # path
        train_path.append(img_path)
    return np.array(train_img), np.array(train_label), np.array(train_path)


def read_imgs_parsing(train_image):
    train_img = []
    for img_path in train_image:
        # img
        img = Image.open(img_path).convert('RGB')
        img = img.resize((fix_image_width, fix_image_height), Image.Resampling.LANCZOS)
        pix_array = np.array(img)

        train_img.append(pix_array)

    return np.array(train_img)
# parsing images
train_img = read_imgs_parsing(files_parsing)

import matplotlib.pyplot as plt

# while True:
#     idx = int(input('idx: '))
#     img_rgb = cv2.imread(files_rgb[idx])
#     img_parsing = cv2.imread(files_parsing[idx])
#     print(img_rgb.shape)
#     print(img_rgb[64,64,:])
#     print(img_parsing[64,64,:])
#
#     img_rgb = Image.open(files_rgb[idx])
#     img_parse = Image.open(files_parsing[idx]).convert('RGB')
#     plt.figure()
#     plt.imshow(img_parse)
#     plt.show()
#
#     pix_rgb = np.array(img_rgb)
#     pix_parse = np.array(img_parse)
#     print(pix_rgb.shape)
#     print(pix_parse.shape)




    # break

np.save(data_path + 'train_parsing_img_ir.npy', train_img)

# # rgb imges
# train_img, train_label, train_path = read_imgs(files_rgb)
# np.save(data_path + 'train_rgb_resized_img.npy', train_img)
# np.save(data_path + 'train_rgb_resized_label.npy', train_label)
# np.save(data_path + 'train_rgb_resized_path.npy', train_path)
# #
# # # ir imges
# train_img, train_label, train_path = read_imgs(files_ir)
# np.save(data_path + 'train_ir_resized_img.npy', train_img)
# np.save(data_path + 'train_ir_resized_label.npy', train_label)
# np.save(data_path + 'train_ir_resized_path.npy', train_path)

# #all images
# train_img, train_label, train_path = read_imgs(files_all)
# np.save(data_path + 'train_all_resized_img.npy', train_img)
# np.save(data_path + 'train_all_resized_label.npy', train_label)
# np.save(data_path + 'train_all_resized_path.npy', train_path)