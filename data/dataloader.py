import os
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import collections
import cv2
import time
from datetime import timedelta

def read_image(data_files, img_w, img_h):
    train_img = []
    for img_path in data_files:
        # img
        img = Image.open(img_path)
        img = img.resize((img_w, img_h), Image.LANCZOS)
        pix_array = np.array(img)

        train_img.append(pix_array)

    return np.array(train_img)

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
    使用偏移量randon_hue来改变色调值，如果色调值超过180度，则将其反转。
    """
    # 将图像从RGB转换到HSV颜色空间
    image_hsv = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2HSV)

    # 生成一个随机色调
    random_hue = np.random.randint(0, 360)  # HSV色调范围是[0, 179]

    # 获得mask的布尔索引，扩展维度以适应HSV图像
    mask_boolean = np.squeeze(mask_numpy.astype(bool))
    mask_hsv = np.zeros_like(image_hsv, dtype=bool)
    mask_hsv[:, :, 0] = mask_boolean

    # 计算添加random_hue后的色调值
    hues = (image_hsv[mask_hsv] + random_hue) % 360
    # 计算“撞墙反弹”后的色调值
    bounced_hues = 180 - (hues - 180)
    # 使用where函数来选择色调值
    image_hsv[mask_hsv] = np.where(hues <= 180, hues, bounced_hues)

    # HSV颜色空间转换回RGB
    adjusted_image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    return adjusted_image_rgb


def trans_color(trans_img, parsing_img, clothe_color_list):
    # parsing_img = parsing_img.cpu().numpy()
    # trans_img = trans_img.cpu().numpy()
    for clothe_color in clothe_color_list:
        clothes_pixels = np.all(parsing_img == clothe_color, axis=-1, keepdims=True)
        trans_img = adjust_clothes_color(trans_img, clothes_pixels)

    return trans_img

def mask_background(image, parsing, background_color=np.array([0, 0, 0])):
    # 将解析图中的背景（颜色为[0,0,0]）找出来
    # background_color = background_color.view(1, 1, 1, -1).expand(*image.shape)
    background_pixels = np.all(parsing == background_color, axis=-1)
    # background_pixels = background_pixels.expand(-1, -1, -1, 3)
    # 将背景部分设置为全黑（或其他你想要的颜色）
    image[background_pixels] = np.array([0, 0, 0])

    return image


def mask_outlier(pseudo_labels):
    """
    Mask outlier data of clustering results.
    """
    index2label = collections.defaultdict(int)
    for label in pseudo_labels:
        index2label[label.item()] += 1
    nums = np.fromiter(index2label.values(), dtype=float)
    labels = np.fromiter(index2label.keys(), dtype=float)
    train_labels = labels[nums >= 1]

    return np.array([i in train_labels and i != -1 for i in pseudo_labels])


class SYSUData_Stage1(data.Dataset):
    def __init__(self,data_dir,transform,rgb_cluster=False,ir_cluster=False):
        self.train_color_image = np.load(data_dir+'train_rgb_resized_img.npy')
        self.train_thermal_image = np.load(data_dir+'train_ir_resized_img.npy')

        self.train_color_label = np.load(data_dir+'train_rgb_resized_label.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        self.train_parsing_image = np.load(data_dir + 'train_parsing_img.npy')

        self.train_color_path = np.load(data_dir + 'train_rgb_resized_path.npy')
        self.train_thermal_path = np.load(data_dir + 'train_ir_resized_path.npy')

        self.transform = transform
        self.ir_cluster = ir_cluster
        self.rgb_cluster = rgb_cluster

    def __getitem__(self, index):
        if self.rgb_cluster:
            img1, target1 = self.train_color_image[index], self.train_color_label[index]
            # img1 = self.transform(img1)
            return img1, target1, self.train_parsing_image[index]

        elif self.ir_cluster:
            img2, target2, path2 = self.train_thermal_image[index], self.train_thermal_label[index], self.train_thermal_path[index]
            img2 = self.transform(img2)
            return img2, target2, path2
        else:
            print('error getitem!')


    def __len__(self):
        if self.ir_cluster:
            return len(self.train_thermal_image)

        elif self.rgb_cluster:
            return len(self.train_color_image)
        else:
            print("error len!!")


trans_idx = [1,3,5,6,7,8,9,10,11,12,18,19]
palette = get_palette(20)
trans_color_list = [np.array(palette[3*i:3*i + 3]) for i in trans_idx]

class SYSUData_Stage1_V2(data.Dataset):
    def __init__(self,data_dir,transform,rgb_cluster=False,ir_cluster=False):
        self.train_color_image = np.load(data_dir+'train_rgb_resized_img.npy')
        self.train_thermal_image = np.load(data_dir+'train_ir_resized_img.npy')

        self.train_color_label = np.load(data_dir+'train_rgb_resized_label.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        self.train_parsing_image_rgb = np.load(data_dir + 'train_parsing_img_rgb.npy')
        self.train_parsing_image_ir = np.load(data_dir + 'train_parsing_img_ir.npy')


        for i in range(len(self.train_color_image)):
            self.train_color_image[i] = mask_background(self.train_color_image[i], self.train_parsing_image_rgb[i])

        for i in range(len(self.train_thermal_image)):
            self.train_thermal_image[i] = mask_background(self.train_thermal_image[i], self.train_parsing_image_ir[i])


        # self.train_color_path = np.load(data_dir + 'train_rgb_resized_path.npy')
        self.train_thermal_path = np.load(data_dir + 'train_ir_resized_path.npy')

        self.transform = transform
        self.ir_cluster = ir_cluster
        self.rgb_cluster = rgb_cluster

    def __getitem__(self, index):
        if self.rgb_cluster:
            img1, target1 = self.train_color_image[index], self.train_color_label[index]
            img1 = mask_background(img1, self.train_parsing_image_rgb[index])
            img1 = trans_color(img1, self.train_parsing_image_rgb[index], trans_color_list)
            # img1 = self.transform(img1)
            return img1, target1, self.train_parsing_image_rgb[index]

        elif self.ir_cluster:
            img2, target2, path2 = self.train_thermal_image[index], self.train_thermal_label[index], self.train_thermal_path[index]
            # img2 = mask_background(img2, self.train_parsing_image_ir[index])
            img2 = self.transform(img2)
            return img2, target2, path2
        else:
            print('error getitem!')


    def __len__(self):
        if self.ir_cluster:
            return len(self.train_thermal_image)

        elif self.rgb_cluster:
            return len(self.train_color_image)
        else:
            print("error len!!")


# class SYSUData_Stage2(data.Dataset):
#     def __init__(self, data_dir, transform_train_rgb=None, transform_train_ir=None, colorIndex=None, thermalIndex=None):
#         # Load training images (path) and labels
#
#         self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')
#         self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
#
#         self.train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
#         self.train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
#
#         # self.train_parsing_image = np.load(data_dir + 'train_parsing_img.npy')
#
#         self.transform_train_rgb = transform_train_rgb
#         self.transform_train_ir = transform_train_ir
#         self.cIndex = colorIndex
#         self.tIndex = thermalIndex
#
#     def __getitem__(self, index):
#         img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
#         img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
#         img1 = self.transform_train_rgb(img1)
#         img2 = self.transform_train_ir(img2)
#
#         return img1, img2, target1, target2
#
#     def __len__(self):
#         return len(self.train_color_label)

class SYSUData_Stage2(data.Dataset):
    def __init__(self, data_dir,transform_train_rgb=None, transform_train_ir=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels

        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        self.train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')

        self.train_parsing_image = np.load(data_dir + 'train_parsing_img.npy')

        self.transform_train_rgb = transform_train_rgb
        self.transform_train_ir = transform_train_ir
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        return img1, img2, target1, target2, self.train_parsing_image[self.cIndex[index]]

    def __len__(self):
        return len(self.train_color_label)


class SYSUData_Stage2_V2(data.Dataset):
    def __init__(self, data_dir,transform_train_rgb=None, transform_train_ir=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels

        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        self.train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')

        self.train_parsing_image_rgb = np.load(data_dir + 'train_parsing_img_rgb.npy')
        self.train_parsing_image_ir = np.load(data_dir + 'train_parsing_img_ir.npy')


        self.transform_train_rgb = transform_train_rgb
        self.transform_train_ir = transform_train_ir
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1 = mask_background(img1, self.train_parsing_image_rgb[self.cIndex[index]])
        img2 = mask_background(img2, self.train_parsing_image_ir[self.tIndex[index]])

        trans_img1 = trans_color(img1, self.train_parsing_image_rgb[self.cIndex[index]], trans_color_list)
        img1 = self.transform_train_rgb(img1)
        img2 = self.transform_train_ir(img2)
        trans_img1 = self.transform_train_rgb(trans_img1)

        return img1, img2, target1, target2, trans_img1
                # , self.train_parsing_image_rgb[self.cIndex[index]], self.train_parsing_image_ir[self.tIndex[index]]

    def __len__(self):
        return len(self.train_color_label)



class Unlabeld_RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform, rgb_cluster=False, ir_cluster=False):
        # Load training images (path) and labels
        data_dir = data_dir
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        train_color_path = []
        for i in range(len(color_img_file)):
            img_path = os.path.join(data_dir, color_img_file[i])
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
            train_color_path.append(img_path)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        train_thermal_path = []
        for i in range(len(thermal_img_file)):
            img_path = os.path.join(data_dir, thermal_img_file[i])
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            train_thermal_path.append(img_path)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label
        self.train_color_path = train_color_path
        self.color_img_file = color_img_file

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        self.train_thermal_path = train_thermal_path
        self.thermal_img_file = thermal_img_file

        self.transform = transform
        self.rgb_cluster = rgb_cluster
        self.ir_cluster = ir_cluster


    def __getitem__(self, index):
        if self.rgb_cluster:
            img1, target1, path1 = self.train_color_image[index], self.train_color_label[index], self.train_color_path[index]
            img1 = self.transform(img1)
            return img1, target1, path1, "RGB"

        elif self.ir_cluster:
            img2, target2, path2 = self.train_thermal_image[index], self.train_thermal_label[index], self.train_thermal_path[index]
            img2 = self.transform(img2)
            return img2, target2, path2, "IR"

        else:
            print('error getitem!')

    def __len__(self):
        if self.rgb_cluster:
            return len(self.train_color_image)

        elif self.ir_cluster:
            return len(self.train_thermal_image)

        else:
            print('error len!')

class Unlabeld_RegDBData_Pseudo(data.Dataset):
    def __init__(self, data_dir, trial, transform, rgb_cluster=False, ir_cluster=False):
        # Load training images (path) and labels
        data_dir = data_dir
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, _ = load_data(train_color_list)
        thermal_img_file, _ = load_data(train_thermal_list)

        train_color_image = []
        train_color_path = []
        for i in range(len(color_img_file)):
            img_path = os.path.join(data_dir, color_img_file[i])
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
            train_color_path.append(img_path)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        train_thermal_path = []
        for i in range(len(thermal_img_file)):
            img_path = os.path.join(data_dir, thermal_img_file[i])
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            train_thermal_path.append(img_path)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = np.load(data_dir+'train_rgb_resized_pseudo_label.npy')
        self.train_color_path = train_color_path
        self.color_img_file = color_img_file

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = np.load(data_dir+'train_ir_resized_pseudo_label.npy')
        self.train_thermal_path = train_thermal_path
        self.thermal_img_file = thermal_img_file

        self.transform = transform
        self.rgb_cluster = rgb_cluster
        self.ir_cluster = ir_cluster


    def __getitem__(self, index):
        if self.rgb_cluster:
            img1, target1, path1 = self.train_color_image[index], self.train_color_label[index], self.train_color_path[index]
            img1 = self.transform(img1)
            return img1, target1, path1, "RGB"

        elif self.ir_cluster:
            img2, target2, path2 = self.train_thermal_image[index], self.train_thermal_label[index], self.train_thermal_path[index]
            img2 = self.transform(img2)
            return img2, target2, path2, "IR"

        else:
            print('error getitem!')

    def __len__(self):
        if self.rgb_cluster:
            return len(self.train_color_image)

        elif self.ir_cluster:
            return len(self.train_thermal_image)

        else:
            print('error len!')


class RegDBData_Stage0(data.Dataset):
    def __init__(self, data_dir, trial, pseudo_labels_rgb=None, pseudo_labels_ir=None, transform_train_rgb=None, transform_train_ir=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        data_dir = data_dir
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        train_color_path = []
        for i in range(len(color_img_file)):
            img_path = os.path.join(data_dir, color_img_file[i])
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
            train_color_path.append(img_path)
        train_color_image = np.array(train_color_image)
        train_color_path = np.array(train_color_path)

        train_thermal_image = []
        train_thermal_path = []
        for i in range(len(thermal_img_file)):
            img_path = os.path.join(data_dir, thermal_img_file[i])
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            train_thermal_path.append(img_path)
        train_thermal_image = np.array(train_thermal_image)
        train_thermal_path = np.array(train_thermal_path)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_pseudo_label = np.array(pseudo_labels_rgb)
        # self.train_color_label = train_color_label
        self.train_color_path = train_color_path

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_pseudo_label = np.asarray(pseudo_labels_ir)
        # self.train_thermal_label = train_thermal_label
        self.train_thermal_path = train_thermal_path

        mask_color = mask_outlier(self.train_color_pseudo_label)
        self.train_color_image = self.train_color_image[mask_color]
        self.train_color_pseudo_label = self.train_color_pseudo_label[mask_color]
        self.train_color_path = self.train_color_path[mask_color]

        mask_thermal = mask_outlier(self.train_thermal_pseudo_label)
        self.train_thermal_image = self.train_thermal_image[mask_thermal]
        self.train_thermal_pseudo_label = self.train_thermal_pseudo_label[mask_thermal]
        self.train_thermal_path = self.train_thermal_path[mask_thermal]

        self.transform_train_rgb = transform_train_rgb
        self.transform_train_ir = transform_train_ir
        self.cIndex = colorIndex
        self.tIndex = thermalIndex


    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_pseudo_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_pseudo_label[self.tIndex[index]]
        img1 = self.transform_train_rgb(img1)
        img2 = self.transform_train_ir(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
    
class RegDBData_Stage2(data.Dataset):
    def __init__(self, data_dir, trial, pseudo_labels_rgb=None, pseudo_labels_ir=None, transform_train_rgb=None, transform_train_ir=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        data_dir = data_dir
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, _ = load_data(train_color_list)
        thermal_img_file, _ = load_data(train_thermal_list)

        train_color_image = []
        train_color_path = []
        for i in range(len(color_img_file)):
            img_path = os.path.join(data_dir, color_img_file[i])
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
            train_color_path.append(img_path)
        train_color_image = np.array(train_color_image)
        train_color_path = np.array(train_color_path)

        train_thermal_image = []
        train_thermal_path = []
        for i in range(len(thermal_img_file)):
            img_path = os.path.join(data_dir, thermal_img_file[i])
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            train_thermal_path.append(img_path)
        train_thermal_image = np.array(train_thermal_image)
        train_thermal_path = np.array(train_thermal_path)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_pseudo_label = np.array(pseudo_labels_rgb)
        self.train_color_label = np.load(data_dir+'pseudo_labels/'+'train_rgb_resized_pseudo_label.npy')
        self.train_color_path = train_color_path

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_pseudo_label = np.asarray(pseudo_labels_ir)
        self.train_thermal_label = np.load(data_dir+'pseudo_labels/'+'train_ir_resized_pseudo_label.npy')
        self.train_thermal_path = train_thermal_path

        mask_color = mask_outlier(self.train_color_pseudo_label)
        self.train_color_image = self.train_color_image[mask_color]
        self.train_color_pseudo_label = self.train_color_pseudo_label[mask_color]
        self.train_color_label = self.train_color_label[mask_color]
        self.train_color_path = self.train_color_path[mask_color]

        mask_thermal = mask_outlier(self.train_thermal_pseudo_label)
        self.train_thermal_image = self.train_thermal_image[mask_thermal]
        self.train_thermal_pseudo_label = self.train_thermal_pseudo_label[mask_thermal]
        self.train_thermal_label = self.train_thermal_label[mask_thermal]
        self.train_thermal_path = self.train_thermal_path[mask_thermal]

        self.transform_train_rgb = transform_train_rgb
        self.transform_train_ir = transform_train_ir
        self.cIndex = colorIndex
        self.tIndex = thermalIndex


    def __getitem__(self, index):

        img1, target1, target1_old = self.train_color_image[self.cIndex[index]], self.train_color_pseudo_label[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2, target2_old = self.train_thermal_image[self.tIndex[index]], self.train_thermal_pseudo_label[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        img1 = self.transform_train_rgb(img1)
        img2 = self.transform_train_ir(img2)

        return img1, img2, target1, target2, target1_old, target2_old

    def __len__(self):
        return len(self.train_color_label)


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length

        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label
