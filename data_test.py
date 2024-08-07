import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from data.dataloader import SYSUData_Stage2, RegDBData_Stage2, IterLoader, TestData, SYSUData_Stage2_V2
from util.utils import IdentitySampler_nosk, GenIdx, IdentitySampler_nosk_stage4
import matplotlib.pyplot as plt
import numpy as np
from util.transforms import RGB_HSV, RandomColoring, RandomColoring_tensor
from util.trans_function import RGB2HSV, HSV2RGB

rgb_hsv = RGB_HSV()
device = torch.device('cuda')

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train_rgb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Pad(10),
    transforms.RandomCrop((288, 144)),
    transforms.ToTensor(),
    normalizer,
    transforms.RandomErasing(p=0.5)
])
transform_train_ir = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Pad(10),
    transforms.RandomCrop((288, 144)),
    transforms.ToTensor(),
    normalizer,
    transforms.RandomErasing(p=0.5),
])
transform_test_rgb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((288, 144)),
    transforms.ToTensor(),
    RandomColoring(p=0.5,is_rgb=True),
    # normalizer,
])
transform_test_rgb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((288, 144)),
    transforms.ToTensor(),
    # RandomColoring_tensor(p=0.5,is_rgb=True),
    # normalizer,
])
transform_test_ir = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((288, 144)),
    transforms.ToTensor(),
    # RandomColoring_tensor(p=0.5,is_rgb=False),
    # normalizer,
])

trainset = SYSUData_Stage2_V2('E:/hhj/SYSU-MM01/', transform_test_rgb, transform_test_ir)

color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

sampler = IdentitySampler_nosk_stage4(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos,
                               2, 2)

trainset.cIndex = sampler.index1
trainset.tIndex = sampler.index2

trainloader = data.DataLoader(trainset, batch_size=64, sampler=sampler,
                              num_workers=0,
                              drop_last=True)

def imshow(img):
    img = img.cpu()
    plt.imshow(img)

def adjust_clothes_color(image_numpy, mask_numpy):
    """
    根据提供的mask随机改变衣服颜色并保持亮度不变。
    """
    # 将图像从RGB转换到HSV颜色空间
    image_hsv = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2HSV)
    # image_hsv = RGB2HSV(image_numpy)
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
    # print(image_hsv.shape)
    # adjusted_image_rgb = HSV2RGB(image_hsv)
    # print(adjusted_image_rgb.shape)
    return adjusted_image_rgb


def trans_color(trans_img, parsing_img, clothe_color_list):
    parsing_img = parsing_img.cpu().numpy()
    trans_img = trans_img.cpu().numpy()
    for clothe_color in clothe_color_list:
        clothes_pixels = np.all(parsing_img == clothe_color, axis=-1)
        trans_img = adjust_clothes_color(trans_img, clothes_pixels)

    return trans_img


def mask_background(image, parsing, background_color=torch.tensor([0, 0, 0], device=device)):
    # 将解析图中的背景（颜色为[0,0,0]）找出来
    # background_color = background_color.view(1, 1, 1, -1).expand(*image.shape)
    background_pixels = torch.all(parsing == background_color, dim=-1)
    # background_pixels = background_pixels.expand(-1, -1, -1, 3)
    # 将背景部分设置为全黑（或其他你想要的颜色）
    image[background_pixels] = torch.tensor([0, 0, 0], device=device, dtype=torch.uint8)

    return image

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

cvt = RGB_HSV()

for n_iter, (img_rgb, img_ir, label_rgb, label_ir, trans_img) in enumerate(trainloader):
    img_rgb = img_rgb.to(device)
    img_ir = img_ir.to(device)
    trans_img = trans_img.to(device)

    hsv_rgb = cvt.rgb_to_hsv(img_rgb)
    hsv_ir = cvt.rgb_to_hsv(img_ir)
    hsv_trans = cvt.rgb_to_hsv(trans_img)

    rgbs = []
    irs = []
    transs = []

    for rgb, ir, trans_c in zip(hsv_rgb, hsv_ir, hsv_trans):
        # rgbs.append(RandomColoring_tensor(p=0.5, is_rgb=True)(rgb))
        irs.append(RandomColoring_tensor(p=0.5, is_rgb=False)(ir))
        transs.append(RandomColoring_tensor(p=0.5,is_rgb=True)(trans_c))

    # rgbs = torch.stack(rgbs, dim=0)
    irs = torch.stack(irs, dim=0)
    transs = torch.stack(transs, dim=0)

    # img_rgb = cvt.hsv_to_rgb(rgbs)
    img_ir = cvt.hsv_to_rgb(irs)
    trans_img = cvt.hsv_to_rgb(transs)

    # trans_img = mask_background(img_rgb, img_parsing)
    # for color in trans_color_list:
    #     color = torch.tensor(color, device=device, dtype=torch.uint8)
        # trans_img = trans_color(trans_img,img_parsing, color)
    # trans_img = trans_color(trans_img, img_parsing)
    # print(img_rgb.shape, img_ir.shape)
    img_rgb = img_rgb.permute(0, 2, 3, 1)
    trans_img = trans_img.permute(0, 2, 3, 1)
    img_ir = img_ir.permute(0, 2, 3, 1)
    # trans_img = img_rgb.clone()
    # trans_img = trans_img.permute(0, 3, 1, 2)
    # trans_img = trans_img / 255.0
    # print(img_rgb.shape)
    # trans_img = rgb_hsv.rgb_to_hsv(trans_img)
    # trans_img = rgb_hsv.hsv_to_rgb(trans_img)
    # trans_img = (trans_img * 255).int()
    # print(trans_img.shape)
    # trans_img = trans_img.permute(0, 2, 3, 1)
    for rgb,ir,trans_c in zip(img_rgb, img_ir, trans_img):

        # trans_c = rgb.clone()
        # trans_c = rgb_hsv.rgb_to_hsv()
        # trans_c = rgb.clone()
        # trans_c = trans_color(trans_c, parsing, trans_color_list)
        # trans_c = torch.tensor(trans_c, device=device)
        # print(rgb.shape, parsing.shape)
        # rgb = rgb.permute(1, 2, 0)
        # trans_c = trans_c.permute(1, 2, 0)
        # ir =  ir.permute(1, 2, 0)

        # 在同一窗口中显示三张图像
        plt.figure(figsize=(10, 10))

        plt.subplot(1, 3, 1)
        imshow(rgb)
        plt.title('RGB')

        plt.subplot(1, 3, 2)
        imshow(trans_c)
        plt.title('trans_img')


        plt.subplot(1, 3, 3)
        imshow(ir)
        plt.title('IR')

        plt.show()
        input("Press Enter to continue...")
        # OpenCV需要将图像从RGB转换为BGR
        # rgb_bgr = cv2.cvtColor(rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)
        # parsing_bgr = cv2.cvtColor(parsing.cpu().numpy(), cv2.COLOR_RGB2BGR)
        # train_c_bgr = cv2.cvtColor(train_c.cpu().numpy(), cv2.COLOR_RGB2BGR)
        #
        # cv2.imshow('RGB', rgb_bgr)
        # cv2.imshow('Parsing', parsing_bgr)
        # cv2.imshow('Train_C', train_c_bgr)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()