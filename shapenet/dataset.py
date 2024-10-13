import torch
import json
from tqdm import tqdm
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


class ShapeNet(Dataset):
    def __init__(self, dataset_path, setting):
        self._root = dataset_path
        self._setting = setting
        self._points, self._pngs, self._prompt = self._get_shapenet_dataset(self._root, self._setting)

    def __len__(self):
        return len(self._prompt)

    @staticmethod
    def _get_shapenet_dataset(path, setting):
        f = open(path + f'/shape_data/train_test_split/shuffled_{setting}_file_list.json', 'r')
        # max_pts = 0
        content = f.read()
        path_list = json.loads(content)
        train_points = []
        train_png = []
        train_prompt = []
        for p in tqdm(path_list):
            folder, category, index = p.split('/')
            # pts = read_pts(path + '/' + folder + '/' + category + '/points' + '/' + index + '.pts')
            pts = path + '/' + folder + '/' + category + '/points' + '/' + index + '.pts'
            prompt = path + '/' + folder + '/' + category + '/prompt' + '/' + index + '.txt'
            png = path + '/' + folder + '/' + category + '/seg_img' + '/' + index + '.png'
            # num = pts.shape[0]
            # max_pts = max(num, max_pts)
            train_points.append(pts)
            train_png.append(png)
            # train_prompt.append(read_txt(prompt))
            train_prompt.append('')
            # train_png.append(read_png(png))
        # train_png = np.stack(train_png, axis=0)
        # train_points = np.stack(train_points, axis=0)
        # print(max_pts) # 2974
        return train_points, train_png, train_prompt

    def __getitem__(self, item):
        points = read_pts(self._points[item])
        pngs = read_png(self._pngs[item])
        # pngs = self._pngs[item]
        sample = {'points': points, 'seg_img': pngs, 'prompt': self._prompt[item]}
        return sample

def read_png(path):
    # img = cv.imread(path)
    # transf = transforms.ToTensor()
    # img_tensor = transf(img)

    image = Image.open(path)
    width, height = image.size
    output_size = (500, 500)
    padding_color = (255, 255, 255)

    target_ratio = output_size[0] / output_size[1]

    ratio = width / height


    # if ratio > target_ratio:
    #
    #     new_width = int(height * target_ratio)
    #     image = image.resize((new_width, height))
    # else:
    #
    #     new_height = int(width / target_ratio)
    #     image = image.resize((width, new_height))
    #
    #
    # new_image = Image.new('RGB', output_size, padding_color)

    if ratio > 1:
        pad_top_town = (width-height) // 2
        pad_left_right = 0
        new_image = Image.new('RGB', (width, width), padding_color)
    else:
        pad_left_right = (height - width) // 2
        pad_top_town = 0
        new_image = Image.new('RGB', (height, height), padding_color)

    # pad_left = (output_size[0] - width) // 2
    # pad_top = (output_size[1] - height) // 2
    new_image.paste(image, (pad_left_right, pad_top_town))
    new_image = new_image.resize((224, 224))
    new_image = np.array(new_image)
    condition = (new_image[:, :, 0] == 255) & (new_image[:, :, 1] == 255) & (new_image[:, :, 2] == 255)
    condition = np.stack([condition] * 3, axis=2)
    new_image = np.where(condition, 0, new_image)
    new_image = new_image.astype(np.float32) / 255.0
    # plt.imshow(new_image, cmap='viridis', interpolation='nearest')  # cmap参数可以指定颜色映射
    # plt.colorbar()  # 显示颜色条，以便知道颜色对应的数值
    # plt.show()  # 显示图像
    return new_image

def read_txt(path):
    with open(path, 'r', encoding='utf-8') as file:
        # 读取文件内容
        content = file.read()
    return content


def read_pts(path, max_points = 3072):
    # 2974
    points = []
    with open(path, 'r') as f:
        for line in f:
            x, y, z = line.split()
            x, y, z = float(x), float(y), float(z)
            points.append([x,y,z])
    points = np.array(points)
    pts_num = points.shape[0]
    pad_num = max_points - pts_num
    repeat_num = pad_num // pts_num + 2
    points = np.repeat(points, repeat_num, axis=0)[:max_points, :]
    return points

# def get_shapenet_dataset(path):
#     f = open(path+'/shape_data/train_test_split/shuffled_val_file_list.json', 'r')
#     content = f.read()
#     path_list = json.loads(content)
#     train_points = []
#     train_png = []
#     for p in tqdm(path_list):
#         folder, category, index = p.split('/')
#         train_points.append(read_pts(path+'/'+folder+'/'+category+'/points'+'/'+index+'.pts'))
#         train_png.append(read_png(path+'/'+folder+'/'+category+'/seg_img'+'/'+index+'.png'))
#     return train_points, train_png


def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''
    batch_data = {
        'input_lidar': [sample['points'] for sample in batch],
        'seg_img': [sample['seg_img'] for sample in batch]
    }

    return batch_data

def make_shapenet_loader(train_set, train_batch_size, num_workers, rng_generator):

    # dataloaders
    loader = torch.utils.data.DataLoader(train_set,pin_memory=False,collate_fn=None,drop_last=True,generator=rng_generator,
                                               batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    return loader

