import random
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.transforms.functional as F
import argparse
import pickle
import os
import time
from random import randint
from PIL import Image
import torchvision
from render_sketch_chairv2 import redraw_Quick2RGB


# 数据处理，从255resize到320然后中心裁剪为299
def get_complete_transform(opt):
    transform_list = []
    if opt.Train:
        transform_list.extend([transforms.Resize(320), transforms.CenterCrop(299)])
    else:
        transform_list.extend([transforms.Resize(299)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)


def get_part_transform(opt):
    transform_list = []
    if opt.Train:
        transform_list.extend([transforms.Resize(180), transforms.CenterCrop(170)])
    else:
        transform_list.extend([transforms.Resize(170)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)


# TODO: 可用矩阵split优化
# 分割图像函数
def split_img(img):
    splitimg = []
    size_img = img.size
    weight = int(size_img[0] // 3)
    height = int(size_img[1] // 3)
    for j in range(3):
        for k in range(3):
            box = (weight * k, height * j, weight * (k + 1), height * (j + 1))
            region = img.crop(box)
            splitimg.append(region)
    return splitimg


# 分割草图函数
def split_img_sketch(img_sketch):
    all =[]
    for img in img_sketch:
        splitimg =[]
        size_img = img.size
        weight = int(size_img[0] // 3)
        height = int(size_img[1] // 3)
        for j in range(3):
            for k in range(3):
                box = (weight * k, height * j, weight * (k + 1), height * (j + 1))
                region = img.crop(box)
                splitimg.append(region)
        all.append(splitimg)
    return all


# 数据集类
class createDataset(data.Dataset):
    def __init__(self, opt, on_Fly=False):
        # 打开序列化的草图坐标文件
        with open(opt.coordinate, 'rb') as fp:
            self.Coordinate = pickle.load(fp)

        # 得到训练集和验证集的路径
        self.Skecth_Train_List = [x for x in self.Coordinate if 'train' in x]
        self.Skecth_Test_List = [x for x in self.Coordinate if 'test' in x]
        self.opt = opt
        # 定义类中数据处理函数
        self.complete_transform = get_complete_transform(opt)
        self.part_transform = get_part_transform(opt)
        self.on_Fly = on_Fly

    def __getitem__(self, item):
        global sample
        if self.opt.mode == 'Train':
            # item实际是一个索引值， Skecth_Train_List是草图训练集的图像相对路径
            sketch_path = self.Skecth_Train_List[item]
            # 从草图的相对路径切出草图名称，然后组成正样本绝对路径
            positive_sample = '_'.join(sketch_path.split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.opt.roor_dir, 'photo', positive_sample + '.png')

            # 负样本绝对路径
            possible_list = list(range(len(self.Skecth_Train_List)))
            possible_list.remove(item)
            
            negetive_item = possible_list[randint(0, len(possible_list) - 1)]
            negetive_sample = '_'.join(self.Skecth_Train_List[negetive_item].split('/')[-1].split('_')[:-1])
            negetive_path = os.path.join(self.opt.roor_dir, 'photo', negetive_sample + '.png')

            # 取得正样本、负样本,包括完整的和不完整的
            # 并按概率对正样本、负样本做水平翻转，提高鲁棒性
            n_flip = random.random()
            if n_flip < 0.5:
                positive_img = Image.open(positive_path)
                negetive_img = Image.open(negetive_path)
            else:
                positive_img = Image.open(positive_path).transpose(Image.FLIP_LEFT_RIGHT)
                negetive_img = Image.open(negetive_path).transpose(Image.FLIP_LEFT_RIGHT)
            # 完整图像已经翻转或未翻转，只需要做切割，切割出的就是翻转的或未翻转的
            positive_img_4 = split_img(positive_img)
            negetive_img_4 = split_img(negetive_img)
            positive_img = self.complete_transform(positive_img)
            negetive_img = self.complete_transform(negetive_img)
            positive_img_4 = [self.part_transform(po_img) for po_img in positive_img_4]
            negetive_img_4 = [self.part_transform(ne_img) for ne_img in negetive_img_4]

            positive_img_4 = torch.cat(positive_img_4).view(-1, 3, 170, 170)
            negetive_img_4 = torch.cat(negetive_img_4).view(-1, 3, 170, 170)

  
            # 取得草图,包括所有的20张
            # 按概率对草图做水平翻转，提高鲁棒性
            vector_x = self.Coordinate[sketch_path]
            sketch_img, Sample_len = redraw_Quick2RGB(vector_x)

            if n_flip < 0.5:
                if not self.on_Fly:
                    sketch_img = Image.fromarray(sketch_img[-1]).convert('RGB')
                else:
                    sketch_img = [Image.fromarray(sk_img).convert('RGB') for sk_img in sketch_img]
            else:
                if not self.on_Fly:
                    sketch_img = Image.fromarray(sketch_img[-1]).convert('RGB').transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    sketch_img = [Image.fromarray(sk_img).convert('RGB').transpose(Image.FLIP_LEFT_RIGHT) for sk_img in
                                  sketch_img]

            # 取得完整和部分草图，且经过的transform函数
            if not self.on_Fly:
                sketch_img = self.complete_transform(sketch_img)
            else:
                # 对完整草图先Resize成320再中心裁剪成299
                sketch_img_complete = [self.complete_transform(sk_img) for sk_img in sketch_img]
                sketch_img_complete = torch.cat(sketch_img_complete).view(-1, 3, 299, 299)
                


                # 矩阵转tensor
                transform1 = transforms.Compose([
                    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                ])


                sketch_sum_all = [torch.sum(transform1(batch)) for batch in sketch_img]

                sketch_part_img = split_img_sketch(sketch_img)
                    
               
                sketch_img_part_all = []
                sum_all_d =[]

                for i in range (len(sketch_sum_all)):


                    sketch_part_sum_all = [torch.sum(transform1(batch)) for batch in sketch_part_img[i]]
                # 对部分草图先Resize成170再中心裁剪成149
                    sketch_img_part = [self.part_transform(sk_img) for sk_img in sketch_part_img[i]]
                    sketch_img_part = torch.cat(sketch_img_part).view(-1, 3, 170, 170)
                  
                    sketch_img_part_all.append(sketch_img_part)

                    d_1 = torch.div(sketch_part_sum_all[0], sketch_sum_all[i])
                    d_2 = torch.div(sketch_part_sum_all[1], sketch_sum_all[i])
                    d_3 = torch.div(sketch_part_sum_all[2], sketch_sum_all[i])
                    d_4 = torch.div(sketch_part_sum_all[3], sketch_sum_all[i])
                    d_5 = torch.div(sketch_part_sum_all[4], sketch_sum_all[i])
                    d_6 = torch.div(sketch_part_sum_all[5], sketch_sum_all[i])
                    d_7 = torch.div(sketch_part_sum_all[6], sketch_sum_all[i])
                    d_8 = torch.div(sketch_part_sum_all[7], sketch_sum_all[i])
                    d_9 = torch.div(sketch_part_sum_all[8], sketch_sum_all[i])
                    d = [d_1, d_2, d_3, d_4,d_5,d_6,d_7,d_8,d_9]
                    sum_all_d.append(d)

            sample = {'sketch_img_complete': sketch_img_complete, 'sketch_img_part': sketch_img_part_all,
                      'sketch_path': sketch_path,
                      'positive_img': positive_img, 'positive_img_4': positive_img_4, 'positive_path': positive_sample,
                      'negetive_img': negetive_img, 'negetive_img_4': negetive_img_4, 'negetive_path': positive_sample,
                      'sum_all_d': sum_all_d}

        elif self.opt.mode == 'Test':
            sketch_path = self.Skecth_Test_List[item]
            positive_sample = '_'.join(sketch_path.split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.opt.roor_dir, 'photo', positive_sample + '.png')
            vector_x = self.Coordinate[sketch_path]
            # 256*256
            sketch_img, Sample_len = redraw_Quick2RGB(vector_x)
          
            text_pro = []

          
            sketch_img = [Image.fromarray(sk_img).convert('RGB') for sk_img in sketch_img]
            sketch_img_complete = [self.complete_transform(sk_img) for sk_img in sketch_img]
            sketch_img_complete = torch.cat(sketch_img_complete).view(-1, 3, 299, 299)
            # print('sketch_img_complete',sketch_img_complete.shape)

            transform1 = transforms.Compose([
                    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                ])
                # 得到草图所有数值综合\
            sketch_sum_all = [torch.sum(transform1(batch)) for batch in sketch_img]

            sketch_part_img = split_img_sketch(sketch_img)
           
            sketch_img_part_all = []

            sum_all_d =[]
            # print('-----------------------------')
            # print(len(sketch_part_img))

            for i in range (len(sketch_part_img)):
                sketch_img_part = [self.part_transform(sk_img) for sk_img in sketch_part_img[i]]
                sketch_img_part = torch.cat(sketch_img_part).view(-1, 3, 170, 170)
                sketch_img_part_all.append(sketch_img_part)

                sketch_part_sum_all = [torch.sum(transform1(batch)) for batch in sketch_part_img[i]]
                # 对部分草图先Resize成170再中心裁剪成149
                d_1 = torch.div(sketch_part_sum_all[0], sketch_sum_all[i])
                d_2 = torch.div(sketch_part_sum_all[1], sketch_sum_all[i])
                d_3 = torch.div(sketch_part_sum_all[2], sketch_sum_all[i])
                d_4 = torch.div(sketch_part_sum_all[3], sketch_sum_all[i])
                d_5 = torch.div(sketch_part_sum_all[4], sketch_sum_all[i])
                d_6 = torch.div(sketch_part_sum_all[5], sketch_sum_all[i])
                d_7 = torch.div(sketch_part_sum_all[6], sketch_sum_all[i])
                d_8 = torch.div(sketch_part_sum_all[7], sketch_sum_all[i])
                d_9 = torch.div(sketch_part_sum_all[8], sketch_sum_all[i])
                d = [d_1, d_2, d_3, d_4,d_5,d_6,d_7,d_8,d_9]
                sum_all_d.append(d)
            positive_img = self.complete_transform(Image.open(positive_path))
            positive_img_4 = split_img(Image.open(positive_path))
            positive_img_4 = [self.part_transform(po_img) for po_img in positive_img_4]
            positive_img_4 = torch.cat(positive_img_4).view(-1, 3, 170, 170)


            sample = {'sketch_img_complete': sketch_img_complete, 'sketch_img_part': sketch_img_part_all,
                      'sketch_path': self.Skecth_Test_List[item],
                      'positive_img': positive_img, 'positive_img_4': positive_img_4, 'positive_path': positive_sample,
                      'Sample_Len': Sample_len, 'sum_all_d': sum_all_d}
        return sample

    def __len__(self):
        if self.opt.mode == 'Train':
            return len(self.Skecth_Train_List)
        elif self.opt.mode == 'Test':
            return len(self.Skecth_Test_List)
