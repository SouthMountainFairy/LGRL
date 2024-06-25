from pyexpat import native_encoding
import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F
from rasterize import rasterize_Sketch

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

class FGSBIR_Dataset(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode
        coordinate_path = os.path.join(hp.root_dir, 'Dataset', hp.dataset_name , hp.dataset_name + '_Coordinate')
        self.root_dir = os.path.join(hp.root_dir, 'Dataset', hp.dataset_name)
        with open(coordinate_path, 'rb') as fp:
            self.Coordinate = pickle.load(fp)

        self.Train_Sketch = [x for x in self.Coordinate if 'train' in x]
        self.Test_Sketch = [x for x in self.Coordinate if 'test' in x]

        self.train_transform = get_ransform('Train')
        self.train_transform_split = get_part_transform('Train')
        self.test_transform = get_ransform('Test')
        self.test_transform_split = get_part_transform('Test')

    def __getitem__(self, item):
        sample  = {}
        if self.mode == 'Train':
            sketch_path = self.Train_Sketch[item]

            positive_sample = '_'.join(self.Train_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')

            possible_list = list(range(len(self.Train_Sketch)))
            possible_list.remove(item)
            negative_item = possible_list[randint(0, len(possible_list) - 1)]
            negative_sample = '_'.join(self.Train_Sketch[negative_item].split('/')[-1].split('_')[:-1])
            negative_path = os.path.join(self.root_dir, 'photo', negative_sample + '.png')

            vector_x = self.Coordinate[sketch_path]
            sketch_img = rasterize_Sketch(vector_x)
            # 完整草图和部分草图
            sketch_img = Image.fromarray(sketch_img).convert('RGB')  
            # 完整图像256*256以及切分后的内容，切分后是128*128
            positive_img = Image.open(positive_path).convert('RGB')
            negative_img = Image.open(negative_path).convert('RGB')
            

    

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                sketch_split = split_img(sketch_img)

                positive_img = F.hflip(positive_img)
                positve_split = split_img(positive_img)

                negative_img = F.hflip(negative_img)
                negative_split = split_img(negative_img)
            else:
                sketch_split = split_img(sketch_img)
                positve_split = split_img(positive_img)
                negative_split = split_img(negative_img)

              
            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)
            
            sketch_part =  [self.train_transform_split(sketch) for sketch in sketch_split]
            positve_part =  [self.train_transform_split(positive) for positive in positve_split]
            negative_part =  [self.train_transform_split(negative) for negative in negative_split]

            positve_part = torch.cat(positve_part).view(-1, 3, 170, 170)
            negative_part = torch.cat(negative_part).view(-1, 3, 170, 170)
            sketch_part = torch.cat(sketch_part).view(-1, 3, 170, 170)
            # print(sketch_part.shape)
            # print(positve_part.shape)

       

          
            sample = {'sketch_img': sketch_img, 'sketch_part': sketch_part,'sketch_path': sketch_path,
                    'positive_img': positive_img,'positve_part':positve_part, 'positive_path': positive_sample,
                    'negative_img': negative_img, 'negative_part':negative_part,'negative_path': negative_sample
                      }

        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            vector_x = self.Coordinate[sketch_path]
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')  
            sketch_split = split_img(sketch_img)
            sketch_img = self.test_transform(sketch_img)
            sketch_part =  [self.test_transform_split(sketch) for sketch in sketch_split]

            
            positive_sample = '_'.join(self.Test_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            
            positive_img = Image.open(positive_path).convert('RGB')
            positve_split = split_img(positive_img)
            positive_img = self.test_transform(positive_img)
            positive_part = [self.test_transform_split(positive) for positive in positve_split]
            
            positive_part = torch.cat(positive_part).view(-1, 3, 170, 170)
            sketch_part = torch.cat(sketch_part).view(-1, 3, 170, 170)

            sample = {'sketch_img': sketch_img, 'sketch_part':sketch_part,'sketch_path': sketch_path, 'Coordinate':vector_x,
                      'positive_img': positive_img, 'positive_part':positive_part,'positive_path': positive_sample}

        return sample
#返回的是草图，草图的索引路径。以及正样本的草图以及正样本草图的路径和索引。其中vector_x不知道指的是什么。好像是coordinate中的值。应该是草图的变量


    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)

def get_dataloader(hp):

    dataset_Train  = FGSBIR_Dataset(hp, mode = 'Train')
    #返回的是每一分个分支train集
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,  num_workers=int(hp.nThreads))

    dataset_Test  = FGSBIR_Dataset(hp, mode = 'Test')
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False,  num_workers=int(hp.nThreads))

    return dataloader_Train, dataloader_Test

def get_ransform(type):
    transform_list = []
    if type is 'Train':
        transform_list.extend([transforms.Resize(299)])
    elif type is 'Test':
        transform_list.extend([transforms.Resize(299)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)

def get_part_transform(type):
    transform_list = []
    if type is 'Train':
        transform_list.extend([transforms.Resize(180), transforms.CenterCrop(170)])
    else:
        transform_list.extend([transforms.Resize(170)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)
