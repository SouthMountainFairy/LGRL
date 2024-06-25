import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
import torch.nn as nn
from dataset_chairv2 import *
import time
from matplotlib import pyplot as plt
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import math
from RL_Networks import InceptionV3_Network,Attention_local,Attention_global,Linear_global,Linear_local
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pi = torch.FloatTensor([math.pi]).to(device)

rgb_dir = '/home/ubuntu/workplace/lyg/Dataset/ChairV2/'
coord_dir = '/home/ubuntu/workplace/lyg/Dataset/ChairV2/ChairV2_Coordinate'

model_dir = '/home/ubuntu/workplace/lyg/two-ranch/chair-64-D/9-pth/InceptionV3_ChairV2_model_best.pth'
model_att_local = '/home/ubuntu/workplace/lyg/two-ranch/chair-64-D/9-pth/ChairV2_64_att_local.pth'
model_att_Gobal = '/home/ubuntu/workplace/lyg/two-ranch/chair-64-D/9-pth/ChairV2_64_attn_global.pth'
model_linear_local = '/home/ubuntu/workplace/lyg/two-ranch/chair-64-D/9-pth/ChairV2_64_linear_local.pth'
model_linear_Global = '/home/ubuntu/workplace/lyg/two-ranch/chair-64-D/9-pth/ChairV2_64_linear_global.pth'


train_sketch_pickle_dir = "./Train_sketch.pickle"
train_image_pickle_dir = "./Train_image.pickle"

def main(opt):
    model_fixed = InceptionV3_Network()
    model_fixed.to(device)
    model_fixed.load_state_dict(torch.load(model_dir), strict=False)
    model_fixed.fix_backbone()
    model_fixed.eval()

    Att_local = Attention_local()
    Att_local.to(device)
    Att_local.load_state_dict(torch.load(model_att_local), strict=False)
    Att_local.fix_backbone()
    Att_local.eval()

    Att_Global = Attention_global()
    Att_Global.to(device)
    Att_Global.load_state_dict(torch.load(model_att_Gobal), strict=False)
    Att_Global.fix_backbone()
    Att_Global.eval()

    lin_Global = Linear_global(feature_num=64)
    lin_Global.to(device)
    lin_Global.load_state_dict(torch.load(model_linear_Global), strict=False)
    lin_Global.fix_backbone()
    lin_Global.eval()

    lin_local = Linear_local(feature_num=64)
    lin_local.to(device)
    lin_local.load_state_dict(torch.load(model_linear_local), strict=False)
    lin_local.fix_backbone()
    lin_local.eval()


    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.coordinate = coord_dir
    opt.roor_dir = rgb_dir
    opt.mode = 'Train'
    opt.Train = True
    opt.shuffle = False
    opt.nThreads = 0
    opt.batch_size = 1

    dataset_sketchy_train = createDataset(opt, on_Fly=True)
    dataloader_sketchy_train = data.DataLoader(dataset_sketchy_train, batch_size=opt.batch_size, shuffle=opt.shuffle,num_workers=int(opt.nThreads))

    Image_Array_Train = torch.FloatTensor().to(device)
    Image_Array_Train_part = torch.FloatTensor().to(device)

    sketch_feature_ALL = torch.FloatTensor().to(device)
  
    Sketch_Array_Train = []
    Sketch_Array_Train_part = []

    Image_Name_Train = []
    Sketch_Name_Train = []
    sketch_text_pro = []
    for i_batch, sanpled_batch in enumerate(dataloader_sketchy_train):


        sketch_feature = Att_Global(model_fixed(sanpled_batch['sketch_img_complete'].squeeze(0).to(device),1))
        # print('完整草图的大小：',sanpled_batch['sketch_img_complete'].squeeze(0).shape)
        # print('完整草图的大小：',sketch_feature.shape)

        sketch_feature_part_ALL =[]
        # sketch_feature_part_ALL = torch.FloatTensor().to(device)


        for i in range(len(sanpled_batch['sketch_img_part'])):

            feature_part = Att_local(model_fixed(sanpled_batch['sketch_img_part'][i].squeeze(0).to(device), 0))

            # sketch_feature_part_ALL = torch.cat((sketch_feature_part_ALL, feature_part.detach()))
            sketch_feature_part_ALL.append(feature_part)

      
        #     print('局部草图的大小----图像的大小rgb_feature:{}',feature_part.shape)

        # print('20账草图的维度',sketch_feature_part_ALL.shape)
        

        Sketch_Name_Train.extend(sanpled_batch['sketch_path'])

        Sketch_Array_Train.append(sketch_feature.cpu())
        Sketch_Array_Train_part.append(sketch_feature_part_ALL)

        sketch_text_pro.extend(sanpled_batch['sum_all_d'])

        if sanpled_batch['positive_path'][0] not in Image_Name_Train:


            rgb_feature = Att_Global(model_fixed(sanpled_batch['positive_img'].to(device),1))

            # print('Global----图像的大小rgb_feature:{}'.format(rgb_feature.shape))
            rgb_feature = lin_Global(rgb_feature)
            # print('Global----图像的大小rgb_feature:{}'.format(rgb_feature.shape))

            img_part_9 = lin_local(Att_local(model_fixed(sanpled_batch['positive_img_4'].squeeze(0).to(device), 0)))
            
            print('img_part_9:{}'.format(img_part_9.shape))

            Image_Array_Train = torch.cat((Image_Array_Train, rgb_feature.detach()))
            Image_Array_Train_part = torch.cat((Image_Array_Train_part, img_part_9.unsqueeze(0)))
            # print(Image_Array_Train_part.shape)
        
            Image_Name_Train.extend(sanpled_batch['positive_path'])

        print('Train Image Feature Loading:', i_batch)


    with open(train_sketch_pickle_dir, "wb") as f:
        pickle.dump((Sketch_Array_Train,Sketch_Array_Train_part,Sketch_Name_Train,sketch_text_pro), f)

    with open(train_image_pickle_dir, "wb") as f:
        pickle.dump((Image_Array_Train,Image_Array_Train_part,Image_Name_Train), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    main(opt)
    print("________________________完整内容_______________________")
