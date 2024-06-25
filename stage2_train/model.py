import torch
import time
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import pickle
from RL_Networks import part_att_fc, sketch_fc
device = torch.device("cuda:0")
pi = torch.FloatTensor([math.pi]).to(device)

train_image_pickle_dir = "./../pickle/Train_image.pickle"
train_sketch_pickle_dir = "./../pickle/Train_sketch.pickle"

test_image_pickle_dir = "./../pickle/Test_image.pickle"
test_sketch_pickle_dir = "./../pickle/Test_sketch.pickle"


class Environment():
    def __init__(self):
        with open(train_sketch_pickle_dir, "rb") as f:
            self.Sketch_Array_Train, self.Sketch_Array_Train_part, self.Sketch_Name_Train, self.sketch_text_pro = pickle.load(f)

        with open(train_image_pickle_dir, "rb") as f:
            self.Image_Array_Train,self.Image_Array_Train_part,self.Image_Name_Train = pickle.load(f)

        with open(test_sketch_pickle_dir, "rb") as f:
            self.Sketch_Array_Test, self.Sketch_Array_Test_part, self.Sketch_Name_Test, self.Test_sketch_text_pro = pickle.load(f)

        with open(test_image_pickle_dir, "rb") as f:
            self.Image_Array_Test,self.Image_Array_Test_part,self.Image_Name_Test = pickle.load(f)

        self.model_list = [sketch_fc(), part_att_fc()]

        for model in self.model_list:
            model.to(device)
        # 定义模型

    def get_sample(self, sketch_name):
        sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])

        position_query = self.Image_Name_Train.index(sketch_query_name)
        positive = self.Image_Array_Train[position_query]

        positive_part = self.Image_Array_Train_part[position_query]
        
        negative_index = position_query
        while(negative_index == position_query):
            negative_index = np.random.randint(0, 300)

        negative = self.Image_Array_Train[negative_index]

        negative_part = self.Image_Array_Train_part[negative_index]
       

        return positive, positive_part, negative, negative_part

    def evaluate_NN(self):
        for model in self.model_list:
            model.eval()
        num_of_Sketch_Step = len(self.Sketch_Array_Test[0])
        avererage_area = []
        avererage_area_percentile = []
        mean_rank_ourB = []
        mean_rank_ourA = []
        avererage_ourB = []
        avererage_ourA = []
        exps = np.linspace(1,num_of_Sketch_Step, num_of_Sketch_Step) / num_of_Sketch_Step
        factor = np.exp(1 - exps) / np.e
        sketch_range = []
        rank_all = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)
        rank_all_percentile = torch.zeros(
            len(self.Sketch_Array_Test), num_of_Sketch_Step)
        sketch_range = torch.Tensor(sketch_range)

        for i_batch, sanpled_batch in enumerate(self.Sketch_Array_Test):
            sketch_name = self.Sketch_Name_Test[i_batch]
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = self.Image_Name_Test.index(sketch_query_name)

            positive = self.Image_Array_Test[position_query]
            positive_part = self.Image_Array_Test_part[position_query]
          
            mean_rank = []
            mean_rank_percentile = []

            # 草图在每一块中的占比内容,20个为一组，
            pro_sketch = self.Test_sketch_text_pro[i_batch]
      
  

            # 将0-19映射到0-1
            num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19]
            Xmin = np.min(num)
            Xmax = np.max(num)
            a = 0
            b = 1
            Atten_num = a + (b-a)/(Xmax-Xmin)*(num-Xmin)

           
            sketch_feature = self.model_list[0](sanpled_batch.to(device))
            # print(sanpled_batch.shape)
            # # 20,64

            for i_sketch in range(sanpled_batch.shape[0]):
        
                target_distance = F.pairwise_distance(sketch_feature[i_sketch].to(device), positive.to(device)).to(device)
                distance = F.pairwise_distance(sketch_feature[i_sketch].unsqueeze(0).to(device), self.Image_Array_Test.to(device)).to(device)
              

                sketch_feature_part = self.model_list[1](self.Sketch_Array_Test_part[i_batch][i_sketch].to(device))
        
                target_distance_0 =  F.pairwise_distance(sketch_feature_part.to(device), positive_part.to(device)).to(device) 

          
                distance_0 = F.pairwise_distance(sketch_feature_part.to(device), self.Image_Array_Test_part.to(device)).to(device)
                
               
              
                pro_part = pro_sketch[i_sketch]
                pro_part = torch.cat(pro_part)


                Attention_num = round(Atten_num[i_sketch], 2)
                

                dif = torch.sum(torch.mul(pro_part.to(device),distance_0.to(device)),dim=1)

                target_distance_all = target_distance + round(math.exp(-Attention_num), 2)*(torch.sum(torch.mul(pro_part.to(device),target_distance_0).to(device)))


                distance_all = distance + round(math.exp(-Attention_num), 2)*(dif)


                rank_all[i_batch, i_sketch] = distance_all.le(
                    target_distance_all).sum()

                rank_all_percentile[i_batch, i_sketch] = (
                    len(distance_all) - rank_all[i_batch, i_sketch]) / (len(distance_all) - 1)

                if rank_all[i_batch, i_sketch].item() == 0:
                    # 并不存在sum=0的情况，无用？
                    mean_rank.append(1.)

                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
                    # 1/(rank)
                    mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item())
                    # rank_percentile

                    mean_rank_ourB.append(1/rank_all[i_batch, i_sketch].item() * factor[i_sketch])
                    mean_rank_ourA.append(rank_all_percentile[i_batch, i_sketch].item()*factor[i_sketch])

            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
            avererage_ourB.append(np.sum(mean_rank_ourB)/len(mean_rank_ourB))
            avererage_ourA.append(np.sum(mean_rank_ourA)/len(mean_rank_ourA))

        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -
                                  1].le(10).sum().numpy() / rank_all.shape[0]
        # A@1 A@5 A%10
        meanIOU = np.mean(avererage_area)
        meanMA = np.mean(avererage_area_percentile)
        meanOurB = np.mean(avererage_ourB)
        meanOurA = np.mean(avererage_ourA)

        return top1_accuracy, top5_accuracy, top10_accuracy, meanIOU, meanMA,meanOurA,meanOurB


if __name__ == "__main__":
    Environment = Environment()
    top1, top5, top10, mean_IOU, mean_MA = Environment.evaluate_NN()
