from model import Environment
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch
import torch.nn.utils as utils
import numpy as np
import argparse
import os
import torch.nn.functional as F
import math
from torch.utils.tensorboard import SummaryWriter
from Netwokrs_LGRL import part_att_fc,sketch_fc
device = torch.device("cuda:0")
GAMMA = 0.9
associate_weight = 1
tb_logdir = "./login/"


def main_train(opt):

   
    mean_IOU_buffer = 0
  
    real_p = [0, 0, 0, 0]
    SBIR_Environment = Environment()

    loss_buffer = []
    loss_buffer_local =[]

    optimizer_list = []
    for model in SBIR_Environment.model_list:
        optimizer_list.append(optim.Adam(model.parameters(), lr=opt.lr)) 
    # 定义优化器
    criterion = torch.nn.TripletMarginLoss(margin=0.3, p=2)
    

    # 定义损失
    tb_writer = SummaryWriter(log_dir=tb_logdir)
    #tensorboard


    Top1_Song = [0]
    Top5_Song = [0]
    Top10_Song = [0]
    meanIOU_Song = []
    meanMA_Song = []
    w_ma=[]
    w_mb=[]

    step_stddev = 1
    for model in SBIR_Environment.model_list:
        model.train()
    #模型设置为训练模式

    for epoch in range(opt.niter):
        print('LR value : {}'.format(optimizer_list[0].param_groups[0]['lr']))

        for i, sanpled_batch in enumerate(SBIR_Environment.Sketch_Array_Train):

            
            loss_step = 0
            loss_triplet = 0
           
            loss_triple_part = 0
            loss_triplet_all = 0
            loss_triple_part_all = 0
            j=0

            feature_com_20 = SBIR_Environment.model_list[0](sanpled_batch.to(device))
            # print(feature_com_20.shape)  20， 2048

            positive, positive_part, negative, negative_part = SBIR_Environment.get_sample(SBIR_Environment.Sketch_Name_Train[i])
           

            for i_sketch in range(len(SBIR_Environment.Sketch_Array_Train_part[i])):
                feature_part = SBIR_Environment.model_list[1](SBIR_Environment.Sketch_Array_Train_part[i][i_sketch].to(device))

                loss_triplet += criterion(feature_com_20[i_sketch], positive, negative)
                loss_triple_part += criterion(feature_part, positive_part, negative_part).to(device) 

            loss_step += loss_triplet + loss_triple_part
            loss_triplet_all += loss_triplet
            loss_triple_part_all += loss_triple_part

            loss_buffer.append(loss_step)
       
            # 累加损失
            step_stddev += 1
            if (i + 1) % opt.save_iter == 0:
                print('Epoch: {}, Iteration: {}, loss_global:{},loss_local:{},step: {}'.format(epoch, i, loss_triplet_all,loss_triple_part_all,step_stddev))
                # print('loss_global:{},loss_local:{}'.format(loss_triplet_all,loss_triple_part_all))
                tb_writer.add_scalar("loss", loss_step.item(), step_stddev)
                tb_writer.add_scalar("loss_triplet", loss_triplet.item(), step_stddev)
                tb_writer.add_scalar("loss_triple_part", loss_triple_part.item(), step_stddev)
                

            if (i + 1) % 20 == 0: #[Update after every 20 images]
                optimizer_list[0].zero_grad()
                optimizer_list[1].zero_grad()

                policy_loss = torch.stack(loss_buffer).mean()
                policy_loss.backward()

                for model in SBIR_Environment.model_list:
                    utils.clip_grad_norm_(model.parameters(), 40)
                # 梯度裁剪
                optimizer_list[0].step()
                optimizer_list[1].step()

                loss_buffer = []
      
            if (i + 1) % opt.save_iter == 0 and epoch>=20:
            # if (i + 1) % opt.save_iter == 0:

                with torch.no_grad():
                    
                    top1, top5, top10, mean_IOU, mean_MA,meanOurA,meanOurB = SBIR_Environment.evaluate_NN()
                    for model in SBIR_Environment.model_list:
                        model.train()
                    print('Epoch: {}, Iteration: {}:'.format(epoch, i))
                    print("TEST A@1: {}".format(top1))
                    print("TEST A@5: {}".format(top5))
                    print("TEST A@10: {}".format(top10))
                    print("TEST M@B: {}".format(mean_IOU))
                    print("TEST M@A: {}".format(mean_MA))
                    print("TEST w@mB: {}".format(meanOurB))
                    print("TEST w@mA: {}".format(meanOurA))
                    Top1_Song.append(top1)
                    Top5_Song.append(top5)
                    Top10_Song.append(top10)
                    meanIOU_Song.append(mean_IOU)
                    meanMA_Song.append(mean_MA)
                    w_ma.append(meanOurA)
                    w_mb.append(meanOurB)

                    tb_writer.add_scalar("w@MB", meanOurB, step_stddev)
                    tb_writer.add_scalar("w@MA", meanOurA, step_stddev)
                    tb_writer.add_scalar("MB", mean_IOU, step_stddev)
                    tb_writer.add_scalar("MA", mean_MA, step_stddev)
                    tb_writer.add_scalar("A@5", top5, step_stddev)
                    tb_writer.add_scalar("A@10", top10, step_stddev)

                if mean_IOU > mean_IOU_buffer:

                    save_tag = 1
                    for model in SBIR_Environment.model_list:
                        torch.save(model.state_dict(), 'model' + str(save_tag) + '.pth')
                        save_tag += 1
                    # torch.save(SBIR_Environment.model.state_dict(), 'model_BestNN.pth')
                    mean_IOU_buffer = mean_IOU
                    w_ma_our = meanOurA
                    w_mb_our = meanOurB

                    # # 这种做法会导致其他指标偏高
                    real_p = [top1, top5, top10, mean_MA]
                    # 更改后符合保存模型时的真实指标
                    print('Model Updated')

              
                print('REAL performance: Top1: {}, Top5: {}, Top10: {}, MB: {}, MA: {},'.format(real_p[0], real_p[1],
                                                                                                real_p[2],
                                                                                                mean_IOU_buffer,
                                                                                                real_p[3]))
                print('wma:{},w@mb:{}'.format(w_ma_our,w_mb_our))


    print("TOP1_MAX: {}".format(max(Top1_Song)))
    print("TOP5_MAX: {}".format(max(Top5_Song)))
    print("TOP10_MAX: {}".format(max(Top10_Song)))
    print("meaIOU_MAX: {}".format(max((meanIOU_Song))))
    print("meaMA_MAX: {}".format(max((meanMA_Song))))
    print("w@ma_MAX: {}".format(max((w_ma))))
    print("w@mb_MAX: {}".format(max((w_mb))))
    print(Top1_Song)
    print(Top5_Song)
    print(Top10_Song)
    print(meanIOU_Song)
    print(meanMA_Song)
    print(w_ma)
    print(w_mb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.mode = 'Train'
    opt.Train = True
    opt.shuffle = True
    opt.batchsize = 1 #has to be one
    opt.nThreads = 4
    opt.lr = 0.01
    opt.niter = 500
    opt.save_iter = 64
    opt.load_earlier = False
    main_train(opt)




