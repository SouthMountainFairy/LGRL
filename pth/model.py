from torch.autograd import Variable
import torch.nn as nn
# from Networks import InceptionV3_Network
from Networks import InceptionV3_Network,Attention_local,Attention_global,Linear_global,Linear_local
from torch import optim
import torch
import time
import torch.nn.functional as F
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class FGSBIR_Model(nn.Module):
    def __init__(self, hp):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network = InceptionV3_Network(hp)
        # self.sample_train_params = self.sample_embedding_network.parameters()
        self.sample_train_params = self.sample_embedding_network.parameters()
        
        self.hp = hp
        self.loss = nn.TripletMarginLoss(margin=0.2)

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        # 注意力块
        self.Attention_local = Attention_local()
        self.Attention_local.apply(init_weights)
        self.attn_train_local_params = self.Attention_local.parameters()

        self.Attention_global = Attention_global()
        self.Attention_global.apply(init_weights)
        self.attn_train_global_params = self.Attention_global.parameters()



        # 线性层
        self.Linear_local = Linear_local(hp.feature_num)
        self.Linear_local.apply(init_weights)
        self.linear_train_local_params = self.Linear_local.parameters()

        self.Linear_global = Linear_global(hp.feature_num)
        self.Linear_global.apply(init_weights)
        self.linear_train_global_params = self.Linear_global.parameters()


        self.optimizer = optim.Adam(self.sample_train_params, hp.learning_rate)

        self.optimizer = optim.Adam([
            {'params': filter(lambda param: param.requires_grad, self.sample_train_params), 'lr': hp.learning_rate},
            {'params': self.attn_train_local_params, 'lr': hp.learning_rate},
            {'params': self.attn_train_global_params, 'lr': hp.learning_rate},
            {'params': self.linear_train_local_params, 'lr': hp.learning_rate},
            {'params': self.linear_train_global_params, 'lr': hp.learning_rate}])


    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device), 1)
        positive_feature = self.Linear_global(self.Attention_global(positive_feature))
        # print('train,positive_feature.shape',positive_feature.shape)

        negative_feature = self.sample_embedding_network(batch['negative_img'].to(device), 1)
        negative_feature=self.Linear_global(self.Attention_global(negative_feature))
        # print('train,negative_feature.shape',negative_feature.shape)

        sample_feature = self.sample_embedding_network(batch['sketch_img'].to(device), 1)
        sample_feature =self.Linear_global(self.Attention_global(sample_feature))
        # print('train,sample_feature.shape',sample_feature.shape)



        sample_feature_P = self.sample_embedding_network(batch['sketch_part'].view(-1, 3, 170, 170).to(device), 0)
        sample_feature_P = self.Linear_local(self.Attention_local(sample_feature_P))
        # print('train,sample_feature_P.shape',sample_feature_P.shape)

        positive_feature_P = self.sample_embedding_network(batch['positve_part'].view(-1, 3, 170, 170).to(device), 0)
        positive_feature_P = self.Linear_local(self.Attention_local(positive_feature_P))
        # print('train,positive_feature_P.shape',positive_feature_P.shape)

        negative_feature_P = self.sample_embedding_network(batch['negative_part'].view(-1, 3, 170, 170).to(device), 0)
        negative_feature_P = self.Linear_local(self.Attention_local(negative_feature_P))
        # print('train,negative_feature_P.shape',negative_feature_P.shape)


        loss_G = self.loss(sample_feature, positive_feature, negative_feature)
        loss_P = self.loss(sample_feature_P, positive_feature_P, negative_feature_P)
        loss = loss_G + loss_P

        loss.backward()
        self.optimizer.step()
        return loss_G.item(), loss_P.item()

    def evaluate(self, datloader_Test):
        Image_Feature_ALL = []
        Sketch_Feature_ALL_P0 = []
        Sketch_Feature_ALL_P1 = []
        Sketch_Feature_ALL_P2 = []
        Sketch_Feature_ALL_P3 = []
        Sketch_Feature_ALL_P4 = []
        Sketch_Feature_ALL_P5 = []
        Sketch_Feature_ALL_P6 = []
        Sketch_Feature_ALL_P7 = []
        Sketch_Feature_ALL_P8 = []


        Image_Name = []
        Sketch_Feature_ALL = []

        Sketch_Name = []
        start_time = time.time()
        self.eval()
        Image_Feature_ALL_P0 = torch.FloatTensor().to(device)
        Image_Feature_ALL_P1 = torch.FloatTensor().to(device)
        Image_Feature_ALL_P2 = torch.FloatTensor().to(device)
        Image_Feature_ALL_P3 = torch.FloatTensor().to(device)
        Image_Feature_ALL_P4 = torch.FloatTensor().to(device)
        Image_Feature_ALL_P5 = torch.FloatTensor().to(device)
        Image_Feature_ALL_P6 = torch.FloatTensor().to(device)
        Image_Feature_ALL_P7 = torch.FloatTensor().to(device)
        Image_Feature_ALL_P8 = torch.FloatTensor().to(device)



        # 遍历一遍
        for i_batch, sanpled_batch in enumerate(datloader_Test):
            sketch_feature, positive_feature, sample_feature_P, positive_feature_P = self.test_forward(sanpled_batch)
            Sketch_Feature_ALL.extend(sketch_feature)

            Sketch_Feature_ALL_P0.extend(sample_feature_P[0].unsqueeze(0))
            Sketch_Feature_ALL_P1.extend(sample_feature_P[1].unsqueeze(0))
            Sketch_Feature_ALL_P2.extend(sample_feature_P[2].unsqueeze(0))
            Sketch_Feature_ALL_P3.extend(sample_feature_P[3].unsqueeze(0))
            Sketch_Feature_ALL_P4.extend(sample_feature_P[4].unsqueeze(0))
            Sketch_Feature_ALL_P5.extend(sample_feature_P[5].unsqueeze(0))
            Sketch_Feature_ALL_P6.extend(sample_feature_P[6].unsqueeze(0))
            Sketch_Feature_ALL_P7.extend(sample_feature_P[7].unsqueeze(0))
            Sketch_Feature_ALL_P8.extend(sample_feature_P[8].unsqueeze(0))



            Sketch_Name.extend(sanpled_batch['sketch_path'])

            for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
                if positive_name not in Image_Name:
                    Image_Name.append(sanpled_batch['positive_path'][i_num])
                    Image_Feature_ALL.append(positive_feature[i_num])
                    Image_Feature_ALL_P0 = torch.cat((Image_Feature_ALL_P0, positive_feature_P[0].unsqueeze(0).detach()))
                    Image_Feature_ALL_P1 = torch.cat((Image_Feature_ALL_P1, positive_feature_P[1].unsqueeze(0).detach()))
                    Image_Feature_ALL_P2 = torch.cat((Image_Feature_ALL_P2, positive_feature_P[2].unsqueeze(0).detach()))
                    Image_Feature_ALL_P3 = torch.cat((Image_Feature_ALL_P3, positive_feature_P[3].unsqueeze(0).detach()))
                    Image_Feature_ALL_P4 = torch.cat((Image_Feature_ALL_P4, positive_feature_P[4].unsqueeze(0).detach()))
                    Image_Feature_ALL_P5 = torch.cat((Image_Feature_ALL_P5, positive_feature_P[5].unsqueeze(0).detach()))
                    Image_Feature_ALL_P6 = torch.cat((Image_Feature_ALL_P6, positive_feature_P[6].unsqueeze(0).detach()))
                    Image_Feature_ALL_P7 = torch.cat((Image_Feature_ALL_P7, positive_feature_P[7].unsqueeze(0).detach()))
                    Image_Feature_ALL_P8 = torch.cat((Image_Feature_ALL_P8, positive_feature_P[8].unsqueeze(0).detach()))


        rank = torch.zeros(len(Sketch_Name))
        Image_Feature_ALL = torch.stack(Image_Feature_ALL)

        for num, sketch_feature in enumerate(Sketch_Feature_ALL):
            s_name = Sketch_Name[num]

            sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
            position_query = Image_Name.index(sketch_query_name)
            #  草图块
            sketch_part_0 = Sketch_Feature_ALL_P0[num]
            sketch_part_1 = Sketch_Feature_ALL_P1[num]
            sketch_part_2 = Sketch_Feature_ALL_P2[num]
            sketch_part_3 = Sketch_Feature_ALL_P3[num]
            sketch_part_4 = Sketch_Feature_ALL_P4[num]
            sketch_part_5 = Sketch_Feature_ALL_P5[num]
            sketch_part_6 = Sketch_Feature_ALL_P6[num]
            sketch_part_7 = Sketch_Feature_ALL_P7[num]
            sketch_part_8 = Sketch_Feature_ALL_P8[num]


            # 正样本
            positive = Image_Feature_ALL[position_query]
            positive_P0 = Image_Feature_ALL_P0[position_query]
            positive_P1 = Image_Feature_ALL_P1[position_query]
            positive_P2 = Image_Feature_ALL_P2[position_query]
            positive_P3 = Image_Feature_ALL_P3[position_query]
            positive_P4 = Image_Feature_ALL_P4[position_query]
            positive_P5 = Image_Feature_ALL_P5[position_query]
            positive_P6 = Image_Feature_ALL_P6[position_query]
            positive_P7 = Image_Feature_ALL_P7[position_query]
            positive_P8 = Image_Feature_ALL_P8[position_query]


        
            distance_P0 = F.pairwise_distance(sketch_part_0.to(device), Image_Feature_ALL_P0.to(device))
            distance_P1 = F.pairwise_distance(sketch_part_1.to(device), Image_Feature_ALL_P1.to(device))
            distance_P2 = F.pairwise_distance(sketch_part_2.to(device), Image_Feature_ALL_P2.to(device))
            distance_P3 = F.pairwise_distance(sketch_part_3.to(device), Image_Feature_ALL_P3.to(device))
            distance_P4 = F.pairwise_distance(sketch_part_4.to(device), Image_Feature_ALL_P4.to(device))
            distance_P5 = F.pairwise_distance(sketch_part_5.to(device), Image_Feature_ALL_P5.to(device))
            distance_P6 = F.pairwise_distance(sketch_part_6.to(device), Image_Feature_ALL_P6.to(device))
            distance_P7 = F.pairwise_distance(sketch_part_7.to(device), Image_Feature_ALL_P7.to(device))
            distance_P8 = F.pairwise_distance(sketch_part_8.to(device), Image_Feature_ALL_P8.to(device))




            distance_P = distance_P0 + distance_P1+distance_P2 + distance_P3 + distance_P4 + distance_P5+distance_P6 + distance_P7 + distance_P8

            target_distance_P0 = F.pairwise_distance(sketch_part_0.unsqueeze(0).to(device), positive_P0.unsqueeze(0).to(device))
            target_distance_P1 = F.pairwise_distance(sketch_part_1.unsqueeze(0).to(device), positive_P1.unsqueeze(0).to(device))
            target_distance_P2 = F.pairwise_distance(sketch_part_2.unsqueeze(0).to(device), positive_P2.unsqueeze(0).to(device))
            target_distance_P3 = F.pairwise_distance(sketch_part_3.unsqueeze(0).to(device), positive_P3.unsqueeze(0).to(device))
            target_distance_P4 = F.pairwise_distance(sketch_part_4.unsqueeze(0).to(device), positive_P4.unsqueeze(0).to(device))
            target_distance_P5 = F.pairwise_distance(sketch_part_5.unsqueeze(0).to(device), positive_P5.unsqueeze(0).to(device))
            target_distance_P6 = F.pairwise_distance(sketch_part_6.unsqueeze(0).to(device), positive_P6.unsqueeze(0).to(device))
            target_distance_P7 = F.pairwise_distance(sketch_part_7.unsqueeze(0).to(device), positive_P7.unsqueeze(0).to(device))
            target_distance_P8 = F.pairwise_distance(sketch_part_8.unsqueeze(0).to(device), positive_P8.unsqueeze(0).to(device))


            target_distance_P = target_distance_P0 + target_distance_P1 + target_distance_P2 + target_distance_P3 +target_distance_P4 + target_distance_P5 + target_distance_P6 + target_distance_P7 + target_distance_P8

            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL).to(device)
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0), positive.unsqueeze(0)).to(device)

            distance = distance.to(device) + distance_P.to(device)
            target_distance = target_distance.to(device) + target_distance_P.to(device)

            rank[num] = distance.le(target_distance).sum()

        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top5 = rank.le(5).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]

        print('Time to EValuate:{}'.format(time.time() - start_time))
        return top1, top5, top10
    



    def test_forward(self, batch):

        sketch_feature = self.sample_embedding_network(batch['sketch_img'].to(device), 1)
        sketch_feature =self.Linear_global(self.Attention_global(sketch_feature))
        # print('test,sketch.shape',sketch_feature.shape)

        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device), 1)
        positive_feature =self.Linear_global(self.Attention_global(positive_feature))
        # print('test,pos.shape',positive_feature.shape)


        sample_feature_P = self.sample_embedding_network(batch['sketch_part'].view(-1, 3, 170, 170).to(device), 0)
        sample_feature_P = self.Linear_local(self.Attention_local(sample_feature_P))
        # print('test,sample_feature_P.shape',sample_feature_P.shape)

        positive_feature_P = self.sample_embedding_network(batch['positive_part'].view(-1, 3, 170, 170).to(device), 0)
        positive_feature_P = self.Linear_local(self.Attention_local(positive_feature_P))
        # print('test,positive_feature_P.shape',positive_feature_P.shape)

        return sketch_feature, positive_feature, sample_feature_P, positive_feature_P
