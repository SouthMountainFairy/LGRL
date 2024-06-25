import argparse
import torch
import time
from model import FGSBIR_Model
from dataset import get_dataloader
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    parser.add_argument('--backbone_name', type=str,
                        default='InceptionV3', help='VGG / InceptionV3/ Resnet50')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--root_dir', type=str, default='/home/ubuntu/workplace/lyg/')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=50)
    parser.add_argument('--print_freq_iter', type=int, default=1)
    parser.add_argument('--feature_num', type=int, default=64)

    hp = parser.parse_args()
    hp.device = device
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)

    model = FGSBIR_Model(hp)
    model.to(device)
    # model.load_state_dict(torch.load('VGG_ShoeV2_model_best.pth', map_location=device))
    step_count, top1, top5, top10 = -1, 0, 0, 0

    for i_epoch in range(hp.max_epoch):
        for batch_data in dataloader_Train:
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss, lossp = model.train_model(batch=batch_data)

            if step_count % hp.print_freq_iter == 0:
                print('Epoch: {}, Iteration: {}, Loss: {:.5f}, Lossp: {:.5f},Top1_Accuracy: {:.5f}, Top5_Accuracy: {:.5f},Top10_Accuracy: {:.5f}, Time: {}'.format
                      (i_epoch, step_count, loss, lossp, top1, top5, top10, time.time()-start))

            if step_count % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    top1_eval, top5_eval, top10_eval = model.evaluate(
                        dataloader_Test)
                    print('results : ', top1_eval, ' / ', top10_eval)

                if top1_eval > top1:
                    torch.save(model.sample_embedding_network.state_dict(), hp.backbone_name + '_' + hp.dataset_name + '_model_best.pth')
                    torch.save(model.Attention_local.state_dict(),hp.dataset_name + '_' + str(hp.feature_num) + '_att_local.pth')
                    torch.save(model.Attention_global.state_dict(),hp.dataset_name + '_' + str(hp.feature_num) + '_attn_global.pth')
                    torch.save(model.Linear_local.state_dict(),hp.dataset_name + '_' + str(hp.feature_num) + '_linear_local.pth')
                    torch.save(model.Linear_global.state_dict(),hp.dataset_name + '_' + str(hp.feature_num) + '_linear_global.pth')
                    top1, top5, top10 = top1_eval, top5_eval, top10_eval
                    print('Model Updated')
