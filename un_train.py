import argparse
import os.path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from utils.get_dataset import get_dataset, get_trans_train, get_trans_test,Data_dict
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from tqdm import tqdm

parser = argparse.ArgumentParser('Training for TransNet')
parser.add_argument('--model_name', '-net', type=str, default='video_un_small',help='video_un_small')
parser.add_argument('--dataset', '-dt', type=str, default='un_ytb', help='un_ytb, un_ms1m')
parser.add_argument('--tp', type=str, default=None, help="[fd, fs, md, ms]")
parser.add_argument('--gpu', default='2', type=str, help='gpu number')
parser.add_argument('--save_pth', type=str, default='./results')
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)




if __name__ == '__main__':

    args = parser.parse_args()
    options = vars(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
    print(options)
    print('visable gpu:-----------',args.gpu)
    ####
    from utils.get_net import Model_dict
    from utils.loader import Youtube_Faces_DB,ContrastiveTransformations,contrast_transforms,FaceDataset64,ContrastiveTransformations_image,contrast_transforms_image,contrast_transforms_wo_da


    save_pth = args.save_pth + '/{}/{}/'.format(args.dataset, args.model_name)
    if not os.path.isdir(save_pth):
        os.makedirs(save_pth)
    log_pth = args.save_pth + '/logs/{}/{}/'.format(args.dataset, args.model_name)
    writer = SummaryWriter(log_dir=log_pth)
    writer.add_text(tag='configures',
                    text_string= str(options))
    if args.dataset=='un_ytb':
        print('trainig on youtube faces')
        ytf_data = Youtube_Faces_DB(img_root='/local/ytb_crop',transform=ContrastiveTransformations(contrast_transforms_wo_da, n_views=2))
        train_loader = DataLoader(ytf_data,batch_size=args.batch,shuffle = True)
    elif args.dataset=='un_ms1m':
        print('training on ms1m')
        ms1m = FaceDataset64('/local/crop_ms1m', rand_mirror=True,trans =ContrastiveTransformations_image(contrast_transforms_image, n_views=2))
        train_loader = DataLoader(ms1m,batch_size=args.batch,shuffle = True)
    else:
        print('no dataset')
    
        
    net = Model_dict[args.model_name](args)
    for epoch in range(args.epochs):
        loss,t1_avg,t5_avg,m_avg = net.train(epoch, train_loader,writer)
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/t1', t1_avg, epoch)
        writer.add_scalar('train/t5', t5_avg, epoch)
        writer.add_scalar('train/m', m_avg, epoch)

        print(epoch, '='*5,'_acc_top1','%0.4f'%t1_avg )
        print(epoch, '='*5,'_acc_top5','%0.4f'%t5_avg )
        print(epoch, '='*5,'_acc_mean_pos', '%0.4f'%m_avg)

        if epoch%1 == 0:
            model_sv_pth = save_pth + '{}_epoch{}'.format(args.model_name,epoch)
            net.save(model_sv_pth)
            print('='*20)
            print('model saved at {}'.format(model_sv_pth))
        
    writer.close()










