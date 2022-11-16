import argparse
import os.path
from torch.utils.data import DataLoader
from utils.get_dataset import get_dataset, get_trans_train, get_trans_test,Data_dict
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser('Training for TransNet')
parser.add_argument('--model_name', '-net', type=str, default='viswin_id2')
parser.add_argument('--dataset', '-dt', type=str, default='kfw2_identification')
# parser.add_argument('--tp', type=str, default=None, help="[fd, fs, md, ms]")
parser.add_argument('--gpu', default='2', type=str, help='gpu number')
parser.add_argument('--save_pth', type=str, default='./results')
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--expand_dim', type=bool, default=True)

parser.add_argument('--model_path_name', type=str, default='no_pretrain')
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--embed_dim', type=str, default=48)




if __name__ == '__main__':

    args = parser.parse_args()
    options = vars(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
    print(options)
    print('visable gpu:-----------',args.gpu)
    ####
    from utils.get_net import Model_dict
    dataset_config = get_dataset(args)
    data_set = Data_dict[args.dataset]
    mat_pth = dataset_config.mats 
    im_root = dataset_config.im_root
    print(mat_pth)
    cross_valid = [1, 2, 3, 4, 5]


    acc_kin = {}
        
    save_pth = args.save_pth + '/{}/{}/'.format(args.dataset, args.model_name)
    
    ###### run 5-corss-validation
    cross_acc = []
    for cross in range(5, 0, -1):
        print("=" * 20 + "cross :{}".format(6 - cross) + "=" * 20)
        current_cross = 6-cross
        ###### write log
        

        train_cross = [item for item in cross_valid if item != cross]
        test_cross = [cross]

        test_loader = DataLoader(data_set(mat_pth, im_root, test_cross, get_trans_test(args.dataset,expand_dim=args.expand_dim), test=True),
                                    batch_size=args.batch, num_workers=0)

        best_acc = 0
        loss_record = []
        acc_record = []
        acc_train_record = []

        net = Model_dict[args.model_name](args)
        
        
        model_sv_pth = save_pth + '{}_cross{}_best'.format(args.model_name, 6-cross)+'.pth'
        
        net.load(model_sv_pth)
        
        net.cs = 6-cross
        acc = net.test(test_loader)
        acc = net.tsne(test_loader,vis_model='3d')
        
        print('test acc: ', acc)
        cross_acc.append(acc)
    

    avg_cross = np.mean(cross_acc)
    print('cross acc: ', cross_acc)
    print('avg acc: ', avg_cross)




