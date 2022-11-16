import argparse
import os.path
from torch.utils.data import DataLoader
from utils.get_dataset import get_dataset, get_trans_train, get_trans_test,Data_dict
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser('Training for TransNet')
parser.add_argument('--model_name', '-net', type=str, default='viswin_id2')
parser.add_argument('--dataset', '-dt', type=str, default='kfw1_identification')
# parser.add_argument('--tp', type=str, default=None, help="[fd, fs, md, ms]")
parser.add_argument('--gpu', default='1', type=str, help='gpu number')
parser.add_argument('--save_pth', type=str, default='./results')
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--expand_dim', type=bool, default=True)

parser.add_argument('--model_path_name', type=str, default='un_ytb')
parser.add_argument('--model_path', type=str, default='./results/un_ytb/video_un_small_fc/video_un_small_fc_epoch18.pth')
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

    save_pth = args.save_pth + '/{}/{}/'.format(args.dataset, args.model_name)
    if not os.path.isdir(save_pth):
        os.makedirs(save_pth)
    ###### run 5-corss-validation
    cross_acc = []
    
    for cross in range(5, 0, -1):
        print("=" * 20 + "cross :{}".format(6 - cross) + "=" * 20)
        current_cross = 6-cross
        ###### write log
        
        log_pth = args.save_pth + '/logs/{}/{}/{}/'.format(args.dataset, args.model_name,current_cross)
        if not os.path.isdir(log_pth):
            os.makedirs(log_pth)
        print('==='*20)
        print(log_pth)
        
        writer = SummaryWriter(log_dir=log_pth)
        # writer.add_text(tag='configures',
        #                 text_string= str(options))


        train_cross = [item for item in cross_valid if item != cross]
        test_cross = [cross]
        train_loader = DataLoader(data_set(mat_pth, im_root, train_cross, get_trans_train(args.dataset,expand_dim=args.expand_dim)),
                                    batch_size=args.batch, shuffle=True, num_workers=0)
        test_loader = DataLoader(data_set(mat_pth, im_root, test_cross, get_trans_test(args.dataset,expand_dim=args.expand_dim), test=True),
                                    batch_size=args.batch, num_workers=0)

        best_acc = 0
        loss_record = []
        acc_record = []
        acc_train_record = []

        net = Model_dict[args.model_name](args)
        for epoch in range(args.epochs):
            loss = net.train(epoch, train_loader)
            loss_record.append(loss)
            writer.add_scalar(tag='loss curve',
                                scalar_value=loss,
                                global_step=epoch
                                )
            acc = net.test(test_loader)
            acc_record.append(acc)
            writer.add_scalar(tag='test accuracy',
                                scalar_value=acc,
                                global_step=epoch
                                )
            if epoch % 1 == 0:
                acc_train = net.test(train_loader)
                acc_train_record.append(acc_train)
                writer.add_scalar(tag='train accuracy',
                                    scalar_value=acc_train,
                                    global_step=epoch
                                    )

                print('Cross_valid:{}, test acc: {:.5f}, train acc {:.5f}'.format( 6 - cross, acc,
                                                                                            acc_train))
            else:
                print('Cross_valid:{}, test acc: {:.5f}'.format( 6 - cross, acc))
            if epoch >= int(args.epochs * 1 / 5):
                if (acc > best_acc) or (epoch == args.epochs - 1):
                    # model_sv_pth = save_pth + '{}_cross{}_{}_{}'.format(args.model_name, cross, epoch,
                    #                                                     datetime.now().replace(second=0,
                    #                                                                             microsecond=0))
                    model_sv_pth = save_pth + '{}_cross{}_best'.format(args.model_name, 6-cross)
                    
                    best_acc = acc

                    net.save(model_sv_pth)
            
        writer.add_text(tag='test acc list',
                        text_string=str(list(zip(range(args.epochs),acc_record))))
        cross_acc.append(best_acc)

        writer.close()
    avg_cross = np.mean(cross_acc)
    writer_cross = SummaryWriter(log_dir=log_pth)
    writer_cross.add_text(tag='configures',
                    text_string= str(options))
    writer_cross.add_text('best acc - avg',str(avg_cross))
    writer_cross.close()
    # acc_kin[tp] = avg_cross
    print('cross_acc')
    print(cross_acc)
    print('-' * 20 + 'The avg acc is {}'.format(avg_cross) + '-' * 20)
    # print(options)
    # print(acc_kin)
    # print(np.mean([acc_kin[item] for item in acc_kin]))




