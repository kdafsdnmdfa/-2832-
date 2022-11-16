import os
from utils.loader import *
from utils.loader import kfw_dataset, fiw_dataset, nemo_dataset, nemo_image_dataset,Youtube_Faces_DB,nemo_image_dataset2
from utils.loader_identification import Kfw_identification_dataset

Data_dict = {'kfw1': kfw_dataset, 'kfw2': kfw_dataset, 'fiw': fiw_dataset, 'nemo': nemo_image_dataset,'nemo_img':nemo_image_dataset2,'ytf':Youtube_Faces_DB,
             'kfw1_identification':Kfw_identification_dataset,'kfw2_identification':Kfw_identification_dataset}

mats_dict = {'kfw1':['fd', 'fs', 'md', 'ms'],
             'kfw2':['fd', 'fs', 'md', 'ms'],
             'fiw':['fd','fs','md','ms','bb','bs','ss'],
             'nemo':['F-D','F-S','M-D','M-S','B-B','B-S','S-S'],
             'nemo_img':['F-D','F-S','M-D','M-S','B-B','B-S','S-S'],
             'kfw1_identification':[],
             'kfw2_identification':[]
             }

imgs_dict = {'kfw1':{'fd':'father-dau', 'fs':'father-son', 'md':'mother-dau', 'ms':'mother-son'},
             'kfw2':{'fd':'father-dau', 'fs':'father-son', 'md':'mother-dau', 'ms':'mother-son'},
             'fiw':{'fd':'','fs':'','md':'','ms':'','bb':'','bs':'','ss':''},
             'nemo':{'F-D':'F-D','F-S':'F-S','M-D':'M-D','M-S':'M-S','B-B':'B-B','B-S':'B-S','S-S':'S-S'},
             'nemo_img':{'F-D':'F-D','F-S':'F-S','M-D':'M-D','M-S':'M-S','B-B':'B-B','B-S':'B-S','S-S':'S-S'},
             'kfw1_identification':[],
             'kfw2_identification':[]
             }

label_type = {'kfw1':'_pairs.mat','kfw2':'_pairs.mat','fiw':'.pkl','nemo':'.pkl','nemo_img':'.pkl',
              'kfw1_identification':'_pairs.mat','kfw2_identification':'_pairs.mat'}

mat_pth_dict = {'kfw1': '/var/scratch/CODE/transformer/transformer-kin/data/KinFaceW-I/meta_data',
                'kfw2':'/var/scratch/CODE/transformer/transformer-kin/data/KinFaceW-II/meta_data',
                'fiw':'/var/scratch/CODE/transformer/transformer-kin/data/FIW/origin/fitted_original_5split',
                'nemo':'/local/Nemo/label/train_list',
                'nemo_img':'/local/Nemo/label/train_list',
                'kfw1_identification':
                ['/home/CODE/transformer/transformer-kin/data/KinFaceW-I/meta_data/fd_pairs.mat',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-I/meta_data/fs_pairs.mat',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-I/meta_data/md_pairs.mat',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-I/meta_data/ms_pairs.mat'],
                'kfw2_identification':
                ['/home/CODE/transformer/transformer-kin/data/KinFaceW-II/meta_data/fd_pairs.mat',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-II/meta_data/fs_pairs.mat',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-II/meta_data/md_pairs.mat',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-II/meta_data/ms_pairs.mat']
                }

im_root_dict = {'kfw1':'/var/scratch/CODE/transformer/transformer-kin/data/KinFaceW-I/images',
                'kfw2':'/var/scratch/CODE/transformer/transformer-kin/data/KinFaceW-II/images',
                'fiw':'/var/scratch/CODE/transformer/transformer-kin/data/FIW/origin/train-faces',
                'nemo':'/local/Nemo/kin_simple/frames/',
                'nemo_img':'/local/Nemo/kin_simple/frames/',
                'kfw1_identification':
                ['/home/CODE/transformer/transformer-kin/data/KinFaceW-I/images/father-dau',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-I/images/father-son',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-I/images/mother-dau',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-I/images/mother-son'],
                'kfw2_identification':
                ['/home/CODE/transformer/transformer-kin/data/KinFaceW-II/images/father-dau',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-II/images/father-son',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-II/images/mother-dau',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-II/images/mother-son'],
                }


class get_dataset:
    def __init__(self,args):
        if 'identification' not in args.dataset:
            self.mats = mats_dict[args.dataset]
            self.img_dict = imgs_dict[args.dataset]
            self.args = args
            self.lb_tp = label_type[args.dataset]
            self.mat_pth_ = mat_pth_dict[args.dataset]
            self.im_root_ = im_root_dict[args.dataset]
        else:
            self.mats = mat_pth_dict[args.dataset]
            self.im_root = im_root_dict[args.dataset]
            
    def mat_pth(self,tp):
        mat_pth = '{}/{}{}'.format(self.mat_pth_,tp,self.lb_tp)
        return mat_pth


    def im_root(self,tp):
        im_root = os.path.join(self.im_root_, self.img_dict[tp])
        return im_root


def get_trans_train(name, expand_dim = False):
    if expand_dim:
        return Expand_transform(mdr_transform_train, n_views=1)
    if 'nemo_img' in name:
        return image_transform_train
    if ('video' in name) or ('nemo' in name):
        return video_transform_train
    elif 'unsupervise' in name:
        return ContrastiveTransformations(contrast_transforms, n_views=2)
    else:
        return mdr_transform_train

def get_trans_test(name, expand_dim = False):
    if expand_dim:
        return Expand_transform(mdr_transform_test, n_views=1)
    if 'nemo_img' in name:
        return image_transform_test
    if ('video' in name) or ('nemo' in name):
        return video_transform_test
    elif 'unsupervise' in name:
        return ContrastiveTransformations(contrast_transforms, n_views=2)
    else:
        return mdr_transform_test

