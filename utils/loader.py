import torch
from torch.utils.data import Dataset
import scipy.io
import os
from PIL import Image
import torchvision.transforms as transforms
import pickle
import random
import numpy as np
from torchvision.io import read_video,write_jpeg,read_image
from tqdm import tqdm

# mdr_transform_train = transforms.Compose([
#     transforms.Resize((73,73)),
#     transforms.CenterCrop((64,64)),
#     transforms.ToTensor()
# ])

class ToTenser_withoutscale:
    def __call__(self, x):
        x = np.array(x)
        x = torch.from_numpy(x).permute(2, 0, 1)
        return x.to(torch.float)

mdr_transform_train = transforms.Compose([
    # transforms.Resize((112,112)),
    transforms.Resize((64,64)),
    # transforms.CenterCrop((112,112)),
     # transforms.ColorJitter(brightness=0.3,
     #                        contrast=0.3,
     #                        saturation=0.3,
     #                        hue=0.3
     #                        ),
     # transforms.CenterCrop((64,64)),
     # transforms.RandomAffine(degrees=1,scale=(0.98,1.02),shear =1),
     # transforms.RandomGrayscale(),
     transforms.RandomHorizontalFlip(p=0.4),
     # transforms.RandomAffine(degrees=(0, 0), translate=(0.3, 0.3), scale=(0.95, 0.95)),
     # transforms.RandomPerspective(distortion_scale=0.2,p=0.3),
     # transforms.RandomResizedCrop(size=(64,64), scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=2),

    # transforms.ToTensor()
    ToTenser_withoutscale()
])


mdr_transform_test = transforms.Compose([
    # transforms.Resize((112,112)),
    transforms.Resize((64,64)),
    # transforms.ToTensor()
    ToTenser_withoutscale()
])


video_transform_train = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.Resize((64,64)),
    # transforms.ToTensor()
])

video_transform_test = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.Resize((64,64)),
    # transforms.ToTensor()
])


image_transform_train = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomGrayscale(),
    transforms.RandomResizedCrop(size=64,scale=(0.9, 1.1)),
    # transforms.RandomApply([
    #     transforms.ColorJitter(brightness=0.5,
    #                             contrast=0.5,
    #                             saturation=0.5,
    #                             hue=0.1),
    #     transforms.GaussianBlur(kernel_size=3)
    # ], p=0.8),
    # transforms.ToTensor()
])

image_transform_test = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.Resize((64,64)),
    # transforms.ToTensor()
])

class ContrastiveTransformations_image(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    
class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
    def trans_each(self,x):
        # trans_frms = map(self.base_transforms,x)
        if not isinstance(x,list):
            x = [x]
        trans_frms = [self.base_transforms(it_) for it_ in x]
        im_frames = torch.stack(list(trans_frms),dim=0).type(torch.float)# [T,C, H, W]
        im_frames = torch.permute(im_frames, (1, 0, 2, 3))#[CTHW]--> [CTHW]
        return im_frames
    def __call__(self, x):
        return [self.trans_each(x) for i in range(self.n_views)]

class Expand_transform(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
    
    def trans_each(self,x):
        if not isinstance(x,list):
            x = [x]
        trans_frms = [self.base_transforms(it_) for it_ in x]
        im_frames = torch.stack(list(trans_frms),dim=0).type(torch.float) # [T,C, H, W]
        im_frames = torch.permute(im_frames, (1, 0, 2, 3)) #[CTHW]--> [CTHW]
        return im_frames
    
    def __call__(self, x):
        return self.trans_each(x) 
    
    
class Vi_Expand_transform(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
    
    def trans_each(self,x):
        if not isinstance(x,list):
            x = [x]
        # trans_frms = [self.base_transforms(it_) for it_ in x]
        im_frames = torch.stack(list(x),dim=0).type(torch.float) # [T,C, H, W]
        im_frames = torch.permute(im_frames, (1, 0, 2, 3)) #[CTHW]--> [CTHW]
        return im_frames
    
    def __call__(self, x):
        return self.trans_each(x) 

class ToTenser_withoutscale:
    def __call__(self, x):
        x = np.array(x)
        x = torch.from_numpy(x).permute(2, 0, 1)
        return x.to(torch.float)
    
contrast_transforms_image = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      
                                        transforms.RandomResizedCrop(size=64,scale=(0.75, 1.1)),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1),
                                              transforms.GaussianBlur(kernel_size=3)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          ToTenser_withoutscale()
                                        #   transforms.ToTensor(),
                                       
                                         ])
    
contrast_transforms = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        #   transforms.RandomResizedCrop(size=64, scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=2),
                                        transforms.RandomResizedCrop(size=64,scale=(0.75, 1.1)),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1),
                                              transforms.GaussianBlur(kernel_size=3)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          
                                        #   transforms.ToTensor(),
                                        #   transforms.Normalize((0.5,), (0.5,))
                                         ])

contrast_transforms_wo_da = transforms.Compose([
                                        transforms.Resize((64,64)),
                                         ])

contrast_transforms_simple = transforms.Compose([
                                        # transforms.RandomHorizontalFlip(),
                                        #   transforms.RandomResizedCrop(size=64, scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=2),
                                        # transforms.RandomResizedCrop(size=64,scale=(0.75, 1.1)),
                                        #   transforms.RandomApply([
                                        #       transforms.ColorJitter(brightness=0.5,
                                        #                              contrast=0.5,
                                        #                              saturation=0.5,
                                        #                              hue=0.1),
                                        #       transforms.GaussianBlur(kernel_size=3)
                                        #   ], p=0.8),
                                        #   transforms.RandomGrayscale(p=0.2),
                                          
                                        #   transforms.ToTensor(),
                                        #   transforms.Normalize((0.5,), (0.5,))
                                        transforms.Resize((64,64)),
                                         ])



supervised_vi_transforms = transforms.Compose([
                                        # transforms.Resize((64,64)),
                                        # transforms.RandomResizedCrop(size=64, scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=2),
                                        # transforms.RandomHorizontalFlip(),
                                        # transforms.RandomResizedCrop(size=64,scale=(0.75, 1.1)),
                                        #   transforms.RandomApply([
                                        #       transforms.ColorJitter(brightness=0.5,
                                        #                              contrast=0.5,
                                        #                              saturation=0.5,
                                        #                              hue=0.1),
                                        #       transforms.GaussianBlur(kernel_size=3)
                                        #   ], p=0.8),
                                        # transforms.RandomGrayscale(p=0.2),
                                          transforms.Resize((64,64)),
                                        #   transforms.ToTensor(),
                                        #   transforms.Normalize((0.5,), (0.5,))
                                         ])

class kfw_dataset(Dataset):

    def __init__(self,mat_pth,im_root,cross_vali, transform, test=False):
        """
        :param mat_pth: path of training/testing lists
        :param im_root: path of training/testing images
        :param cross_vali: number of validation cross e.g [5],[4],...
        :param transform: transformation
        :param test: whether test or not
        """
        self.im_root = im_root
        self.kin_list = self.read_mat(mat_pth)
        self.kin_list = self.get_cross(cross_vali)
        self.trans = transform
        self.test = test

    def read_mat(self,mat_pth):
        mat = scipy.io.loadmat(mat_pth)
        conv_type = lambda ls: [int(ls[0][0]), int(ls[1][0]), str(ls[2][0]), str(ls[3][0])]
        pair_list = [conv_type(ls) for ls in mat['pairs']]
        pair_list = [[item[0],item[1],os.path.join(self.im_root,item[2]),os.path.join(self.im_root,item[3])]
                     for item in pair_list]
        return pair_list

    def get_cross(self,cross_vali):
        return [i for i in self.kin_list if i[0] in cross_vali]

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        im1_pth = self.kin_list[item][2]
        im1  = Image.open(im1_pth)
        im2_pth = self.kin_list[item][3]
        im2  = Image.open(im2_pth)
        label = self.kin_list[item][1]
        im1 = self.trans(im1)
        im2 = self.trans(im2)

        return im1,im2,label,im1_pth,im2_pth


class fiw_dataset(Dataset):

    def __init__(self,mat_pth,im_root,cross_vali, transform, test=False):
        """
        :param mat_pth: path of training/testing lists
        :param im_root: path of training/testing images
        :param cross_vali: number of validation cross e.g [5],[4],...
        :param transform: transformation
        :param test: whether test or not
        """
        self.cache_img = True
        self.cache_dict = {}
        self.im_root = im_root
        self.test = test
        self.kin_list = self.read_mat(mat_pth)
        self.kin_list = self.get_cross(cross_vali)
        self.trans = transform
        


    def read_mat(self,mat_pth):
        with open (mat_pth, 'rb') as fp:
            imgpth_ls = pickle.load(fp)

        pair_list = self.get_img_name(imgpth_ls)
        return pair_list

    # def get_img_name(self,ims_ls):
    #     new_ls = []
    #     for im in ims_ls:
    #         im1_pth = os.path.join(self.im_root,im[2])
    #         im2_pth = os.path.join(self.im_root,im[3])
    #         if not self.test:
    #             im1ls = sorted(os.listdir(im1_pth))
    #             im2ls = sorted(os.listdir(im2_pth))
    #             for im1 in im1ls:
    #                 for im2 in im2ls:
    #                     new_ls.append([im[0], im[1],
    #                                    os.path.join(self.im_root,os.path.join(im[2], im1)),
    #                                    os.path.join(self.im_root,os.path.join(im[3], im2))])
    #
    #
    #         else:
    #             im1_nm = sorted(os.listdir(im1_pth))
    #             im2_nm = sorted(os.listdir(im2_pth))
    #             # lenth = zip(im1_nm,im2_nm)
    #             for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
    #                 new_ls.append([im[0], im[1],
    #                                os.path.join(self.im_root, os.path.join(im[2], im1)),
    #                                os.path.join(self.im_root, os.path.join(im[3], im2))])
    #
    #     return new_ls
    def get_img_name(self, ims_ls):
        new_ls = []
        for im in ims_ls:
            im1_pth = os.path.join(self.im_root, im[2])
            im2_pth = os.path.join(self.im_root, im[3])
            if not self.test:
                im1ls = sorted(os.listdir(im1_pth))
                im2ls = sorted(os.listdir(im2_pth))
                for im1 in im1ls:
                    for im2 in im2ls:
                        im1_ = os.path.join(self.im_root, os.path.join(im[2], im1))
                        im2_ = os.path.join(self.im_root, os.path.join(im[3], im2))
                        new_ls.append([im[0], im[1], im1_, im2_])
                        if not (im1_ in self.cache_dict):
                            im1_cache = Image.open(im1_)
                            self.cache_dict[im1_] = im1_cache.copy()
                        if not (im2_ in self.cache_dict):
                            im2_cache = Image.open(im2_)
                            self.cache_dict[im2_] = im2_cache.copy()
            else:
                im1_nm = sorted(os.listdir(im1_pth))
                im2_nm = sorted(os.listdir(im2_pth))
                # lenth = zip(im1_nm,im2_nm)
                for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
                    im1_ = os.path.join(self.im_root, os.path.join(im[2], im1))
                    im2_ = os.path.join(self.im_root, os.path.join(im[3], im2))
                    new_ls.append([im[0], im[1],im1_,im2_])
                    if not (im1_ in self.cache_dict):
                        img1_cache = Image.open(im1_)
                        self.cache_dict[im1_] = img1_cache.copy()
                    if not (im2_ in self.cache_dict):
                        img2_cache = Image.open(im2_)
                        self.cache_dict[im2_] = img2_cache.copy()

        return new_ls

    def get_cross(self,cross_vali):
        return [i for i in self.kin_list if i[0] in cross_vali]

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        im1_pth = self.kin_list[item][2]
        # im1  = Image.open(im1_pth)
        im1 = self.cache_dict[im1_pth]
        im2_pth = self.kin_list[item][3]
        # im2  = Image.open(im2_pth)
        im2 = self.cache_dict[im2_pth]
        label = self.kin_list[item][1]
        im1 = self.trans(im1)
        im2 = self.trans(im2)

        return im1,im2,label

# class fiw_dataset(Dataset):

#     def __init__(self,mat_pth,im_root,cross_vali, transform, test=False):
#         """
#         :param mat_pth: path of training/testing lists
#         :param im_root: path of training/testing images
#         :param cross_vali: number of validation cross e.g [5],[4],...
#         :param transform: transformation
#         :param test: whether test or not
#         """
#         self.im_root = im_root
#         self.test = test
#         self.kin_list = self.read_mat(mat_pth)
#         self.kin_list = self.get_cross(cross_vali)
#         self.trans = transform


#     def read_mat(self,mat_pth):
#         with open (mat_pth, 'rb') as fp:
#             imgpth_ls = pickle.load(fp)

#         pair_list = self.get_img_name(imgpth_ls)
#         return pair_list

#     def get_img_name(self,ims_ls):
#         new_ls = []
#         for im in ims_ls:
#             im1_pth = os.path.join(self.im_root,im[2])
#             im2_pth = os.path.join(self.im_root,im[3])
#             if not self.test:
#                 im1ls = sorted(os.listdir(im1_pth))
#                 im2ls = sorted(os.listdir(im2_pth))
#                 for im1 in im1ls:
#                     for im2 in im2ls:
#                         new_ls.append([im[0], im[1],
#                                        os.path.join(self.im_root,os.path.join(im[2], im1)),
#                                        os.path.join(self.im_root,os.path.join(im[3], im2))])

#             else:
#                 im1_nm = sorted(os.listdir(im1_pth))
#                 im2_nm = sorted(os.listdir(im2_pth))
#                 # lenth = zip(im1_nm,im2_nm)
#                 for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
#                     new_ls.append([im[0], im[1],
#                                    os.path.join(self.im_root, os.path.join(im[2], im1)),
#                                    os.path.join(self.im_root, os.path.join(im[3], im2))])

#         return new_ls

#     def get_cross(self,cross_vali):
#         return [i for i in self.kin_list if i[0] in cross_vali]

#     def __len__(self):
#         return len(self.kin_list)

#     def __getitem__(self, item):

#         if torch.is_tensor(item):
#             item = item.tolist()
#         im1_pth = self.kin_list[item][2]
#         im1  = Image.open(im1_pth)
#         im2_pth = self.kin_list[item][3]
#         im2  = Image.open(im2_pth)
#         label = self.kin_list[item][1]
#         im1 = self.trans(im1)
#         im2 = self.trans(im2)

#         return im1,im2,label


# class nemo_dataset(Dataset):
#     def __init__(self,list_path,img_root, cross_vali,transform,
#                  test = False):
#
#         """
#         :param list_path:       folder list of training/testing dataset
#         :param img_root:     image path
#         :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
#         :param transform:     add data augmentation
#         :param sf_sequence:   shuffle the sequence order while training
#         :param cross_shuffle: shuffle names among pair list
#         :param sf_aln:        whether shuffle all names or only img2s' names
#         :param test:          weather test
#         """
#         # kin_list is the whole 1,2,3,4,5 folds from mat
#         self.kin_list = self._read_nemols(list_path)
#         self.im_root = img_root
#         self.trans = transform
#         self.cout = 0
#         self.test = test
#
#         if cross_vali is not None:
#             #extract matched folds e.g. [1,2,4,5]
#             self.kin_list = self._get_cross(cross_vali)
#
#
#     def __len__(self):
#         return len(self.kin_list)
#
#     def __getitem__(self, item):
#
#         if torch.is_tensor(item):
#             item = item.tolist()
#         im1_th = self.kin_list[item][2]
#         mem1 = 'f_{}/m_{}/0.mp4'.format(im1_th.split('-')[0][1:],im1_th.split('-')[1])
#         im1_pth = os.path.join(self.im_root,mem1)
#         im1_frames, _, _ = read_video(str(im1_pth))
#
#         im2_th = self.kin_list[item][3]
#         mem2 = 'f_{}/m_{}/0.mp4'.format(im2_th.split('-')[0][1:], im2_th.split('-')[1])
#         im2_pth = os.path.join(self.im_root, mem2)
#         im2_frames, _, _ = read_video(str(im2_pth))
#
#         label = self.kin_list[item][1]
#
#         im1_frames = torch.permute(im1_frames,(3,0,1,2))[:,::4,:,:]#[CTHW]
#         im2_frames = torch.permute(im2_frames,(3,0,1,2))[:,::4,:,:]
#         if self.trans:
#
#             im1_frames = self.trans(im1_frames)
#             im2_frames = self.trans(im2_frames)
#
#         return im1_frames,im2_frames,label
#
#
#     def _read_nemols(self,nemo_ls):
#
#         with open (nemo_ls, 'rb') as fp:
#             nemo_ls = pickle.load(fp)
#
#         return nemo_ls
#
#
#     def _get_cross(self,cross):
#
#         return [i for i in self.kin_list if i[0] in cross]

class nemo_dataset(Dataset):
    def __init__(self,list_path,img_root, cross_vali,transform,
                 test = False):

        """
        :param list_path:       folder list of training/testing dataset
        :param img_root:     image path
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param sf_sequence:   shuffle the sequence order while training
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param test:          weather test
        """
        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.im_root = img_root
        self.img_dict = {}
        self.trans = transform
        self.cout = 0
        self.test = test
        self.kin_list = self._read_nemols(list_path)
        self.get_img_dict(self.kin_list)
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)


    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        im1_th = self.kin_list[item][2]

        im1_frames = self.img_dict[im1_th]

        im2_th = self.kin_list[item][3]

        im2_frames = self.img_dict[im2_th]

        label = self.kin_list[item][1]

        # im1_frames = torch.permute(im1_frames,(3,0,1,2))
        # im2_frames = torch.permute(im2_frames,(3,0,1,2))
        if self.trans:

            im1_frames = self.trans(im1_frames)
            im2_frames = self.trans(im2_frames)

        return im1_frames,im2_frames,label


    def _read_nemols(self,nemo_ls):

        with open (nemo_ls, 'rb') as fp:
            nemo_ls = pickle.load(fp)

        return nemo_ls

    def _addto_img(self,it_nemo):
        if not it_nemo in self.img_dict:
            # load imgs and pack
            mem = 'f_{}/m_{}/0.mp4'.format(it_nemo.split('-')[0][1:], it_nemo.split('-')[1])
            im_pth = os.path.join(self.im_root, mem)
            im_frames, _, _ = read_video(str(im_pth))  # [T, H, W, C]
            im_frames = torch.permute(im_frames, (3, 0, 1, 2))[:,::4,:,:]#[CTHW]
            self.img_dict[it_nemo] = im_frames

    def get_img_dict(self,nemo_ls):
        # if img not in dict
        print("-"*20,'start loading frames')
        for it_nemo in tqdm(nemo_ls):
            # load imgs and pack
            self._addto_img(it_nemo[2])
            self._addto_img(it_nemo[3])
        print("-" * 20, 'end loading frames')

    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]

class nemo_image_dataset(Dataset):
    def __init__(self,list_path,img_root, cross_vali,transform,
                 test = False):

        """
        :param list_path:       folder list of training/testing dataset
        :param img_root:     image path
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param sf_sequence:   shuffle the sequence order while training
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param test:          weather test
        """
        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.im_root = img_root
        self.img_dict = {}
        self.trans = transform
        self.cout = 0
        self.test = test
        self.kin_list = self._read_nemols(list_path)
        self.get_img_dict(self.kin_list)
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)


    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        im1_th = self.kin_list[item][2]

        im1_frames = self.img_dict[im1_th]

        im2_th = self.kin_list[item][3]

        im2_frames = self.img_dict[im2_th]

        label = self.kin_list[item][1]

        # im1_frames = torch.permute(im1_frames,(3,0,1,2))
        # im2_frames = torch.permute(im2_frames,(3,0,1,2))
        if self.trans:

            im1_frames = self.trans(im1_frames).type(torch.float)
            im2_frames = self.trans(im2_frames).type(torch.float)

        return im1_frames,im2_frames,label


    def _read_nemols(self,nemo_ls):

        with open (nemo_ls, 'rb') as fp:
            nemo_ls = pickle.load(fp)

        return nemo_ls


    def _addto_img(self,it_nemo):
        if not it_nemo in self.img_dict:
            # load imgs and pack
            # frm_root = self.im_root.replace('family','crop_frames')
            # mem = 'f_{}/m_{}/'.format(it_nemo.split('-')[0][1:], it_nemo.split('-')[1])
            # mem_pth = os.path.join(frm_root, mem)
            # frm_ls = sorted(os.listdir(mem_pth),key=lambda x:int(x.split('.')[0]))
            frm_root = self.im_root
            mem = it_nemo
            mem_pth = os.path.join(frm_root, mem)
            # frm_ls = sorted(os.listdir(mem_pth),key=lambda x:int(x.split('.')[0]))
            frm_ls = sorted(os.listdir(mem_pth))

            frms = []
            for frm in frm_ls:
                frm_pth = os.path.join(mem_pth,frm)
                tem_img = read_image(frm_pth)
                frms.append(tem_img)
            im_frames = torch.stack(frms,dim=0)# [T,C, H, W]

            im_frames = torch.permute(im_frames, (1, 0, 2, 3))[:,::4,:,:]#[CTHW]--> [CHW]
            self.img_dict[it_nemo] = im_frames

    def get_img_dict(self,nemo_ls):
        # if img not in dict
        print("-"*20,'start loading frames')
        for it_nemo in tqdm(nemo_ls):
            # load imgs and pack
            self._addto_img(it_nemo[2])
            self._addto_img(it_nemo[3])
        print("-" * 20, 'end loading frames')

    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]



class nemo_image_dataset2(Dataset):
    def __init__(self,list_path,img_root, cross_vali,transform,
                 test = False):

        """
        :param list_path:       folder list of training/testing dataset
        :param img_root:     image path
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param sf_sequence:   shuffle the sequence order while training
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param test:          weather test
        """
        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.im_root = img_root
        self.img_dict = {}
        self.trans = transform
        self.cout = 0
        self.test = test
        temp_ls = self._read_nemols(list_path)
        self.kin_list = self.get_img_dict(temp_ls)
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)


    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        im1_th = self.kin_list[item][2]

        im1_frames = self.img_dict[im1_th]

        im2_th = self.kin_list[item][3]

        im2_frames = self.img_dict[im2_th]

        label = self.kin_list[item][1]

        # im1_frames = torch.permute(im1_frames,(3,0,1,2))
        # im2_frames = torch.permute(im2_frames,(3,0,1,2))
        if self.trans:

            im1_frames = self.trans(im1_frames).type(torch.float)
            im2_frames = self.trans(im2_frames).type(torch.float)

        return im1_frames,im2_frames,label


    def _read_nemols(self,nemo_ls):

        with open (nemo_ls, 'rb') as fp:
            nemo_ls = pickle.load(fp)

        return nemo_ls


    def _addto_img(self,it_nemo):
        if not it_nemo in self.img_dict:
            # load imgs and pack
            # frm_root = self.im_root.replace('family','crop_frames')
            # mem = 'f_{}/m_{}/'.format(it_nemo.split('-')[0][1:], it_nemo.split('-')[1])
            frm_root = self.im_root
            mem = it_nemo
            mem_pth = os.path.join(frm_root, mem)
            # frm_ls = sorted(os.listdir(mem_pth),key=lambda x:int(x.split('.')[0]))
            frm_ls = sorted(os.listdir(mem_pth))
            frms = []
            for frm in frm_ls:
                frm_pth = os.path.join(mem_pth,frm)
                # tem_img = read_image(frm_pth)
                frms.append(frm_pth)
            return frms

    def get_img_dict(self,nemo_ls):
        # if img not in dict
        print("-"*20,'start loading frames')
        new_ls = []
        for it_nemo in tqdm(nemo_ls):
            # load imgs and pack
            # self._addto_img(it_nemo[2])
            # self._addto_img(it_nemo[3])
            # if not it_nemo in self.img_dict:
            # load imgs and pack
            cs = it_nemo[0]
            lb = it_nemo[1]
            temp1 = it_nemo[2]
            temp2 = it_nemo[3]
            frms_1 = self._addto_img(temp1)
            frms_2 = self._addto_img(temp2)
            for frms_1_pth, frms_2_pth in zip(frms_1, frms_2):
                new_ls.append([cs,lb,frms_1_pth,frms_2_pth])
                if frms_1_pth not in self.img_dict:
                    self.img_dict[frms_1_pth] = read_image(frms_1_pth)
                if frms_2_pth not in self.img_dict:
                    self.img_dict[frms_2_pth] = read_image(frms_2_pth)    
        print("-" * 20, 'end loading frames')
        return new_ls

    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]
    
    
# ‘YouTube Faces DB’ dataset
class Youtube_Faces_DB():
    def __init__(self,img_root='/local/ytb_crop',transform=None,
                 test = False):

        """
        :param list_path:       folder list of training/testing dataset
        :param img_root:     image path
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param sf_sequence:   shuffle the sequence order while training
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param test:          weather test
        """
        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.im_root = img_root
        self.img_dict = []
        self.trans = transform
        self.cout = 0
        self.test = test
        self.read_YTF(img_root)
      
    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        img_frms = self.img_dict[item]
        if self.trans:
            im1_frames = self.trans(img_frms)
        
        return im1_frames


    def read_YTF(self,img_root):
        if os.path.exists('./data/ytf.pickle'):
            with open('./data/ytf.pickle', 'rb') as handle:
                self.img_dict = pickle.load(handle)
            print('lenth of YTF: ',len(self.img_dict))
        else:
            ######## min len frames is 48 
            print("-"*20,'start loading frames')
            p_folder_ls = sorted(os.listdir(img_root))
            for p_folder in  tqdm(p_folder_ls):
                p_folder_pth = os.path.join(img_root,p_folder)
                c_folder_ls = sorted(os.listdir(p_folder_pth))
                for c_folder in c_folder_ls:
                    c_folder_pth = os.path.join(p_folder_pth,c_folder)
                    frm_ls = sorted(os.listdir(c_folder_pth))
                    frms = []
                    for frm in frm_ls[:48]:
                        frm_pth = os.path.join(c_folder_pth,frm)
                        tem_img = read_image(frm_pth) #check channel
                        frms.append(tem_img)
                    # im_frames = torch.stack(frms,dim=0)# [T,C, H, W]
                    # im_frames = torch.permute(im_frames, (1, 0, 2, 3))#[CTHW]--> [CTHW]
                    self.img_dict.append(frms)
            print("-" * 20, 'end loading frames')
            print('-'*20,'saving img_dict')
            with open('./data/ytf.pickle', 'wb') as handle:
                pickle.dump(self.img_dict, handle)



# ‘YouTube Faces DB’ dataset
class Unsupervise_FIW():
    def __init__(self,img_root='/home/DATA/FIW/origin/train-faces',transform=None,
                 test = False):

        """
        :param list_path:       folder list of training/testing dataset
        :param img_root:     image path
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param sf_sequence:   shuffle the sequence order while training
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param test:          weather test
        """
        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.im_root = img_root
        self.img_dict = []
        self.trans = transform
        self.cout = 0
        self.test = test
        self.read_FIW(img_root)
      
    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        img_frms = self.img_dict[item]
        if self.trans:
            im1_frames = self.trans(img_frms)
        
        return im1_frames


    def read_FIW(self,img_root):
        if os.path.exists('./data/fiw.pickle'):
            with open('./data/fiw.pickle', 'rb') as handle:
                self.img_dict = pickle.load(handle)
            print('lenth of FIW: ',len(self.img_dict))
        else:
            ######## min len frames is 48 
            print("-"*20,'start loading frames')
            p_folder_ls = sorted(os.listdir(img_root))
            for p_folder in  tqdm(p_folder_ls):
                p_folder_pth = os.path.join(img_root,p_folder)
                c_folder_ls = sorted(os.listdir(p_folder_pth))
                c_folder_ls = [item for item in c_folder_ls if not item.endswith('csv') and item != 'unrelated_and_nonfaces']
                for c_folder in c_folder_ls:
                    c_folder_pth = os.path.join(p_folder_pth,c_folder)
                    frm_ls = sorted(os.listdir(c_folder_pth))
                    # frms = []
                    for frm in frm_ls:
                        frm_pth = os.path.join(c_folder_pth,frm)
                        tem_img = read_image(frm_pth) #check channel
                        # frms.append(tem_img)
                        self.img_dict.append(tem_img)
                    # im_frames = torch.stack(frms,dim=0)# [T,C, H, W]
                    # im_frames = torch.permute(im_frames, (1, 0, 2, 3))#[CTHW]--> [CTHW]
                    # self.img_dict.append(frms)
            print("-" * 20, 'end loading frames')
            print('-'*20,'saving img_dict')
            with open('./data/fiw.pickle', 'wb') as handle:
                pickle.dump(self.img_dict, handle)
                
                

class Unsupervise_Nemo():
    def __init__(self,img_root='/local/Nemo/kin_simple/frames/',transform=None,
                 test = False):

        """
        :param list_path:       folder list of training/testing dataset
        :param img_root:     image path
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param sf_sequence:   shuffle the sequence order while training
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param test:          weather test
        """
        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.im_root = img_root
        self.img_dict = []
        self.trans = transform
        self.cout = 0
        self.test = test
        self.read_nemo(img_root)
      
    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        img_frms = self.img_dict[item]
        if self.trans:
            im1_frames = self.trans(img_frms)
        
        return im1_frames

        
    def read_nemo(self,img_root):
        if os.path.exists('./data/nemo_unsupervise.pickle'):
            with open('./data/nemo_unsupervise.pickle', 'rb') as handle:
                self.img_dict = pickle.load(handle)
            print('lenth of FIW: ',len(self.img_dict))
        else:
            ######## min len frames is 48 
            check_ls = []
            print("-"*20,'start loading frames')
            p_folder_ls = sorted(os.listdir(img_root))
            for p_folder in  tqdm(p_folder_ls):
                p_folder_pth = os.path.join(img_root,p_folder)
                c_folder_ls = sorted(os.listdir(p_folder_pth))
                
                for c_folder in c_folder_ls:
                    if c_folder not in check_ls:
                        check_ls.append(c_folder)
                        c_folder_pth = os.path.join(p_folder_pth,c_folder)
                        frm_ls = sorted(os.listdir(c_folder_pth))
                        frms = []
                        for frm in frm_ls:
                            frm_pth = os.path.join(c_folder_pth,frm)
                            tem_img = read_image(frm_pth) #check channel
                            frms.append(tem_img)
                            # self.img_dict.append(tem_img)
                        # im_frames = torch.stack(frms,dim=0)# [T,C, H, W]
                        # im_frames = torch.permute(im_frames, (1, 0, 2, 3))#[CTHW]--> [CTHW]
                        self.img_dict.append(frms)
            print("-" * 20, 'end loading frames')
            print('-'*20,'saving img_dict')
            with open('./data/nemo_unsupervise.pickle', 'wb') as handle:
                pickle.dump(self.img_dict, handle)

class Supervise_FIW():
    def __init__(self,img_root='/home/DATA/FIW/origin/train-faces',label_root = '/home/DATA/FIW/origin/fitted_original_5split',transform=None,
                 test = False):

        """
        :param list_path:       folder list of training/testing dataset
        :param img_root:     image path
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param sf_sequence:   shuffle the sequence order while training
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param test:          weather test
        """
        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.im_root = img_root
        self.label_root = label_root
        self.img_list = []
        self.trans = transform
        self.cout = 0
        self.test = test
        self.cache_dict = {}
        self.read_FIW_identification()
        
        if test:
            self.img_list = self.img_list[:10000]
        else:
            self.img_list = self.img_list[10000:]
        
      
    def __len__(self):
        return len(self.img_list)
    
    def read_label(self,mat_pth):
        with open (mat_pth, 'rb') as fp:
            imgpth_ls = pickle.load(fp)
        return imgpth_ls

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        im1_pth = self.img_list[item][2]
        # im1  = Image.open(im1_pth)
        im1 = self.cache_dict[im1_pth]
        im2_pth = self.img_list[item][3]
        # im2  = Image.open(im2_pth)
        im2 = self.cache_dict[im2_pth]
        label = self.img_list[item][1]
        im1 = self.trans(im1)
        im2 = self.trans(im2)

        return im1,im2,label

    def read_FIW_identification(self):
        tp_dict = {'fd':1,'fs':2,'md':3,'ms':4,'bb':5,'bs':6,
              'ss':7,'gfgd':8,'gfgs':9,'gmgd':10,'gmgs':11}
        
        for tp in tp_dict:
            label_pth = os.path.join(self.label_root,'{}.pkl'.format(tp))
            
            pair_list = self.read_label(label_pth)
            
            for pair in pair_list:
                img1_pth = os.path.join(self.im_root,pair[2])
                img2_pth = os.path.join(self.im_root,pair[3])
                im1ls = sorted(os.listdir(img1_pth))
                im2ls = sorted(os.listdir(img2_pth))
                temp_cs_ls = pair[0]
                temp_tp_lb = 0 if pair[1]==0 else tp_dict[tp]
                for im1 in im1ls:
                    for im2 in im2ls:
                        im1_ = os.path.join(img1_pth,im1)
                        im2_ = os.path.join(img2_pth,im2)
                        self.img_list.append([temp_cs_ls,temp_tp_lb,im1_,im2_])
                        if not (im1_ in self.cache_dict):
                            im1_cache = Image.open(im1_)
                            self.cache_dict[im1_] = im1_cache.copy()
                        if not (im2_ in self.cache_dict):
                            im2_cache = Image.open(im2_)
                            self.cache_dict[im2_] = im2_cache.copy()                        
    def read_FIW_classification(self):
        pass
    
class Supervise_Youtube_Faces_DB():
    def __init__(self,img_root='/local/ytb_crop',
                 label_root='./data/ytb_splits.txt',transform=None,
                 test = False):

        self.im_root = img_root
        self.label_root = label_root
        self.img_dict = {}
        self.trans = transform
        self.cout = 0
        self.test = test
        self.read_YTF_pair()
        
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        im1_pth = self.img_list[item][2]
        # im1  = Image.open(im1_pth)
        im1 = self.img_dict[im1_pth]
        im2_pth = self.img_list[item][3]
        # im2  = Image.open(im2_pth)
        im2 = self.img_dict[im2_pth]
        label = self.img_list[item][1]
        im1 = self.trans(im1)
        im2 = self.trans(im2)

        return im1,im2,label
    
    def read_txt(self,txt_pth):
        with open(txt_pth,'r') as f:
            lines = f.readlines()
        new_lines = [item.split(',') for item in lines[1:]]
        new_lines = [[int(item[0]),int(item[4][1]),item[2][1:],item[3][1:]] for item in new_lines]
        return new_lines


    def read_YTF_pair(self):
        if not self.test:
            self.img_list = self.read_txt(self.label_root)
            self.img_list = [item for item in self.img_list if item[0]!=10]
            if os.path.exists('./data/supervise_ytf_train.pickle'):
                with open('./data/supervise_ytf_train.pickle', 'rb') as handle:
                    self.img_dict = pickle.load(handle)
                print('lenth of supervise_ytf_train: ',len(self.img_dict))
            else:
                ######## min len frames is 48 
                print("-"*20,'start loading training frames')
                for img_pair_ls in tqdm(self.img_list):
                    vi_ls1 = img_pair_ls[2]
                    vi_ls2 = img_pair_ls[3]
                    vi1_pth = os.path.join(self.im_root,vi_ls1)
                    vi2_pth = os.path.join(self.im_root,vi_ls2)
                    
                    frm_ls1 = sorted(os.listdir(vi1_pth))
                    frms = []
                    for frm in frm_ls1[:48]:
                        frm_pth = os.path.join(vi1_pth,frm)
                        tem_img = read_image(frm_pth) #check channel
                        tem_img = transforms.Resize((64,64))(tem_img)
                        frms.append(tem_img)
                    im_frames = torch.stack(frms,dim=0)# [T,C, H, W]
                    im_frames = torch.permute(im_frames, (1, 0, 2, 3))#[CTHW]--> [CTHW]
                        
                    if vi_ls1 not in self.img_dict:
                        self.img_dict[vi_ls1] = im_frames
        
                    frm_ls2 = sorted(os.listdir(vi2_pth))
                    frms = []
                    for frm in frm_ls2[:48]:
                        frm_pth = os.path.join(vi2_pth,frm)
                        tem_img = read_image(frm_pth) #check channel
                        tem_img = transforms.Resize((64,64))(tem_img)
                        frms.append(tem_img)
                    im_frames = torch.stack(frms,dim=0)# [T,C, H, W]
                    im_frames = torch.permute(im_frames, (1, 0, 2, 3))#[CTHW]--> [CTHW]
                        
                    if vi_ls2 not in self.img_dict:
                        self.img_dict[vi_ls2] = im_frames
                    
                print("-" * 20, 'end loading frames')
                print('-'*20,'saving img_dict')
                with open('./data/supervise_ytf_train.pickle', 'wb') as handle:
                    pickle.dump(self.img_dict, handle)
        else:
            self.img_list = self.read_txt(self.label_root)
            self.img_list = [item for item in self.img_list if item[0]==10]
            if os.path.exists('./data/supervise_ytf_test.pickle'):
                with open('./data/supervise_ytf_test.pickle', 'rb') as handle:
                    self.img_dict = pickle.load(handle)
                print('lenth of supervise_ytf_test: ',len(self.img_dict))
            else:
                ######## min len frames is 48 
                print("-"*20,'start loading testing frames')
                for img_pair_ls in tqdm(self.img_list):
                    vi_ls1 = img_pair_ls[2]
                    vi_ls2 = img_pair_ls[3]
                    vi1_pth = os.path.join(self.im_root,vi_ls1)
                    vi2_pth = os.path.join(self.im_root,vi_ls2)
                    
                    frm_ls1 = sorted(os.listdir(vi1_pth))
                    frms = []
                    for frm in frm_ls1[:48]:
                        frm_pth = os.path.join(vi1_pth,frm)
                        tem_img = read_image(frm_pth) #check channel
                        tem_img = transforms.Resize((64,64))(tem_img)
                        frms.append(tem_img)
                        
                    im_frames = torch.stack(frms,dim=0)# [T,C, H, W]
                    im_frames = torch.permute(im_frames, (1, 0, 2, 3))#[CTHW]--> [CTHW]
                    if vi_ls1 not in self.img_dict:
                        self.img_dict[vi_ls1] = im_frames
        
                    frm_ls2 = sorted(os.listdir(vi2_pth))
                    frms = []
                    for frm in frm_ls2[:48]:
                        frm_pth = os.path.join(vi2_pth,frm)
                        tem_img = read_image(frm_pth) #check channel
                        tem_img = transforms.Resize((64,64))(tem_img)
                        frms.append(tem_img)
                    im_frames = torch.stack(frms,dim=0)# [T,C, H, W]
                    im_frames = torch.permute(im_frames, (1, 0, 2, 3))#[CTHW]--> [CTHW]
                    if vi_ls2 not in self.img_dict:
                        self.img_dict[vi_ls2] = im_frames
                    
                print("-" * 20, 'end loading frames')
                print('-'*20,'saving img_dict')
                with open('./data/supervise_ytf_test.pickle', 'wb') as handle:
                    pickle.dump(self.img_dict, handle)



class Supervise_Youtube_Faces_DB_classification():
    def __init__(self,img_root='/local/ytb_crop',
                 label_root='./data/ytb_splits.txt',transform=None,
                 test = False):

        self.im_root = img_root
        self.label_root = label_root
        self.img_dict = {}
        self.trans = transform
        self.cout = 0
        self.test = test
        self.read_YTF_pair()
        
        
    def __len__(self):
        return len(self.img_label_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        im_pth = self.img_label_list[item][0]
        # im1  = Image.open(im1_pth)
        im = self.img_dict[im_pth]
        label = self.img_label_list[item][1]
        im = self.trans(im)
        

        return im,label
    
    def read_txt(self,txt_pth):
        with open(txt_pth,'r') as f:
            lines = f.readlines()
        new_lines = [item.split(',') for item in lines[1:]]
        new_lines = [[int(item[0]),int(item[4][1]),item[2][1:],item[3][1:]] for item in new_lines]
        return new_lines


    def read_YTF_pair(self):
        if not self.test:
            if os.path.exists('./data/ytf_train_classification.pickle'):
                with open('./data/ytf_train_classification.pickle', 'rb') as handle:
                    unpacked_dict = pickle.load(handle)
                self.img_dict = unpacked_dict['img_dict']
                self.img_label_list = unpacked_dict['img_label_list']
                print('lenth of supervise_ytf_train: ',len(self.img_dict))
            else:
                self.img_label_list = []
                names_ls = os.listdir(self.im_root)
                
                
                names_ls = names_ls[:int(len(names_ls)*0.9)]
                
                class_dict = {}
                for i, name in enumerate(names_ls):
                    class_dict[name]=i
                
                 ######## min len frames is 48 
                print("-"*20,'start loading training frames')
                
                for name in tqdm(names_ls):
                    name_pth = os.path.join(self.im_root,name)
                    vi_ls = sorted(os.listdir(name_pth))
                    for vi_item in vi_ls:
                        # class_id = os.path.join(name,vi_item)
                        vi_pth = os.path.join(name_pth,vi_item)
                        frm_ls = sorted(os.listdir(vi_pth))
                        frms = []
                        for frm in frm_ls[:48]:
                            frm_pth = os.path.join(vi_pth,frm)
                            tem_img = read_image(frm_pth)
                            tem_img = transforms.Resize((64,64))(tem_img)
                            frms.append(tem_img)     
                        im_frames = torch.stack(frms,dim=0)# [T,C, H, W]
                        im_frames = torch.permute(im_frames, (1, 0, 2, 3))#[CTHW]--> [CTHW]                       
                        self.img_dict[vi_pth] = im_frames
                        self.img_label_list.append([vi_pth,class_dict[name]])
                
                print("-" * 20, 'end loading frames')
                print('-'*20,'saving img_dict')
                with open('./data/ytf_train_classification.pickle', 'wb') as handle:
                    pickle.dump({'img_dict':self.img_dict,'img_label_list':self.img_label_list}, handle)   
                
            
            
            
       

  

def gen_frames(img_pth):
    pth_ls = sorted(os.listdir(img_pth))
    for pth in tqdm(pth_ls):
        fam_pth = os.path.join(img_pth,pth)
        mem_ls = sorted(os.listdir(fam_pth))
        for mem in mem_ls:
            vi_pth = os.path.join(fam_pth,mem)
            frm_pth = vi_pth.replace('family','vi_frames')
            if not os.path.exists(frm_pth):
                os.makedirs(frm_pth)
            im_frames,_,_ = read_video(vi_pth+'/0.mp4')#[T, H, W, C]
            for ind in range(len(im_frames)):
                im_fm = im_frames[ind]
                im_fm = torch.permute(im_fm,(2,0,1))
                write_jpeg(im_fm,frm_pth+'/{}.jpg'.format(ind))

import matplotlib.pyplot as plt

def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        # row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()



class FaceDataset64(Dataset):
    def __init__(self, path_imgrec, rand_mirror,trans = transforms.ToTensor()):
        self.rand_mirror = rand_mirror
        self.trans = trans
        assert path_imgrec
        if path_imgrec:
            imgs_ls_temp = sorted(os.listdir(path_imgrec))
            self.imgs_ls = [os.path.join(path_imgrec,item) for item in imgs_ls_temp]
            self.lb = [int(item.split('_')[-1].split('.')[0]) for item in imgs_ls_temp]

    def __getitem__(self, index):
        
        img_pth = self.imgs_ls[index]
        img = Image.open(img_pth)
        label = self.lb[index]
        if self.trans:
            img = self.trans(img)
        
        return img, label

    def __len__(self):
        return len(self.imgs_ls)
if __name__=='__main__':
    #### 1. check dataloader
    # from torch.utils.data import DataLoader
    # dset = nemo_image_dataset('/home/Documents/DATA/kinship/Nemo/label/train_list/B-B.pkl',
    #              '/home/Documents/DATA/kinship/Nemo/family',
    #              [1,2,3,4],
    #              transform=video_transform_train)
    # # fiw_dataset('/home/Documents/DATA/kinship/KinFaceW-I/meta_data',
    # #             '/home/Documents/DATA/kinship/KinFaceW-I/images/father-dau',
    # #             [1,2,3,4],
    # #             mdr_transform_train)
    # loader = DataLoader(dset,batch_size=3,shuffle = True)
    # loader = iter(loader)
    # vi1,vi2,label = loader.next()
    # print(label)
    #### 2. generate images
    # gen_frames('/home/Documents/DATA/kinship/Nemo/family')
    #### 3. check ytf dataset
    # from torch.utils.data import DataLoader
    # ytf = Youtube_Faces_DB(img_root='/local/frame_images_DB_crop',transform=ContrastiveTransformations(contrast_transforms, n_views=2))
    # loader = DataLoader(ytf,batch_size=3,shuffle = True)
    # for img in loader:
    #     print(img.shape)
    #     break
    #### 4. check ytf augmentation dataset
    # from torch.utils.data import DataLoader
    # import torchvision
    # ytf = Youtube_Faces_DB(img_root='/local/frame_images_DB_crop',transform=ContrastiveTransformations(contrast_transforms, n_views=2))
    # loader = DataLoader(ytf,batch_size=3,shuffle = True)
    # loader = iter(loader)
    # img = loader.next()
    # # for img in loader:
    # # print(img.shape)
    # imgs1 = torch.permute(img[0][1],(1,0,2,3))
    # aug_images1 = [transforms.ToPILImage()(imgs1[i]) for i in range(5)]
    # plot(aug_images1)
    # # imgs1_grid = torchvision.utils.make_grid(imgs1)
    # # npimg = imgs1_grid.numpy()
    # # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # # plt.show()
    # # imgs2 = torch.permute(img[1][1],(1,0,2,3))
    # # aug_images2 = [transforms.ToPILImage()(imgs2[i]) for i in range(5)]
    # # plot(aug_images2)
    # # imgs2_grid = torchvision.utils.make_grid(imgs2)
    # # npimg = imgs2_grid.numpy()
    # # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # # plt.show()
    pass