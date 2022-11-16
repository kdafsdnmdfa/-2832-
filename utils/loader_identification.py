import scipy.io
import os
import torch
from torch.utils.data import Dataset,DataLoader
import random
from random import shuffle
import pickle
# from utils.transform import crop_full_transform,facenet_trans

from PIL import Image
import copy

class cross_validation():
    """
    get  n-folds cross validation,
    generate [1,2,3,4,...n]
    yeild [1,..del(remove),..n] and [remove]
    """
    def __init__(self,n):
        self.n = n

    def __iter__(self):
        for i in range(self.n,0,-1):
            train_ls = self._tra_ls(i)
            yield train_ls, [i]

    def _tra_ls(self,remove):
        return [i for i in range(1,self.n+1) if i !=remove ]

def read_mat(mat_path):
    mat = scipy.io.loadmat(mat_path)
    conv_type = lambda ls: [int(ls[0][0]),int(ls[1][0]),str(ls[2][0]),str(ls[3][0])]
    pair_list = [conv_type(ls) for ls in mat['pairs']]
    return  pair_list


class KinDataset_condufusion2(Dataset):
    def __init__(self,mat_path,im_root, cross_vali = None,transform = None,
                 sf_sequence = False,cross_shuffle = False,sf_aln = False,test = False,test_each = False,real_sn = False):
        """
        THE dataset of KinfaceW-I/ KinfaceW-II
        :param mat_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """

        # kin_list is the whole 1,2,3,4,5 folds from mat
        if not test_each:
            self.kin_list = self.read_mat(mat_path,im_root)
            self.fd_ls,self.fs_ls,self.md_ls,self.ms_ls = self.read_each_kin_mat(mat_path,im_root)
        else:
            self.kin_list = self.read_test_mat(mat_path,im_root)
        self.im_root = im_root
        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
            if not test_each:
                self.fd_ls = [i for i in self.fd_ls if i[0] in cross_vali]
                self.fs_ls = [i for i in self.fs_ls if i[0] in cross_vali]
                self.md_ls = [i for i in self.md_ls if i[0] in cross_vali]
                self.ms_ls = [i for i in self.ms_ls if i[0] in cross_vali]

        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle
        self.lth = len(self.kin_list)
        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.img2_list = [i[3] for i in self.kin_list]
        self.alln_list = self.get_alln(self.kin_list)
        self.sf_aln = sf_aln
        # self.crs_insect = 0
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        self.test  = test
        if not test_each:
            self.allfd_list = self.get_alln(self.fd_ls)
            self.allfs_list = self.get_alln(self.fs_ls)
            self.allmd_list = self.get_alln(self.md_ls)
            self.allms_list = self.get_alln(self.ms_ls)
            self.get_same_lth()
        if real_sn:
            self.kin_list = self.get_real_sn(self.kin_list)

    def get_real_sn(self,lis):
        """
        get real senario list
        :param ls:
        :return:
        """
        lb_dict = {'fd':1,'fs':2,'md':3,'ms':4}
        new_kin_ls = []
        ls_pool = []
        for ls in lis:
            if ls[2] not in ls_pool:
                ls_pool.append(ls[2])
            if ls[3] not in ls_pool:
                ls_pool.append(ls[3])
        for lsn1 in ls_pool:
            for lsn2 in ls_pool:
                if lsn1[:-6] == lsn2[:-6]:
                    if lsn1.split('_')[2] == lsn2.split('_')[2]:
                        pass
                    else:
                        lb = lb_dict[lsn1.split('_')[0].split('/')[-1]]
                        new_kin_ls.append([0,lb,lsn1,lsn2])
                else:
                    new_kin_ls.append([0, 0, lsn1, lsn2])

        # print(new_kin_ls)
        # new_kin_ls = new_kin_ls[:10000]
        return new_kin_ls

    def get_same_lth(self):
        while len(self.fd_ls)<self.lth:
            self.fd_ls +=self.fd_ls
        self.fd_ls= self.fd_ls[0:self.lth]

        while len(self.fs_ls)<self.lth:
            self.fs_ls +=self.fs_ls
        self.fs_ls= self.fs_ls[0:self.lth]

        while len(self.md_ls)<self.lth:
            self.md_ls +=self.md_ls
        self.md_ls= self.md_ls[0:self.lth]

        while len(self.ms_ls)<self.lth:
            self.ms_ls +=self.ms_ls
        self.ms_ls= self.ms_ls[0:self.lth]

    def get_alln(self,ls):
        all_name = []
        for i in ls:
            all_name.append(i[2])
            all_name.append(i[3])
        return all_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if self.test:

            if torch.is_tensor(item):
                item = item.tolist()
            # extract img1
            img1_p = self.kin_list[item][2]
            img1 = Image.open(img1_p)
            # extract img2
            img2_p = self.kin_list[item][3]
            img2 = Image.open(img2_p)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            imgs = torch.cat((img1,img2))
            # after each epoch, shuffle once
            return imgs, kin, img1_p, img2_p
        else:

            if torch.is_tensor(item):
                item = item.tolist()
            # extract img1
            img1_p = self.kin_list[item][2]
            img1 = Image.open(img1_p)
            # extract img2
            img2_p = self.kin_list[item][3]
            img2 = Image.open(img2_p)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            imgs = torch.cat((img1, img2))

            ###################### fd
            # extract img1
            img1_p = self.fd_ls[item][2]
            img1 = Image.open(img1_p)
            # extract img2
            img2_p = self.fd_ls[item][3]
            img2 = Image.open(img2_p)
            # get kin label 0/1
            kin_fd = self.fd_ls[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            img_fd = torch.cat((img1, img2))
            ###################### fs
            # extract img1
            img1_p = self.fs_ls[item][2]
            img1 = Image.open(img1_p)
            # extract img2
            img2_p = self.fs_ls[item][3]
            img2 = Image.open(img2_p)
            # get kin label 0/1
            kin_fs = self.fs_ls[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            img_fs = torch.cat((img1, img2))
            ###################### md
            # extract img1
            img1_p = self.md_ls[item][2]
            img1 = Image.open(img1_p)
            # extract img2
            img2_p = self.md_ls[item][3]
            img2 = Image.open(img2_p)
            # get kin label 0/1
            kin_md = self.md_ls[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            img_md = torch.cat((img1, img2))
            ###################### ms
            # extract img1
            img1_p = self.ms_ls[item][2]
            img1 = Image.open(img1_p)
            # extract img2
            img2_p = self.ms_ls[item][3]
            img2 = Image.open(img2_p)
            # get kin label 0/1
            kin_ms = self.ms_ls[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            img_ms = torch.cat((img1, img2))

            if self.crf:
                self.cout +=1
                if self.cout == self.lth:
                    self.cout = 0
                    self._cross_shuffle()
                    self.fd_ls = self._cross_shuffle_each_kin(self.fd_ls,self.allfd_list)
                    self.fs_ls = self._cross_shuffle_each_kin(self.fs_ls,self.allfs_list)
                    self.md_ls = self._cross_shuffle_each_kin(self.md_ls,self.allmd_list)
                    self.ms_ls = self._cross_shuffle_each_kin(self.ms_ls,self.allms_list)
                # self.get_same_lth()

            return imgs,img_fd,img_fs,img_md,img_ms, kin,kin_fd,kin_fs,kin_md,kin_ms


    def _cross_shuffle_each_kin(self,kinlis,ls):
        """
        shuffle the second images name after each epoch
        :return:
        """

        im2_ls = ls
        new_pair_list = []

        for pair_l in kinlis:
            if pair_l[1] == 0:
                new_img2 = im2_ls[random.randint(0, len(im2_ls)-1)]
                while pair_l[2].split('/')[-1][:-6] == new_img2.split('/')[-1][:-6]:
                    new_img2 = im2_ls[random.randint(0, len(im2_ls)-1)]
                pair_l[3] = new_img2
            new_pair_list.append(pair_l)

        return  new_pair_list

    def _cross_shuffle(self):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []
        # if self.crs_insect%2 ==0:
        #     ls_bak = copy.deepcopy(self.kin_list_bak)
        for pair_l in self.kin_list:
            if pair_l[1] == 0:
                new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                while pair_l[2].split('/')[-1][:-6] == new_img2.split('/')[-1][:-6]:
                    new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                pair_l[3] = new_img2
            new_pair_list.append(pair_l)
        # else:
        #     for pair_l in self.kin_list:
        #         if pair_l[1] == 0:
        #             new_img1 = im2_ls[random.randint(0, self.lth-1)]
        #             new_img2 = im2_ls[random.randint(0, self.lth-1)]
        #             while new_img1.split('_')[1] == new_img2.split('_')[1]:
        #                 new_img2 = im2_ls[random.randint(0, self.lth-1)]
        #             pair_l[3] = new_img2
        #             pair_l[2] = new_img1
        #         new_pair_list.append(pair_l)
        # infact it's no use to use this line:
        # self.crs_insect +=1
        self.kin_list = new_pair_list
    def read_mat(self,mat_paths,im_roots):
        ls_dict = {}
        for i,(mat_path,im_root) in enumerate(zip(mat_paths,im_roots)):
            kin_list = self._read_mat(mat_path)
            for kl in kin_list:
                if not kl[1]==0:
                    kl[1]=kl[1]+i
                kl[2]=os.path.join(im_root,kl[2])
                kl[3]=os.path.join(im_root,kl[3])
            ls_dict[i]=kin_list
        new_kin_list = []
        for cr_num in range(1,6):
            for kn in ls_dict:
                for ls in ls_dict[kn]:
                    if ls[0] == cr_num:
                        new_kin_list.append(ls)
        return new_kin_list


    def read_each_kin_mat(self,mat_paths,im_roots):
        fd_ls = self.read_test_mat(mat_paths[0],im_roots[0])
        fs_ls = self.read_test_mat(mat_paths[1], im_roots[1])
        md_ls = self.read_test_mat(mat_paths[2], im_roots[2])
        ms_ls = self.read_test_mat(mat_paths[3], im_roots[3])
        return fd_ls,fs_ls,md_ls,ms_ls


    def read_test_mat(self,mat_path,im_root):
        kin_list = self._read_mat(mat_path)
        for kl in kin_list:
            kl[2] = os.path.join(im_root, kl[2])
            kl[3] = os.path.join(im_root, kl[3])
        new_kin_list = kin_list
        return new_kin_list

    def _read_mat(self,mat_path):
        mat = scipy.io.loadmat(mat_path)
        conv_type = lambda ls: [int(ls[0][0]), int(ls[1][0]), str(ls[2][0]), str(ls[3][0])]
        pair_list = [conv_type(ls) for ls in mat['pairs']]
        return pair_list

    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]


###################################### fiw
class fiw_4type(Dataset):
    def __init__(self,mat_path,im_root, cross_vali = None,transform = None,
                 sf_sequence = False,cross_shuffle = False,sf_aln = False,test = False,test_each = False,real_sn = False):
        """
        THE dataset of KinfaceW-I/ KinfaceW-II
        :param mat_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """
        self.im_root = im_root
        # kin_list is the whole 1,2,3,4,5 folds from mat
        if not test_each:
            self.kin_list = self.read_mat(mat_path,im_root)
            self.fd_ls,self.fs_ls,self.md_ls,self.ms_ls = self.read_each_kin_mat(mat_path,im_root)
        else:
            self.kin_list = self.read_test_mat(mat_path,im_root)

        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
            if not test_each:
                self.fd_ls = [i for i in self.fd_ls if i[0] in cross_vali]
                self.fs_ls = [i for i in self.fs_ls if i[0] in cross_vali]
                self.md_ls = [i for i in self.md_ls if i[0] in cross_vali]
                self.ms_ls = [i for i in self.ms_ls if i[0] in cross_vali]

        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle
        self.lth = len(self.kin_list)
        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.img2_list = [i[3] for i in self.kin_list]
        self.alln_list = self.get_alln(self.kin_list)
        self.sf_aln = sf_aln
        # self.crs_insect = 0
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        self.test  = test
        if not test_each:
            self.allfd_list = self.get_alln(self.fd_ls)
            self.allfs_list = self.get_alln(self.fs_ls)
            self.allmd_list = self.get_alln(self.md_ls)
            self.allms_list = self.get_alln(self.ms_ls)
            self.get_same_lth()
        if real_sn:
            self.kin_list = self.get_real_sn(self.kin_list)

    def get_real_sn(self,lis):
        """
        get real senario list
        :param ls:
        :return:
        """
        lb_dict = {'fd':1,'fs':2,'md':3,'ms':4}
        new_kin_ls = []
        ls_pool = []
        for ls in lis:
            if ls[2] not in ls_pool:
                ls_pool.append(ls[2])
            if ls[3] not in ls_pool:
                ls_pool.append(ls[3])
        for lsn1 in ls_pool:
            for lsn2 in ls_pool:
                if lsn1[:-6] == lsn2[:-6]:
                    if lsn1.split('_')[2] == lsn2.split('_')[2]:
                        pass
                    else:
                        lb = lb_dict[lsn1.split('_')[0].split('/')[-1]]
                        new_kin_ls.append([0,lb,lsn1,lsn2])
                else:
                    new_kin_ls.append([0, 0, lsn1, lsn2])

        # print(new_kin_ls)
        # new_kin_ls = new_kin_ls[:10000]
        return new_kin_ls

    def get_same_lth(self):
        while len(self.fd_ls)<self.lth:
            self.fd_ls +=self.fd_ls
        self.fd_ls= self.fd_ls[0:self.lth]

        while len(self.fs_ls)<self.lth:
            self.fs_ls +=self.fs_ls
        self.fs_ls= self.fs_ls[0:self.lth]

        while len(self.md_ls)<self.lth:
            self.md_ls +=self.md_ls
        self.md_ls= self.md_ls[0:self.lth]

        while len(self.ms_ls)<self.lth:
            self.ms_ls +=self.ms_ls
        self.ms_ls= self.ms_ls[0:self.lth]

    def get_alln(self,ls):
        all_name = []
        for i in ls:
            all_name.append(i[2])
            all_name.append(i[3])
        return all_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if self.test:

            if torch.is_tensor(item):
                item = item.tolist()
            # extract img1
            img1_p = self.kin_list[item][2]
            img1 = Image.open(img1_p)
            # extract img2
            img2_p = self.kin_list[item][3]
            img2 = Image.open(img2_p)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            imgs = torch.cat((img1,img2))
            # after each epoch, shuffle once
            return imgs, kin, img1_p, img2_p
        else:

            if torch.is_tensor(item):
                item = item.tolist()
            # extract img1
            img1_p = self.kin_list[item][2]
            img1 = Image.open(img1_p)
            # extract img2
            img2_p = self.kin_list[item][3]
            img2 = Image.open(img2_p)
            # get kin label 0/1
            kin = self.kin_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            imgs = torch.cat((img1, img2))

            ###################### fd
            # extract img1
            img1_p = self.fd_ls[item][2]
            img1 = Image.open(img1_p)
            # extract img2
            img2_p = self.fd_ls[item][3]
            img2 = Image.open(img2_p)
            # get kin label 0/1
            kin_fd = self.fd_ls[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            img_fd = torch.cat((img1, img2))
            ###################### fs
            # extract img1
            img1_p = self.fs_ls[item][2]
            img1 = Image.open(img1_p)
            # extract img2
            img2_p = self.fs_ls[item][3]
            img2 = Image.open(img2_p)
            # get kin label 0/1
            kin_fs = self.fs_ls[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            img_fs = torch.cat((img1, img2))
            ###################### md
            # extract img1
            img1_p = self.md_ls[item][2]
            img1 = Image.open(img1_p)
            # extract img2
            img2_p = self.md_ls[item][3]
            img2 = Image.open(img2_p)
            # get kin label 0/1
            kin_md = self.md_ls[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            img_md = torch.cat((img1, img2))
            ###################### ms
            # extract img1
            img1_p = self.ms_ls[item][2]
            img1 = Image.open(img1_p)
            # extract img2
            img2_p = self.ms_ls[item][3]
            img2 = Image.open(img2_p)
            # get kin label 0/1
            kin_ms = self.ms_ls[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            img_ms = torch.cat((img1, img2))

            if self.crf:
                self.cout +=1
                if self.cout == self.lth:
                    self.cout = 0
                    self._cross_shuffle(1)
                    self.fd_ls = self._cross_shuffle_each_kin(self.fd_ls,self.allfd_list)
                    self.fs_ls = self._cross_shuffle_each_kin(self.fs_ls,self.allfs_list)
                    self.md_ls = self._cross_shuffle_each_kin(self.md_ls,self.allmd_list)
                    self.ms_ls = self._cross_shuffle_each_kin(self.ms_ls,self.allms_list)
                # self.get_same_lth()

            return imgs,img_fd,img_fs,img_md,img_ms, kin,kin_fd,kin_fs,kin_md,kin_ms


    def _cross_shuffle_each_kin(self,kinlis,ls):
        """
        shuffle the second images name after each epoch
        :return:
        """

        im2_ls = ls
        new_pair_list = []

        for pair_l in kinlis:
            if pair_l[1] == 0:
                new_img2 = im2_ls[random.randint(0, len(im2_ls)-1)]
                while pair_l[2].split('/')[-1][:-6] == new_img2.split('/')[-1][:-6]:
                    new_img2 = im2_ls[random.randint(0, len(im2_ls)-1)]
                pair_l[3] = new_img2
            new_pair_list.append(pair_l)

        return  new_pair_list



    def _cross_shuffle(self,neg_ratio):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []

        for pair_l in self.kin_list:
            if pair_l[1] == -1:
                for neg_iter in range(neg_ratio):
                    new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('/')[-3] == new_img2.split('/')[-3]:
                        new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                    pair_l[3] = new_img2
                    new_pair_list.append(pair_l)
            else:
                new_pair_list.append(pair_l)

        self.kin_list = new_pair_list


    def read_mat(self,mat_paths,im_roots):
        ### updated
            self.neg_flag = -1
            ls_dict = {}
            for i,mat_path in enumerate(mat_paths):
                kin_list = self._read_mat(mat_path)
                for kl in kin_list:
                    if not kl[1]==0:
                        kl[1]=kl[1]+i
                    else:
                        kl[1] = 0
                    kl[2]=os.path.join(im_roots,kl[2])
                    kl[3]=os.path.join(im_roots,kl[3])
                ls_dict[i]=kin_list
            new_kin_list = []
            for cr_num in range(1,6):
                for kn in ls_dict:
                    for ls in ls_dict[kn]:
                        if ls[0] == cr_num:
                            new_kin_list.append(ls)
            return new_kin_list

    def read_each_kin_mat(self,mat_paths,im_roots):
        fd_ls = self.read_test_mat(mat_paths[0],im_roots)
        fs_ls = self.read_test_mat(mat_paths[1], im_roots)
        md_ls = self.read_test_mat(mat_paths[2], im_roots)
        ms_ls = self.read_test_mat(mat_paths[3], im_roots)
        return fd_ls,fs_ls,md_ls,ms_ls


    def read_test_mat(self,mat_path,im_root):
        kin_list = self._read_mat(mat_path)
        for kl in kin_list:
            kl[2] = os.path.join(im_root, kl[2])
            kl[3] = os.path.join(im_root, kl[3])
        new_kin_list = kin_list
        return new_kin_list

    def _read_mat(self,mat_path):
        with open (mat_path, 'rb') as fp:
            nemo_ls = pickle.load(fp)

        nemo_ls = self.get_img_name(nemo_ls)
        return nemo_ls


    def get_img_name(self,ims_ls):
        new_ls = []
        for im in ims_ls:
            im1_pth = os.path.join(self.im_root,im[2])
            im2_pth = os.path.join(self.im_root,im[3])

            # im1_nm = sorted(os.listdir(im1_pth))[0]
            # im2_nm = sorted(os.listdir(im2_pth))[0]
            # new_ls.append([im[0],im[1],os.path.join(im[2],im1_nm),os.path.join(im[3],im2_nm)])
            im1_nm = sorted(os.listdir(im1_pth))
            im2_nm = sorted(os.listdir(im2_pth))
            # lenth = zip(im1_nm,im2_nm)
            for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
                new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])

        return new_ls

    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]





######################################### KinFaceW #########################################

class Kfw_identification_dataset(Dataset):
    def __init__(self,mat_path,im_root, cross_vali = None,transform = None,
                 cross_shuffle = False,sf_aln = False,
                 test = False,test_each = False,real_sn = False,read_all_imgs = False,
                 get_neg=False,get_pos=False,neg_ratio=1):
        """
        THE dataset of KinfaceW-I/ KinfaceW-II
        :param mat_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """
        self.neg_ratio = neg_ratio
        self.get_neg = get_neg
        self.get_pos = get_pos
        self.read_all_imgs = read_all_imgs
        self.neg_flag = 0
        self.test = test
        self.im_root = im_root
        self.kin_list = self.read_mat(mat_path,im_root)
        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle

        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.img2_list = [i[3] for i in self.kin_list]
        self.alln_list = self.get_alln(self.kin_list)
        self.sf_aln = sf_aln
        # self.crs_insect = 0
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        if real_sn:
            self.kin_list = self.get_real_sn(self.kin_list)

        self.kin_list = self._init_list(self.kin_list)

        self.lth = len(self.kin_list)

    def _init_list(self,ls):
        return ls

    def get_real_sn(self,lis):
        new_kin_ls = []

        return new_kin_ls

    def get_alln(self,ls):
        all_name = []
        for i in ls:
            all_name.append(i[2])
            all_name.append(i[3])
        return all_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        # extract img1
        img1_p = self.kin_list[item][2]
        img1 = Image.open(img1_p)
        # extract img2
        img2_p = self.kin_list[item][3]
        img2 = Image.open(img2_p)
        # get kin label 0/1
        kin = self.kin_list[item][1]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # imgs = torch.cat((img1,img2))
        # after each epoch, shuffle once
        if self.crf:
            self.cout +=1
            if self.cout == self.lth:
                self.cout = 0
                self._cross_shuffle(1)


        return img1,img2,kin

    def _cross_shuffle(self,neg_ratio):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []

        for pair_l in self.kin_list:
            if pair_l[1] == 0:
                new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                while pair_l[2].split('/')[-1][:-6] == new_img2.split('/')[-1][:-6]:
                    new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                pair_l[3] = new_img2
            new_pair_list.append(pair_l)


        self.kin_list = new_pair_list
        return self.kin_list

    def read_mat(self,mat_paths,im_roots):
        ls_dict = {}
        for i,(mat_path,im_root) in enumerate(zip(mat_paths,im_roots)):
            kin_list = self._read_mat(mat_path)
            for kl in kin_list:
                if not kl[1]==0:
                    kl[1]=kl[1]+i
                kl[2]=os.path.join(im_root,kl[2])
                kl[3]=os.path.join(im_root,kl[3])
            ls_dict[i]=kin_list
        new_kin_list = []
        for cr_num in range(1,6):
            for kn in ls_dict:
                for ls in ls_dict[kn]:
                    if ls[0] == cr_num:
                        new_kin_list.append(ls)
        return new_kin_list

    def read_test_mat(self,mat_path,im_root):
        new_kin_list = []
        return new_kin_list

    def _read_mat(self,mat_path):
        mat = scipy.io.loadmat(mat_path)
        conv_type = lambda ls: [int(ls[0][0]), int(ls[1][0]), str(ls[2][0]), str(ls[3][0])]
        pair_list = [conv_type(ls) for ls in mat['pairs']]
        return pair_list

    def get_img_name(self,ims_ls):
        new_ls = []
        return new_ls

    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]








if __name__=='__main__':
    import torchvision.transforms as transforms
    # father-daughter
    train_ls = ['/home/CODE/transformer/transformer-kin/data/KinFaceW-I/meta_data/fd_pairs.mat',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-I/meta_data/fs_pairs.mat',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-I/meta_data/md_pairs.mat',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-I/meta_data/ms_pairs.mat']
    data_pth = ['/home/CODE/transformer/transformer-kin/data/KinFaceW-I/images/father-dau',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-I/images/father-son',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-I/images/mother-dau',
                '/home/CODE/transformer/transformer-kin/data/KinFaceW-I/images/mother-son']
    nemo_data = Kfw_identification_dataset(train_ls,data_pth,[1,2,3,4],transform= transforms.ToTensor(),cross_shuffle =True,sf_aln = True)
    nemoloader = DataLoader(nemo_data,shuffle=True)
    for j in range(3):
        for i,data in enumerate(nemoloader):
            # print(i)
            pass
        print(i)