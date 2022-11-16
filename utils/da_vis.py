
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
import os
from tqdm import tqdm

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

plt.rcParams["savefig.bbox"] = 'tight'
# # orig_img = Image.open('/local/ytb_crop/Jamie_Dimon/2/2.437.jpg')
# pt = '/local/ytb_crop/Carly_Gullickson/3/3.132.jpg'

# pth_lst = os.listdir('/local/ytb_crop/Carly_Gullickson/3/')
root_pth = '/local/ytb_crop/Carly_Gullickson/3/'
pth_lst = os.listdir(root_pth)

print(pth_lst)
# for pt in tqdm(pth_lst):
#     # orig_img = Image.open('/local/ytb_crop/Jamie_Dimon/2/2.437.jpg')
#     pt_pth = root_pth+pt
#     orig_img = Image.open(pt_pth)
#     savename = pt

for i_img in tqdm(range(1,13)):
    # if you change the seed, make sure that the randomly-applied transforms
    # properly show that the image can be both transformed and *not* transformed!
    # torch.manual_seed(0)
    pt_pth = '/local/ytb_crop/Carly_Gullickson/3/3.{}.jpg'.format(i_img)
    orig_img = Image.open(pt_pth)
    savename = '3.{}.jpg'.format(i_img)

   
    def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0]) + with_orig
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            row = [orig_img] + row if with_orig else row
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
        # plt.show()
        plt.savefig('./data/da_img_3/{}'.format(savename))
        

    # gray_img = T.Grayscale()(orig_img)
    gray_img = contrast_transforms(orig_img)
    plot([gray_img], cmap='gray')
    orig_img.save('./data/da_img_ori/{}'.format(savename))