# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from PIL import Image
import cv2
from Code.utils.onehot import onehot


class LungDataset(Dataset):
    def __init__(self, imgs_path, pseudo_path, label_path, transform=None, is_test=False, is_data_augment=False, is_label_smooth=False, random_cutout=0):
        self.transform = transform
        self.imgs_path = imgs_path  # 'data/class3_images/'
        self.pseudo_path = pseudo_path
        self.label_path = label_path    # 'data/class3_label/'
        self.is_test = is_test
        self.is_data_augment = is_data_augment
        self.is_label_smooth = is_label_smooth
        self.random_cutout = random_cutout
        self.num_class = 3

        self.imgA = []
        self.imgB = []
        self.imgC = []
        img_names = os.listdir(self.imgs_path)
        for img_name in img_names:
            imgA = cv2.resize(cv2.imread(self.imgs_path + img_name), (352, 352))
            imgB = cv2.resize(cv2.imread(self.label_path + img_name.split('.')[0] + '.png', 0), (352, 352))
            imgC = cv2.resize(cv2.imread(self.pseudo_path + img_name.split('.')[0] + '.png', 0), (352, 352))

            # only need to process the original dataset, tr and rp already processed
            if 'tr' not in img_name and 'rp' not in img_name:
                imgB[imgB < 19] = 0
                imgB[(imgB <= 38) & (imgB >= 19)] = 1
                imgB[imgB > 38] = 2

            self.imgA.append(imgA)
            self.imgB.append(imgB)
            self.imgC.append(imgC)

    def __len__(self):
        return len(self.imgA)

    def __getitem__(self, idx):
        imgA = self.imgA[idx]
        imgB = self.imgB[idx]
        imgC = self.imgC[idx]

        if not self.is_test:
            imgB = cv2.resize(imgB, (352, 352))
        img_label = imgB
        # print(np.unique(img_label))
        # make data augmentation here
        if self.is_data_augment:
            # convert to pil format so we can data augment them
            img_label = np.expand_dims(img_label, -1)
            pil_imgA = TF.to_pil_image(imgA)
            pil_img_label = TF.to_pil_image(img_label)
            pil_imgC = TF.to_pil_image(imgC)

            if random.random() > 0.5:
                # random cropping
                crop_size = int(min(imgA.shape[:2]) * 0.8)
                i, j, w, h = transforms.RandomCrop.get_params(pil_imgA, output_size=(crop_size, crop_size))
                pil_imgA = TF.crop(pil_imgA, i, j, w, h)
                pil_img_label = TF.crop(pil_img_label, i, j, w, h)
                pil_imgC = TF.crop(pil_imgC, i, j, w, h)


            # -- data augmentation --
            # Random horizontal flipping
            if random.random() > 0.5:
                pil_imgA = TF.hflip(pil_imgA)
                pil_img_label = TF.hflip(pil_img_label)
                pil_imgC = TF.hflip(pil_imgC)

            # Random vertical flipping
            if random.random() > 0.5:
                pil_imgA = TF.vflip(pil_imgA)
                pil_img_label = TF.vflip(pil_img_label)
                pil_imgC = TF.vflip(pil_imgC)

            # random cutout
            if self.random_cutout:
                if random.random() > 0.5:
                    cutout_size = int(min(imgA.shape[:2]) * self.random_cutout)
                    i, j, w, h = transforms.RandomCrop.get_params(pil_imgA,
                                                              output_size=(random.randint(0, cutout_size),
                                                                           random.randint(0, cutout_size)))
                    color_code = random.randint(0, 255)
                    rect = Image.new('RGB', (w, h), (color_code, color_code, color_code))
                    pil_imgA.paste(rect, (i, j))

            pil_imgA = pil_imgA.resize((352, 352))
            pil_img_label = pil_img_label.resize((352, 352))
            pil_imgC = pil_imgC.resize((352, 352))

            # convert pil back to numpy
            imgA = np.array(pil_imgA)
            img_label = np.array(pil_img_label)
            imgC = np.array(pil_imgC)

        img_label_onehot = (np.arange(self.num_class) == img_label[...,None]).astype(float)
        img_label_onehot = img_label_onehot.transpose(2, 0, 1)  # n_class * w * H

        # label smoothing
        if self.is_label_smooth:
            img_label_onehot[0] = img_label_onehot[0] * 0.9 # since there are so many labels on the first axis, we smooth it

        onehot_label = torch.FloatTensor(img_label_onehot)
        if self.transform:
            imgA = self.transform(imgA)
            imgC = self.transform(imgC)

        # return imgA, imgC, onehot_label, img_name
        return imgA, imgC, onehot_label, None


class IndicesLungDataset(Dataset):
    def __init__(self, img_names, pseudo_path, label_path, transform=None, is_test=False, is_data_augment=False, is_label_smooth=False, random_cutout=0):
        self.transform = transform
        self.img_names = img_names  # 'data/class3_images/'
        self.pseudo_path = pseudo_path
        self.label_path = label_path    # 'data/class3_label/'
        self.is_test = is_test
        self.is_data_augment = is_data_augment
        self.is_label_smooth = is_label_smooth
        self.random_cutout = random_cutout
        self.num_class = 3

    def __len__(self):
        return self.img_names.shape[0]

    def __getitem__(self, idx):
        # processing img
        img_name = self.img_names[idx]
        # image path
        imgA = cv2.imread(img_name)
        imgA = cv2.resize(imgA, (352, 352))

        img_filename = os.path.basename(img_name).split('.')[0]
        # processing pseudo
        imgC = cv2.imread(self.pseudo_path + img_filename.split('.')[0] + '.png')
        imgC = cv2.resize(imgC, (352, 352))

        # processing label
        imgB = cv2.imread(self.label_path + img_filename + '.png', 0)
        if not self.is_test:
            imgB = cv2.resize(imgB, (352, 352))
        img_label = imgB
        # print(np.unique(img_label))
        # make data augmentation here
        if self.is_data_augment:
            # convert to pil format so we can data augment them
            img_label = np.expand_dims(img_label, -1)
            pil_imgA = TF.to_pil_image(imgA)
            pil_img_label = TF.to_pil_image(img_label)
            pil_imgC = TF.to_pil_image(imgC)

            if random.random() > 0.5:
                # random cropping
                crop_size = int(min(imgA.shape[:2]) * 0.8)
                i, j, w, h = transforms.RandomCrop.get_params(pil_imgA, output_size=(crop_size, crop_size))
                pil_imgA = TF.crop(pil_imgA, i, j, w, h)
                pil_img_label = TF.crop(pil_img_label, i, j, w, h)
                pil_imgC = TF.crop(pil_imgC, i, j, w, h)


            # -- data augmentation --
            # Random horizontal flipping
            if random.random() > 0.5:
                pil_imgA = TF.hflip(pil_imgA)
                pil_img_label = TF.hflip(pil_img_label)
                pil_imgC = TF.hflip(pil_imgC)

            # Random vertical flipping
            if random.random() > 0.5:
                pil_imgA = TF.vflip(pil_imgA)
                pil_img_label = TF.vflip(pil_img_label)
                pil_imgC = TF.vflip(pil_imgC)

            # random cutout
            if self.random_cutout:
                if random.random() > 0.5:
                    cutout_size = int(min(imgA.shape[:2]) * self.random_cutout)
                    i, j, w, h = transforms.RandomCrop.get_params(pil_imgA,
                                                              output_size=(random.randint(0, cutout_size),
                                                                           random.randint(0, cutout_size)))
                    color_code = random.randint(0, 255)
                    rect = Image.new('RGB', (w, h), (color_code, color_code, color_code))
                    pil_imgA.paste(rect, (i, j))

            pil_imgA = pil_imgA.resize((352, 352))
            pil_img_label = pil_img_label.resize((352, 352))
            pil_imgC = pil_imgC.resize((352, 352))

            # convert pil back to numpy
            imgA = np.array(pil_imgA)
            img_label = np.array(pil_img_label)
            imgC = np.array(pil_imgC)

        # only need to process the original dataset, tr and rp already processed
        if 'tr' not in img_filename and 'rp' not in img_filename:
            img_label[img_label < 19] = 0
            img_label[(img_label <= 38) & (img_label >= 19)] = 1
            img_label[img_label > 38] = 2

        img_label_onehot = (np.arange(self.num_class) == img_label[...,None]).astype(float)
        img_label_onehot = img_label_onehot.transpose(2, 0, 1)  # n_class * w * H

        # label smoothing
        if self.is_label_smooth:
            img_label_onehot[0] = img_label_onehot[0] * 0.9 # since there are so many labels on the first axis, we smooth it

        onehot_label = torch.FloatTensor(img_label_onehot)
        if self.transform:
            imgA = self.transform(imgA)
            imgC = self.transform(imgC)

        return imgA, imgC, onehot_label, img_name

# if __name__ == '__main__':
#
#     for train_batch in train_dataloader:
#         print(train_batch)
#
#     for test_batch in test_dataloader:
#         print(test_batch)
