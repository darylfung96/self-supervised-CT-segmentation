# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import random


class COVIDDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, trainsize, is_data_augment=False, random_cutout=0):
        self.trainsize = trainsize
        self.is_data_augment = is_data_augment
        self.random_cutout = random_cutout
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        if len(edge_root) != 0:
            self.edge_flage = True
            self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.png')]
            self.edges = sorted(self.edges)
        else:
            self.edge_flage = False

        self.filter_files()
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        # augment data
        if self.is_data_augment:
            if random.random() > 0.5:
                crop_size = int(min(image.size) * 0.8)
                # random cropping
                i, j, w, h = transforms.RandomCrop.get_params(image, output_size=(crop_size, crop_size))
                image = TF.crop(image, i, j, w, h)
                gt = TF.crop(gt, i, j, w, h)

            # -- data augmentation --
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                gt = TF.hflip(gt)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                gt = TF.vflip(gt)

            # random cutout
            if self.random_cutout:
                cutout_size = int(min(image.size) * self.random_cutout)
                i, j, w, h = transforms.RandomCrop.get_params(image,
                                                              output_size=(random.randint(0, cutout_size), random.randint(0, cutout_size)))
                color_code = random.randint(0, 255)
                rect = Image.new('RGB', (w, h), (color_code, color_code, color_code))
                image.paste(rect, (i, j))

        # transform image and gt
        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        if self.edge_flage:
            edge = self.binary_loader(self.edges[index])
            edge = self.gt_transform(edge)
            return image, gt, edge
        else:
            return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class IndicesDataset(data.Dataset):
    def __init__(self, images, gts, edges, trainsize, is_data_augment=False, random_cutout=0, is_test=False):
        self.images = images
        self.gts = gts
        self.edges = edges
        self.random_cutout = random_cutout

        if self.edges is not None:
            self.edge_flage = True
        else:
            self.edge_flage = False

        self.is_test = is_test

        self.trainsize = trainsize
        self.is_data_augment = is_data_augment
        self.filter_files()
        self.size = len(self.images)

        self.transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gt_transform_roc = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), Image.NEAREST),
            transforms.ToTensor()])

    def __getitem__(self, index):
        if self.is_test:
            return self.test_get_item(index)
        else:
            return self.train_get_item(index)

    def test_get_item(self, index):
        image = self.rgb_loader(self.images[index])
        image = self.transform(image)  # .unsqueeze(0)
        gt = self.binary_loader(self.gts[index])
        gt_cont = self.gt_transform(gt)
        gt_roc = self.gt_transform_roc(gt)
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        # return image, gt, name, np.array(F.interpolate(image, gt.size, mode='bilinear'))
        return image, gt_cont, gt_roc, name

    def train_get_item(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        # augment data
        if self.is_data_augment:
            if random.random() > 0.5:
                crop_size = int(min(image.size) * 0.8)
                # random cropping
                i, j, w, h = transforms.RandomCrop.get_params(image, output_size=(crop_size, crop_size))
                image = TF.crop(image, i, j, w, h)
                gt = TF.crop(gt, i, j, w, h)

            # -- data augmentation --
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                gt = TF.hflip(gt)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                gt = TF.vflip(gt)

            # random cutout
            if self.random_cutout:
                cutout_size = int(min(image.size) * self.random_cutout)
                i, j, w, h = transforms.RandomCrop.get_params(image,
                                                              output_size=(random.randint(0, cutout_size),
                                                                           random.randint(0, cutout_size)))
                color_code = random.randint(0, 255)
                rect = Image.new('RGB', (w, h), (color_code, color_code, color_code))
                image.paste(rect, (i, j))

        # transform image and gt
        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        if self.edge_flage:
            edge = self.binary_loader(self.edges[index])
            edge = self.gt_transform(edge)
            return image, gt, edge
        else:
            return image, gt


    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts
        self.size = len(self.images)

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')


def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True,
               is_data_augment=False, random_cutout=0):
    dataset = COVIDDataset(image_root, gt_root, edge_root, trainsize, is_data_augment, random_cutout)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=False)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])

        self.gt_transform_roc = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize), Image.NEAREST),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def __len__(self):
        return self.size

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        # return image, gt, name, np.array(F.interpolate(image, gt.size, mode='bilinear'))
        return image, gt, name

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        image = self.transform(image) #.unsqueeze(0)
        gt = self.binary_loader(self.gts[index])
        gt_cont = self.gt_transform(gt)
        gt_roc = self.gt_transform_roc(gt)
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        # return image, gt, name, np.array(F.interpolate(image, gt.size, mode='bilinear'))
        return image, gt_cont, gt_roc, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
