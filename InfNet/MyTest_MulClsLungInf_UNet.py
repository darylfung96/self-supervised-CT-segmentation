# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import os
import numpy as np
from Code.utils.dataloader_MulClsLungInf_UNet import LungDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
from fcn8 import create_fcn, FCN8s
from Code.model_lung_infection.InfNet_UNet import *  # 当前用的UNet模型
import imageio
from Code.utils.split_class import split_class
import shutil
import argparse


def inference(num_classes, input_channels, snapshot_dir, save_path, test_path, pseudo_path, model_name):
    test_dataset = LungDataset(
        imgs_path=os.path.join(test_path, 'Imgs'),
        pseudo_path=pseudo_path,  # NOTES: generated from Semi-Inf-Net
        label_path=os.path.join(test_path, 'GT'),
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        is_test=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dict = {'baseline': Inf_Net_UNet, 'improved': Inf_Net_UNet_Improved, 'FCN': create_fcn}
    lung_model = model_dict[model_name](input_channels, num_classes).to(device)
    print(lung_model)
    lung_model.load_state_dict(torch.load(snapshot_dir, map_location=torch.device(device)))
    lung_model.eval()
    os.makedirs(save_path, exist_ok=True)

    for index, (img, pseudo, img_mask, name) in enumerate(test_dataloader):
        img = img.to(device)
        pseudo = pseudo.to(device)
        img_mask = img_mask.to(device)

        inputs = torch.cat((img, pseudo), dim=1)
        if type(lung_model) == FCN8s:
            inputs = img
        output = lung_model(inputs)
        output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
        b, _, w, h = output.size()
        _, _, w_gt, h_gt = img_mask.size()

        # output b*n_class*h*w -- > b*h*w
        pred = output.cpu().permute(0, 2, 3, 1).contiguous().view(-1, num_classes).max(1)[1].view(b, w, h).numpy().squeeze()
        pred_rgb = (np.arange(3) == pred[..., None]).astype(np.float64)
        # swap the rgb content so the background is black instead of red
        pred = np.zeros(pred_rgb.shape)
        pred[:, :, 0] = pred_rgb[:, :, 1]
        pred[:, :, 1] = pred_rgb[:, :, 2]

        # pred = misc.imresize(pred, size=(w_gt, h_gt))
        imageio.imwrite(save_path + name[0].replace('.jpg', '.png'), pred)
        # split_class(save_path, name[0].replace('.jpg', '.png'), w_gt, h_gt) #undo this line for now

    # shutil.rmtree(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str, default='./Snapshots/save_weights/self-multi-inf-net_new/unet_model_38.pkl')
    parser.add_argument('--pseudo_path', type=str, default='./Results/Lung infection segmentation/baseline-inf-net/')
    parser.add_argument('--test_path', type=str, default='./Dataset/TestingSet/MultiClassInfection-Test/')
    parser.add_argument('--save_path', type=str, default='./Results/Multi-class lung infection segmentation/self-multi-inf-net_new/')
    parser.add_argument('--input_channels', type=int, default=6)
    parser.add_argument('--model_name', type=str, default='improved')  # can be baseline or improved
    arg = parser.parse_args()

    inference(num_classes=3,
              input_channels=arg.input_channels,
              snapshot_dir=arg.pth_path,
              save_path=arg.save_path,
              pseudo_path=arg.pseudo_path,
              test_path=arg.test_path,
              model_name=arg.model_name
              )
