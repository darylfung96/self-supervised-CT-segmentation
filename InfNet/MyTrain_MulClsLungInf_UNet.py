# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import os
import torch
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from sklearn.metrics import auc, roc_curve
from Code.utils.dataloader_MulClsLungInf_UNet import LungDataset
from torchvision import transforms
# from LungData import test_dataloader, train_dataloader  # pls change batch_size
from torch.utils.data import DataLoader
from Code.model_lung_infection.InfNet_UNet import *
from metric import dice_similarity_coefficient, jaccard_similarity_coefficient, sensitivity_similarity_coefficient, \
    specificity_similarity_coefficient

best_loss = 1e9


def train(epo_num, num_classes, input_channels, batch_size, lr, is_data_augment, is_label_smooth, random_cutout,
          graph_path, save_path,
          device, load_net_path):
    global best_loss
    os.makedirs(f'./Snapshots/save_weights/{save_path}/', exist_ok=True)

    train_dataset = LungDataset(
        imgs_path='./Dataset/TrainingSet/MultiClassInfection-Train/Imgs/',
        # NOTES: prior is borrowed from the object-level label of train split
        pseudo_path='./Dataset/TrainingSet/MultiClassInfection-Train/Prior/',
        label_path='./Dataset/TrainingSet/MultiClassInfection-Train/GT/',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), is_data_augment=is_data_augment, is_label_smooth=is_label_smooth, random_cutout=random_cutout)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # test dataset
    test_dataset = LungDataset(
        imgs_path='./Dataset/ValSet/MultiClassInfection-Val/Imgs/',
        pseudo_path='./Dataset/ValSet/MultiClassInfection-Val/Prior/',  # NOTES: generated from Semi-Inf-Net
        label_path='./Dataset/ValSet/MultiClassInfection-Val/GT/',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        is_test=False
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    lung_model = Inf_Net_UNet(input_channels, num_classes)  # input_channels=3， n_class=3
    # lung_model.load_state_dict(torch.load('./Snapshots/save_weights/multi_baseline/unet_model_200.pkl', map_location=torch.device(device)))

    print(lung_model)
    lung_model = lung_model.to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(lung_model.parameters(), lr=lr, momentum=0.7)

    # load model if available
    if load_net_path:
        net_state_dict = torch.load(load_net_path, map_location=torch.device(device))
        net_state_dict = {k: v for k, v in net_state_dict.items() if k in lung_model.state_dict()}
        lung_model.load_state_dict(net_state_dict)

    # summary writers
    train_writer = SummaryWriter(os.path.join(graph_path, 'training'))
    test_writer = SummaryWriter(os.path.join(graph_path, 'testing'))

    print("#" * 20, "\nStart Training (Inf-Net)\nThis code is written for 'Inf-Net: Automatic COVID-19 Lung "
                    "Infection Segmentation from CT Scans', 2020, arXiv.\n"
                    "----\nPlease cite the paper if you use this code and dataset. "
                    "And any questions feel free to contact me "
                    "via E-mail (gepengai.ji@163.com)\n----\n", "#" * 20)

    global_iteration = 0
    for epo in range(epo_num):

        train_loss = 0
        lung_model.train()

        total_train_loss = []
        for index, (img, pseudo, img_mask, _) in enumerate(train_dataloader):
            global_iteration += 1

            img = img.to(device)
            pseudo = pseudo.to(device)
            img_mask = img_mask.to(device)

            optimizer.zero_grad()
            output = lung_model(torch.cat((img, pseudo), dim=1))  # change 2nd img to pseudo for original

            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, img_mask)

            loss.backward()
            iter_loss = loss.item()

            train_loss += iter_loss
            total_train_loss.append(train_loss)

            optimizer.step()

            if np.mod(index, 20) == 0:
                print('Epoch: {}/{}, Step: {}/{}, Train loss is {}'.format(epo, epo_num, index, len(train_dataloader), iter_loss))


        # old saving method
        # os.makedirs('./checkpoints//UNet_Multi-Class-Semi', exist_ok=True)
        # if np.mod(epo+1, 10) == 0:
        #     torch.save(lung_model.state_dict(),
        #                './Snapshots/save_weights/{}/unet_model_{}.pkl'.format(save_path, epo + 1))
        #     print('Saving checkpoints: unet_model_{}.pkl'.format(epo + 1))

        # average_train_loss = sum(total_train_loss) / len(total_train_loss)
        # train_writer.add_scalar('train/loss', average_train_loss, epo)
        #
        del img
        del img_mask
        total_test_loss = []
        total_test_dice = []
        total_test_jaccard = []
        total_test_sensitivity = []
        total_test_specificity = []

        lung_model.eval()
        for index, (img, pseudo, img_mask, name) in enumerate(test_dataloader):
            img = img.to(device)
            pseudo = pseudo.to(device)
            img_mask = img_mask.to(device)
            output = lung_model(torch.cat((img, pseudo), dim=1))  # change 2nd img to pseudo for original

            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, img_mask)
            print(f'test loss is {loss.item()}')
            total_test_loss.append(loss.item())
            dice = dice_similarity_coefficient(output, img_mask)
            jaccard = jaccard_similarity_coefficient(output, img_mask)
            sensitivity = sensitivity_similarity_coefficient(output, img_mask)
            specificity = specificity_similarity_coefficient(output, img_mask)

            total_test_dice.append(dice)
            total_test_jaccard.append(jaccard)
            total_test_sensitivity.append(sensitivity)
            total_test_specificity.append(specificity)

        average_test_loss = sum(total_test_loss) / len(total_test_loss)
        average_test_dice = sum(total_test_dice) / len(total_test_dice)
        average_test_jaccard = sum(total_test_jaccard) / len(total_test_jaccard)
        average_test_sensitivity = sum(total_test_sensitivity) / len(total_test_sensitivity)
        average_test_specificity = sum(total_test_specificity) / len(total_test_specificity)
        test_writer.add_scalar('test/loss', average_test_loss, epo)
        test_writer.add_scalar('test/dice', average_test_dice, epo)
        test_writer.add_scalar('test/jaccard', average_test_jaccard, epo)
        test_writer.add_scalar('test/sensitivity', average_test_sensitivity, epo)
        test_writer.add_scalar('test/specificity', average_test_specificity, epo)

        if average_test_loss < best_loss:
            best_loss = average_test_loss
            torch.save(lung_model.state_dict(),
                       './Snapshots/save_weights/{}/unet_model_{}.pkl'.format(save_path, epo + 1))
            print('Saving checkpoints: unet_model_{}.pkl'.format(epo + 1))

        del img
        del img_mask


def eval(device, pseudo_test_path, load_net_path, batch_size, input_channels, num_classes, gg_threshold, cons_threshold):
    # test dataset
    test_dataset = LungDataset(
        imgs_path='./Dataset/TestingSet/MultiClassInfection-Test/Imgs/',
        pseudo_path=pseudo_test_path,  # NOTES: generated from Semi-Inf-Net
        label_path='./Dataset/TestingSet/MultiClassInfection-Test/GT/',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        is_test=False
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    lung_model = Inf_Net_UNet(input_channels, num_classes)  # input_channels=3， n_class=3
    # lung_model.load_state_dict(torch.load('./Snapshots/save_weights/multi_baseline/unet_model_200.pkl', map_location=torch.device(device)))

    net_state_dict = torch.load(load_net_path, map_location=torch.device(device))
    net_state_dict = {k: v for k, v in net_state_dict.items() if k in lung_model.state_dict()}
    lung_model.load_state_dict(net_state_dict)

    criterion = nn.BCELoss().to(device)

    gg_total_test_loss = []
    gg_total_test_dice = []
    gg_total_test_jaccard = []
    gg_total_test_sensitivity = []
    gg_total_test_specificity = []

    cons_total_test_loss = []
    cons_total_test_dice = []
    cons_total_test_jaccard = []
    cons_total_test_sensitivity = []
    cons_total_test_specificity = []

    gg_roc = []
    cons_roc = []
    gg_gt = []
    cons_gt = []

    lung_model.eval()
    for index, (img, pseudo, img_mask, name) in enumerate(test_dataloader):
        img = img.to(device)
        pseudo = pseudo.to(device)
        img_mask = img_mask.to(device)
        output = lung_model(torch.cat((img, pseudo), dim=1))  # change 2nd img to pseudo for original

        output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])

        gg_output = output[0, 1]
        cons_output = output[0, 2]
        gg_img_mask = img_mask[0, 1]
        cons_img_mask = img_mask[0, 2]

        # append gt
        gg_gt += gg_img_mask.view(-1).detach().numpy().tolist()
        cons_gt += cons_img_mask.view(-1).detach().numpy().tolist()
        gg_roc += gg_output.view(-1).detach().numpy().tolist()
        cons_roc += cons_output.view(-1).detach().numpy().tolist()

        # calculate ground-glass opacities metrics
        loss = torch.mean(torch.abs(gg_output - gg_img_mask))
        dice = dice_similarity_coefficient(gg_output, gg_img_mask)
        jaccard = jaccard_similarity_coefficient(gg_output, gg_img_mask)
        sensitivity = sensitivity_similarity_coefficient(gg_output, gg_img_mask, gg_threshold)
        specificity = specificity_similarity_coefficient(gg_output, gg_img_mask, gg_threshold)
        gg_total_test_loss.append(loss.item())
        gg_total_test_dice.append(dice)
        gg_total_test_jaccard.append(jaccard)
        gg_total_test_sensitivity.append(sensitivity)
        gg_total_test_specificity.append(specificity)

        # calculate consolidation metrics
        loss = torch.mean(torch.abs(cons_output - cons_img_mask))
        dice = dice_similarity_coefficient(cons_output, cons_img_mask)
        jaccard = jaccard_similarity_coefficient(cons_output, cons_img_mask)
        sensitivity = sensitivity_similarity_coefficient(cons_output, cons_img_mask, cons_threshold)
        specificity = specificity_similarity_coefficient(cons_output, cons_img_mask, cons_threshold)
        cons_total_test_loss.append(loss.item())
        cons_total_test_dice.append(dice)
        cons_total_test_jaccard.append(jaccard)
        cons_total_test_sensitivity.append(sensitivity)
        cons_total_test_specificity.append(specificity)

    with open('metric.txt', 'a') as f:
        f.write(load_net_path + '\n')
        f.write('ground glass opacities\n')
        for gg in gg_total_test_dice:
            f.write(str(gg) + '\n')
        f.write('\nconsolidation\n')
        for cons in cons_total_test_dice:
            f.write(str(cons) + '\n')
        f.write('===========================')

    # calculate ground-glass metrics
    gg_fpr, gg_tpr, gg_thresholds = roc_curve(gg_gt, gg_roc)
    roc_auc = auc(gg_fpr, gg_tpr)
    print(f'ground-glass opacity auc: {roc_auc}')
    # get threshold cutoff
    optimal_idx = np.argmax(gg_tpr - gg_fpr)
    optimal_threshold = gg_thresholds[optimal_idx]
    print(f'ground-glass opacity optimal threshold: {optimal_threshold} , '
          f'tpr: {gg_tpr[optimal_idx]}, fpr: {gg_fpr[optimal_idx]}')

    gg_np_total_test_loss = np.array(gg_total_test_loss)
    gg_mean_test_loss = np.mean(gg_np_total_test_loss)
    gg_error_test_loss = np.std(gg_np_total_test_loss) / np.sqrt(gg_np_total_test_loss.size) * 1.96

    gg_np_total_test_dice = np.array(gg_total_test_dice)
    gg_mean_test_dice = np.mean(gg_np_total_test_dice)
    gg_error_test_dice = np.std(gg_np_total_test_dice) / np.sqrt(gg_np_total_test_dice.size) * 1.96

    gg_np_total_test_jaccard = np.array(gg_total_test_jaccard)
    gg_mean_test_jaccard = np.mean(gg_np_total_test_jaccard)
    gg_error_test_jaccard = np.std(gg_np_total_test_jaccard) / np.sqrt(gg_np_total_test_jaccard.size) * 1.96

    gg_np_total_test_sensitivity = np.array(gg_total_test_sensitivity)
    gg_mean_test_sensitivity = np.mean(gg_np_total_test_sensitivity)
    gg_error_test_sensitivity = np.std(gg_np_total_test_sensitivity) / np.sqrt(gg_np_total_test_sensitivity.size) * 1.96

    gg_np_total_test_specificity = np.array(gg_total_test_specificity)
    gg_mean_test_specificity = np.mean(gg_np_total_test_specificity)
    gg_error_test_specificity = np.std(gg_np_total_test_specificity) / np.sqrt(gg_np_total_test_specificity.size) * 1.96

    # calculate consolidation metrics
    cons_fpr, cons_tpr, cons_thresholds = roc_curve(cons_gt, cons_roc)
    roc_auc = auc(cons_fpr, cons_tpr)
    print(f'consolidation auc: {roc_auc}')
    # get threshold cutoff
    optimal_idx = np.argmax(cons_tpr - cons_fpr)
    optimal_threshold = cons_thresholds[optimal_idx]
    print(f'consolidation optimal threshold: {optimal_threshold} , tpr: {cons_tpr[optimal_idx]}, '
          f'fpr: {cons_fpr[optimal_idx]}')

    cons_np_total_test_loss = np.array(cons_total_test_loss)
    cons_mean_test_loss = np.mean(cons_np_total_test_loss)
    cons_error_test_loss = np.std(cons_np_total_test_loss) / np.sqrt(cons_np_total_test_loss.size) * 1.96

    cons_np_total_test_dice = np.array(cons_total_test_dice)
    cons_mean_test_dice = np.mean(cons_np_total_test_dice)
    cons_error_test_dice = np.std(cons_np_total_test_dice) / np.sqrt(cons_np_total_test_dice.size) * 1.96

    cons_np_total_test_jaccard = np.array(cons_total_test_jaccard)
    cons_mean_test_jaccard = np.mean(cons_np_total_test_jaccard)
    cons_error_test_jaccard = np.std(cons_np_total_test_jaccard) / np.sqrt(cons_np_total_test_jaccard.size) * 1.96

    cons_np_total_test_sensitivity = np.array(cons_total_test_sensitivity)
    cons_mean_test_sensitivity = np.mean(cons_np_total_test_sensitivity)
    cons_error_test_sensitivity = np.std(cons_np_total_test_sensitivity) / np.sqrt(cons_np_total_test_sensitivity.size) * 1.96

    cons_np_total_test_specificity = np.array(cons_total_test_specificity)
    cons_mean_test_specificity = np.mean(cons_np_total_test_specificity)
    cons_error_test_specificity = np.std(cons_np_total_test_specificity) / np.sqrt(cons_np_total_test_specificity.size) * 1.96

    print(f'ground-glass mean absolute dice: {gg_mean_test_dice}')
    print(f'ground-glass error absolute dice: {gg_error_test_dice}')
    print('==============================')
    print(f'ground-glass mean absolute jaccard: {gg_mean_test_jaccard}')
    print(f'ground-glass error absolute jaccard: {gg_error_test_jaccard}')
    print('==============================')
    print(f'ground-glass mean absolute sensitivity: {gg_mean_test_sensitivity}')
    print(f'ground-glass error absolute sensitivity: {gg_error_test_sensitivity}')
    print('==============================')
    print(f'ground-glass mean absolute specificity: {gg_mean_test_specificity}')
    print(f'ground-glass error absolute specificity: {gg_error_test_specificity}')
    print('==============================')
    print(f'ground-glass mean absolute error: {gg_mean_test_loss}')
    print(f'ground-glass error absolute error: {gg_error_test_loss}')
    print('==============================')
    print('==============================')
    print('==============================')
    print('==============================')
    print(f'consolidation mean absolute dice: {cons_mean_test_dice}')
    print(f'consolidation error absolute dice: {cons_error_test_dice}')
    print('==============================')
    print(f'consolidation mean absolute jaccard: {cons_mean_test_jaccard}')
    print(f'consolidation error absolute jaccard: {cons_error_test_jaccard}')
    print('==============================')
    print(f'consolidation mean absolute sensitivity: {cons_mean_test_sensitivity}')
    print(f'consolidation error absolute sensitivity: {cons_error_test_sensitivity}')
    print('==============================')
    print(f'consolidation mean absolute specificity: {cons_mean_test_specificity}')
    print(f'consolidation error absolute specificity: {cons_error_test_specificity}')
    print('==============================')
    print(f'consolidation mean absolute error: {cons_mean_test_loss}')
    print(f'consolidation error absolute error: {cons_error_test_loss}')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--graph_path', type=str, default='multi_graph_baseline')
    parser.add_argument('--save_path', type=str, default='Semi-Inf-Net_UNet')
    parser.add_argument('--pseudo_test_path', type=str)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--is_data_augment', type=bool, default=False)
    parser.add_argument('--is_label_smooth', type=bool, default=False)
    parser.add_argument('--random_cutout', type=float, default=0)
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--is_eval', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load_net_path', type=str)
    parser.add_argument('--gg_threshold', type=float)
    parser.add_argument('--cons_threshold', type=float)

    arg = parser.parse_args()

    if arg.is_eval:
        eval(arg.device, arg.pseudo_test_path, arg.load_net_path, batch_size=1, input_channels=6, num_classes=3,
             gg_threshold=arg.gg_threshold, cons_threshold=arg.cons_threshold)
    else:
        train(epo_num=arg.epoch,
              num_classes=3,
              input_channels=6,
              batch_size=arg.batchsize,
              lr=1e-2,
              is_data_augment=arg.is_data_augment,
              is_label_smooth=arg.is_label_smooth,
              random_cutout=arg.random_cutout,
              graph_path=arg.graph_path,
              save_path=arg.save_path,
              device=arg.device,
              load_net_path=arg.load_net_path)
