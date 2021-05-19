# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import os
import torch
import math
import random
import time
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold

import sys

sys.path.append('..')

from Code.utils.dataloader_MulClsLungInf_UNet import LungDataset, IndicesLungDataset
from torchvision import transforms
# from LungData import test_dataloader, train_dataloader  # pls change batch_size
from torch.utils.data import DataLoader
from Code.model_lung_infection.InfNet_UNet import *
from Code.utils.utils import timer
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from metric import dice_similarity_coefficient, jaccard_similarity_coefficient, sensitivity_similarity_coefficient, \
    precision_similarity_coefficient
from focal_loss import FocalLoss
from lookahead import Lookahead


def train(lung_model, train_dataset, test_dataset, epo_num, num_classes, input_channels, batch_size, lr, is_data_augment,
          is_label_smooth, random_cutout,
          graph_path, save_path,
          device, load_net_path, model_name, arg):
    best_loss = 1e9
    best_dice = 0
    best_jaccard = 0
    best_sensitivity = 0
    best_precision = 0

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if arg.focal_loss:
        criterion = FocalLoss().to(device)  # nn.BCELoss().to(device)
    else:
        criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(lung_model.parameters(), lr=lr, momentum=0.7)

    if arg.lookahead:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)
    optimizer.zero_grad()

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
                print('Epoch: {}/{}, Step: {}/{}, Train loss is {}'.format(epo, epo_num, index, len(train_dataloader),
                                                                           iter_loss))

        # old saving method
        # os.makedirs('./checkpoints//UNet_Multi-Class-Semi', exist_ok=True)
        # if np.mod(epo+1, 10) == 0:
        #     torch.save(lung_model.state_dict(),
        #                './Snapshots/save_weights/{}/unet_model_{}.pkl'.format(save_path, epo + 1))
        #     print('Saving checkpoints: unet_model_{}.pkl'.format(epo + 1))

        # average_train_loss = sum(total_train_loss) / len(total_train_loss)
        # train_writer.add_scalar('train/loss', average_train_loss, epo)
        #
        # del img
        # del img_mask
        total_test_loss = []

        background_test_dice = []
        background_test_jaccard = []
        background_test_sensitivity = []
        background_test_precision = []

        gg_test_dice = []
        gg_test_jaccard = []
        gg_test_sensitivity = []
        gg_test_precision = []

        cons_test_dice = []
        cons_test_jaccard = []
        cons_test_sensitivity = []
        cons_test_precision = []

        lung_model.eval()
        for index, (img, pseudo, img_mask, name) in enumerate(test_dataloader):
            img = img.to(device)
            pseudo = pseudo.to(device)
            img_mask = img_mask.to(device)
            output = lung_model(torch.cat((img, pseudo), dim=1))  # change 2nd img to pseudo for original

            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            b, _, w, h = output.size()
            pred = output.cpu().permute(0, 2, 3, 1).contiguous().view(-1, num_classes).max(1)[1].view(b, w,
                                                                                                      h).numpy().squeeze()
            pred_onehot = (np.arange(3) == pred[..., None]).astype(np.float64)

            background_output = torch.from_numpy(pred_onehot[:, :, :, 0]).to(device)
            gg_output = torch.from_numpy(pred_onehot[:, :, :, 1]).to(device)
            cons_output = torch.from_numpy(pred_onehot[:, :, :, 2]).to(device)

            background_img_mask = img_mask[:, 0]
            gg_img_mask = img_mask[:, 1]
            cons_img_mask = img_mask[:, 2]

            loss = criterion(output, img_mask)
            total_test_loss.append(loss.item())
            print(f'test loss is {loss.item()}')
            total_test_loss.append(loss.item())

            # calculate background metrics
            dice = dice_similarity_coefficient(background_output, background_img_mask, None)
            jaccard = jaccard_similarity_coefficient(background_output, background_img_mask, None)
            sensitivity = sensitivity_similarity_coefficient(background_output, background_img_mask, None)
            precision = precision_similarity_coefficient(background_output, background_img_mask, None)

            if not math.isnan(dice):
                background_test_dice.append(dice)
            if not math.isnan(jaccard):
                background_test_jaccard.append(jaccard)
            if not math.isnan(sensitivity):
                background_test_sensitivity.append(sensitivity)
            if not math.isnan(precision):
                background_test_precision.append(precision)

            # calculate ground-glass opacities metrics
            dice = dice_similarity_coefficient(gg_output, gg_img_mask, None)
            jaccard = jaccard_similarity_coefficient(gg_output, gg_img_mask, None)
            sensitivity = sensitivity_similarity_coefficient(gg_output, gg_img_mask, None)
            precision = precision_similarity_coefficient(gg_output, gg_img_mask, None)

            if not math.isnan(dice):
                gg_test_dice.append(dice)
            if not math.isnan(jaccard):
                gg_test_jaccard.append(jaccard)
            if not math.isnan(sensitivity):
                gg_test_sensitivity.append(sensitivity)
            if not math.isnan(precision):
                gg_test_precision.append(precision)

            # calculate consolidation metrics
            loss = torch.mean(torch.abs(cons_output - cons_img_mask))
            dice = dice_similarity_coefficient(cons_output, cons_img_mask, None)
            jaccard = jaccard_similarity_coefficient(cons_output, cons_img_mask, None)
            sensitivity = sensitivity_similarity_coefficient(cons_output, cons_img_mask, None)
            precision = precision_similarity_coefficient(cons_output, cons_img_mask, None)
            if not math.isnan(dice):
                cons_test_dice.append(dice)
            if not math.isnan(jaccard):
                cons_test_jaccard.append(jaccard)
            if not math.isnan(sensitivity):
                cons_test_sensitivity.append(sensitivity)
            if not math.isnan(precision):
                cons_test_precision.append(precision)

        average_test_loss = sum(total_test_loss) / len(total_test_loss)
        average_test_dice = (sum(background_test_dice) + sum(gg_test_dice) + sum(cons_test_dice)) / \
                            (len(background_test_dice) + len(gg_test_dice) + len(cons_test_dice))
        average_test_jaccard = (sum(background_test_jaccard) + sum(gg_test_jaccard) + sum(cons_test_jaccard)) / \
                               (len(background_test_jaccard) + len(gg_test_jaccard) + len(cons_test_jaccard))
        average_test_sensitivity = (sum(background_test_sensitivity) + sum(gg_test_sensitivity) + sum(
            cons_test_sensitivity)) \
                                   / (len(background_test_sensitivity) + len(gg_test_sensitivity) + len(
            cons_test_sensitivity))
        average_test_precision = (sum(background_test_precision) + sum(gg_test_precision) + sum(cons_test_precision)) / \
                                 (len(background_test_precision) + len(gg_test_precision) + len(cons_test_precision))
        test_writer.add_scalar('test/loss', average_test_loss, epo)
        test_writer.add_scalar('test/dice', average_test_dice, epo)
        test_writer.add_scalar('test/jaccard', average_test_jaccard, epo)
        test_writer.add_scalar('test/sensitivity', average_test_sensitivity, epo)
        test_writer.add_scalar('test/precision', average_test_precision, epo)

        if average_test_loss < best_loss:
            best_loss = average_test_loss
            best_dice = average_test_dice
            best_jaccard = average_test_jaccard
            best_sensitivity = average_test_sensitivity
            best_precision = average_test_precision
            torch.save(lung_model.state_dict(),
                       './Snapshots/save_weights/{}/unet_model_{}.pkl'.format(save_path, epo + 1))
            print('Saving checkpoints: unet_model_{}.pkl'.format(epo + 1))

        del img
        del img_mask
    return best_loss, best_dice, best_jaccard, best_sensitivity, best_precision


def calculate_metrics(test_dataloader, num_classes, load_net_path, lung_model, device, gg_threshold=0,
                      cons_threshold=0):
    criterion = nn.BCELoss().to(device)

    background_total_test_loss = []
    background_total_test_dice = []
    background_total_test_jaccard = []
    background_total_test_sensitivity = []
    background_total_test_precision = []

    gg_total_test_loss = []
    gg_total_test_dice = []
    gg_total_test_jaccard = []
    gg_total_test_sensitivity = []
    gg_total_test_precision = []

    cons_total_test_loss = []
    cons_total_test_dice = []
    cons_total_test_jaccard = []
    cons_total_test_sensitivity = []
    cons_total_test_precision = []

    for index, (img, pseudo, img_mask, name) in enumerate(test_dataloader):
        img = img.to(device)
        pseudo = pseudo.to(device)
        img_mask = img_mask.to(device)
        output = lung_model(torch.cat((img, pseudo), dim=1))  # change 2nd img to pseudo for original

        output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
        b, _, w, h = output.size()
        pred = output.cpu().permute(0, 2, 3, 1).contiguous().view(-1, num_classes).max(1)[1].view(b, w,
                                                                                                  h).numpy().squeeze()
        pred_onehot = (np.arange(3) == pred[..., None]).astype(np.float64)

        background_output = torch.from_numpy(pred_onehot[:, :, 0]).to(device)
        gg_output = torch.from_numpy(pred_onehot[:, :, 1]).to(device)
        cons_output = torch.from_numpy(pred_onehot[:, :, 2]).to(device)

        background_img_mask = img_mask[0, 0]
        gg_img_mask = img_mask[0, 1]
        cons_img_mask = img_mask[0, 2]

        # calculate background metrics
        loss = torch.mean(torch.abs(background_output - background_img_mask))
        dice = dice_similarity_coefficient(background_output, background_img_mask, gg_threshold)
        jaccard = jaccard_similarity_coefficient(background_output, background_img_mask, gg_threshold)
        sensitivity = sensitivity_similarity_coefficient(background_output, background_img_mask, gg_threshold)
        precision = precision_similarity_coefficient(background_output, background_img_mask, gg_threshold)
        background_total_test_loss.append(loss.item())

        if not math.isnan(dice):
            background_total_test_dice.append(dice)
        if not math.isnan(jaccard):
            background_total_test_jaccard.append(jaccard)
        if not math.isnan(sensitivity):
            background_total_test_sensitivity.append(sensitivity)
        if not math.isnan(precision):
            background_total_test_precision.append(precision)

        # calculate ground-glass opacities metrics
        loss = torch.mean(torch.abs(gg_output - gg_img_mask))
        dice = dice_similarity_coefficient(gg_output, gg_img_mask, gg_threshold)
        jaccard = jaccard_similarity_coefficient(gg_output, gg_img_mask, gg_threshold)
        sensitivity = sensitivity_similarity_coefficient(gg_output, gg_img_mask, gg_threshold)
        precision = precision_similarity_coefficient(gg_output, gg_img_mask, gg_threshold)
        gg_total_test_loss.append(loss.item())

        if not math.isnan(dice):
            gg_total_test_dice.append(dice)
        if not math.isnan(jaccard):
            gg_total_test_jaccard.append(jaccard)
        if not math.isnan(sensitivity):
            gg_total_test_sensitivity.append(sensitivity)
        if not math.isnan(precision):
            gg_total_test_precision.append(precision)

        # calculate consolidation metrics
        loss = torch.mean(torch.abs(cons_output - cons_img_mask))
        dice = dice_similarity_coefficient(cons_output, cons_img_mask, cons_threshold)
        jaccard = jaccard_similarity_coefficient(cons_output, cons_img_mask, cons_threshold)
        sensitivity = sensitivity_similarity_coefficient(cons_output, cons_img_mask, cons_threshold)
        precision = precision_similarity_coefficient(cons_output, cons_img_mask, cons_threshold)
        cons_total_test_loss.append(loss.item())
        if not math.isnan(dice):
            cons_total_test_dice.append(dice)
        if not math.isnan(jaccard):
            cons_total_test_jaccard.append(jaccard)
        if not math.isnan(sensitivity):
            cons_total_test_sensitivity.append(sensitivity)
        if not math.isnan(precision):
            cons_total_test_precision.append(precision)

    with open('metric.txt', 'a') as f:
        f.write(load_net_path + '\n')
        f.write('ground glass opacities\n')
        for gg in gg_total_test_dice:
            f.write(str(gg) + '\n')
        f.write('\nconsolidation\n')
        for cons in cons_total_test_dice:
            f.write(str(cons) + '\n')
        f.write('===========================')

    # background
    background_np_total_test_dice = np.array(background_total_test_dice)
    background_np_total_test_jaccard = np.array(background_total_test_jaccard)
    background_np_total_test_sensitivity = np.array(background_total_test_sensitivity)
    background_np_total_test_precision = np.array(background_total_test_precision)

    # ground-glass opacities
    gg_np_total_test_dice = np.array(gg_total_test_dice)
    gg_np_total_test_jaccard = np.array(gg_total_test_jaccard)
    gg_np_total_test_sensitivity = np.array(gg_total_test_sensitivity)
    gg_np_total_test_precision = np.array(gg_total_test_precision)

    # consolidation
    cons_np_total_test_dice = np.array(cons_total_test_dice)
    cons_np_total_test_jaccard = np.array(cons_total_test_jaccard)
    cons_np_total_test_sensitivity = np.array(cons_total_test_sensitivity)
    cons_np_total_test_precision = np.array(cons_total_test_precision)

    # background
    background_mean_test_dice = np.mean(background_np_total_test_dice)
    background_error_test_dice = np.std(background_np_total_test_dice) / np.sqrt(
        background_np_total_test_dice.size) * 1.96
    background_variance_dice = np.var(background_np_total_test_dice, ddof=1)

    background_mean_test_jaccard = np.mean(background_np_total_test_jaccard)
    background_error_test_jaccard = np.std(background_np_total_test_jaccard) / np.sqrt(
        background_np_total_test_jaccard.size) * 1.96
    background_variance_jaccard = np.var(background_np_total_test_jaccard, ddof=1)

    background_mean_test_sensitivity = np.mean(background_np_total_test_sensitivity)
    background_error_test_sensitivity = np.std(background_np_total_test_sensitivity) / np.sqrt(
        background_np_total_test_sensitivity.size) * 1.96
    background_variance_sensitivity = np.var(background_np_total_test_sensitivity, ddof=1)

    background_mean_test_precision = np.mean(background_np_total_test_precision)
    background_error_test_precision = np.std(background_np_total_test_precision) / np.sqrt(
        background_np_total_test_precision.size) * 1.96
    background_variance_precision = np.var(background_np_total_test_precision, ddof=1)

    # ground-glass opacities
    gg_mean_test_dice = np.mean(gg_np_total_test_dice)
    gg_error_test_dice = np.std(gg_np_total_test_dice) / np.sqrt(gg_np_total_test_dice.size) * 1.96
    gg_variance_dice = np.var(gg_np_total_test_dice, ddof=1)

    gg_mean_test_jaccard = np.mean(gg_np_total_test_jaccard)
    gg_error_test_jaccard = np.std(gg_np_total_test_jaccard) / np.sqrt(gg_np_total_test_jaccard.size) * 1.96
    gg_variance_jaccard = np.var(gg_np_total_test_jaccard, ddof=1)

    gg_mean_test_sensitivity = np.mean(gg_np_total_test_sensitivity)
    gg_error_test_sensitivity = np.std(gg_np_total_test_sensitivity) / np.sqrt(
        gg_np_total_test_sensitivity.size) * 1.96
    gg_variance_sensitivity = np.var(gg_np_total_test_sensitivity, ddof=1)

    gg_mean_test_precision = np.mean(gg_np_total_test_precision)
    gg_error_test_precision = np.std(gg_np_total_test_precision) / np.sqrt(gg_np_total_test_precision.size) * 1.96
    gg_variance_precision = np.var(gg_np_total_test_precision, ddof=1)

    # consolidation
    cons_mean_test_dice = np.mean(cons_np_total_test_dice)
    cons_error_test_dice = np.std(cons_np_total_test_dice) / np.sqrt(cons_np_total_test_dice.size) * 1.96
    cons_variance_dice = np.var(cons_np_total_test_dice, ddof=1)

    cons_mean_test_jaccard = np.mean(cons_np_total_test_jaccard)
    cons_error_test_jaccard = np.std(cons_np_total_test_jaccard) / np.sqrt(cons_np_total_test_jaccard.size) * 1.96
    cons_variance_jaccard = np.var(cons_np_total_test_jaccard, ddof=1)

    cons_mean_test_sensitivity = np.mean(cons_np_total_test_sensitivity)
    cons_error_test_sensitivity = np.std(cons_np_total_test_sensitivity) / np.sqrt(
        cons_np_total_test_sensitivity.size) * 1.96
    cons_variance_sensitivity = np.var(cons_np_total_test_sensitivity, ddof=1)

    cons_mean_test_precision = np.mean(cons_np_total_test_precision)
    cons_error_test_precision = np.std(cons_np_total_test_precision) / np.sqrt(
        cons_np_total_test_precision.size) * 1.96
    cons_variance_precision = np.var(cons_np_total_test_precision, ddof=1)

    metrics_string = ""
    metrics_string += 'background\n'
    metrics_string += '==============================\n'
    metrics_string += f'{round(background_mean_test_dice, 2)} & {round(background_mean_test_jaccard, 2)} & {round(background_mean_test_sensitivity, 2)} & {round(background_mean_test_precision, 2)}\n'
    metrics_string += '\n'
    metrics_string += f'background dice variance[{background_np_total_test_dice.size}]: {background_variance_dice}\n'
    metrics_string += f'background jaccard [{background_np_total_test_jaccard.size}]: {background_variance_jaccard}\n'
    metrics_string += f'background recall variance[{background_np_total_test_sensitivity.size}]: {background_variance_sensitivity}\n'
    metrics_string += f'background precision variance[{background_np_total_test_precision.size}]: {background_variance_precision}\n'
    metrics_string += '============error=============\n'
    metrics_string += f'$\pm${round(background_error_test_dice, 3)} & $\pm${round(background_error_test_jaccard, 3)} $\pm${round(background_error_test_sensitivity, 3)} & $\pm${round(background_error_test_precision, 3)}'
    metrics_string += '==============================\n'
    metrics_string += '==============================\n'
    metrics_string += 'ground glass opacities\n'
    metrics_string += '==============================\n'
    metrics_string += f'{round(gg_mean_test_dice, 2)} & {round(gg_mean_test_jaccard, 2)} & {round(gg_mean_test_sensitivity, 2)} & {round(gg_mean_test_precision, 2)}\n'
    metrics_string += '\n'
    metrics_string += f'gg dice variance[{gg_np_total_test_dice.size}]: {gg_variance_dice}\n'
    metrics_string += f'gg jaccard [{gg_np_total_test_jaccard.size}]: {gg_variance_jaccard}\n'
    metrics_string += f'gg recall variance[{gg_np_total_test_sensitivity.size}]: {gg_variance_sensitivity}\n'
    metrics_string += f'gg precision variance[{gg_np_total_test_precision.size}]: {gg_variance_precision}\n'
    metrics_string += '============error=============\n'
    metrics_string += f'$\pm${round(gg_error_test_dice, 3)} & $\pm${round(gg_error_test_jaccard, 3)} & $\pm${round(gg_error_test_sensitivity, 3)} & $\pm${round(gg_error_test_precision, 3)}\n'
    metrics_string += '==============================\n'
    metrics_string += '==============================\n'
    metrics_string += 'consolidation\n'
    metrics_string += '==============================\n'
    metrics_string += f'{round(cons_mean_test_dice, 2)} & {round(cons_mean_test_jaccard, 2)} & {round(cons_mean_test_sensitivity, 2)} & {round(cons_mean_test_precision, 2)}\n'
    metrics_string += '\n'
    metrics_string += f'cons dice variance[{cons_np_total_test_dice.size}]: {cons_variance_dice}\n'
    metrics_string += f'cons jaccard [{cons_np_total_test_jaccard.size}]: {cons_variance_jaccard}\n'
    metrics_string += f'cons recall variance[{cons_np_total_test_sensitivity.size}]: {cons_variance_sensitivity}\n'
    metrics_string += f'cons precision variance[{cons_np_total_test_precision.size}]: {cons_variance_precision}\n'
    metrics_string += '============error=============\n'
    metrics_string += f'$\pm${round(cons_error_test_dice, 3)} & $\pm${round(cons_error_test_jaccard, 3)} & $\pm${round(cons_error_test_sensitivity, 3)} & $\pm${round(cons_error_test_precision, 3)}\n'
    metrics_string += '==============================\n'
    metrics_string += '==============================\n'

    overall_dice = (background_mean_test_dice + gg_mean_test_dice + cons_mean_test_dice) / 3
    overall_jaccard = (background_mean_test_jaccard + gg_mean_test_jaccard + cons_mean_test_jaccard) / 3
    overall_sensitivity = (background_mean_test_sensitivity + gg_mean_test_sensitivity + cons_mean_test_sensitivity) / 3
    overall_precision = (background_mean_test_precision + gg_mean_test_precision + cons_mean_test_precision) / 3

    overall_error_dice = (background_error_test_dice + gg_error_test_dice + cons_error_test_dice) / 3
    overall_error_jaccard = (background_error_test_jaccard + gg_error_test_jaccard + cons_error_test_jaccard) / 3
    overall_error_sensitivity = (
                                        background_error_test_sensitivity + gg_error_test_sensitivity + cons_error_test_sensitivity) / 3
    overall_error_precision = (
                                      background_error_test_precision + gg_error_test_precision + cons_error_test_precision) / 3

    overall_variance_dice = (background_variance_dice + cons_variance_dice + gg_variance_dice) / 3
    overall_variance_jaccard = (background_variance_jaccard + cons_variance_jaccard + gg_variance_jaccard) / 3
    overall_variance_sensitivity = (
                                           background_variance_sensitivity + cons_variance_sensitivity + gg_variance_sensitivity) / 3
    overall_variance_precision = (background_variance_precision + cons_variance_precision + gg_variance_precision) / 3

    metrics_string += 'overall'
    metrics_string += '=============================='
    metrics_string += f'{round(overall_dice, 2)} & {round(overall_jaccard, 2)} & {round(overall_sensitivity, 2)} & {round(overall_precision, 2)}\n'
    metrics_string += '\n'
    metrics_string += f'overall dice variance[{overall_dice.size}]: {overall_variance_dice}'
    metrics_string += f'overall jaccard [{overall_jaccard.size}]: {overall_variance_jaccard}'
    metrics_string += f'overall recall variance[{overall_sensitivity.size}]: {overall_variance_sensitivity}'
    metrics_string += f'overall precision variance[{overall_precision.size}]: {overall_variance_precision}'
    metrics_string += '============error============='
    metrics_string += f'$\pm${round(overall_error_dice, 3)} & $\pm${round(overall_error_jaccard, 3)} & $\pm${round(overall_error_sensitivity, 3)} & $\pm${round(overall_error_precision, 3)}\n'
    metrics_string += '=============================='
    metrics_string += '=============================='

    print(metrics_string)

    all_metrics_information = {'mean_background_dice': background_mean_test_dice,
                               'error_background_dice': background_error_test_dice,
                               'mean_gg_dice': gg_mean_test_dice,
                               'error_gg_dice': gg_error_test_dice,
                               'mean_cons_dice': cons_mean_test_dice,
                               'error_cons_dice': cons_error_test_dice,

                               'mean_background_jaccard': background_mean_test_jaccard,
                               'error_background_jaccard': background_error_test_jaccard,
                               'mean_gg_jaccard': gg_mean_test_jaccard,
                               'error_gg_jaccard': gg_error_test_jaccard,
                               'mean_cons_jaccard': cons_mean_test_jaccard,
                               'error_cons_jaccard': cons_error_test_jaccard,

                               'mean_background_sensitivity': background_mean_test_sensitivity,
                               'error_background_sensitivity': background_error_test_sensitivity,
                               'mean_gg_sensitivity': gg_mean_test_sensitivity,
                               'error_gg_sensitivity': gg_error_test_sensitivity,
                               'mean_cons_sensitivity': cons_mean_test_sensitivity,
                               'error_cons_sensitivity': cons_error_test_sensitivity,

                               'mean_background_precision': background_mean_test_precision,
                               'error_background_precision': background_error_test_precision,
                               'mean_gg_precision': gg_mean_test_precision,
                               'error_gg_precision': gg_error_test_precision,
                               'mean_cons_precision': cons_mean_test_precision,
                               'error_cons_precision': cons_error_test_precision,

                               'overall_dice': overall_dice,
                               'error_overall_dice': overall_error_dice,
                               'overall_jaccard': overall_jaccard,
                               'error_overall_jaccard': overall_error_jaccard,
                               'overall_sensitivity': overall_sensitivity,
                               'error_overall_sensitivity': overall_error_sensitivity,
                               'overall_precision': overall_precision,
                               'error_overall_precision': overall_error_precision,

                               'metrics_string': metrics_string
                               }

    return all_metrics_information, background_np_total_test_dice, background_np_total_test_jaccard, background_np_total_test_sensitivity, background_np_total_test_precision, \
           gg_np_total_test_dice, gg_np_total_test_jaccard, gg_np_total_test_sensitivity, gg_np_total_test_precision, \
           cons_np_total_test_dice, cons_np_total_test_jaccard, cons_np_total_test_sensitivity, cons_np_total_test_precision


# load_net_path is the first model that we evaluate
# load_net_path_2 is a second model that we evaluate (can be None/empty)
# if load_net_path_2 is provided, then the wilcox test will be calculated to compare between load_net_path and
# load_net_path_2 to determine if they are statistically significant
def eval(test_dataset, device, pseudo_test_path, batch_size, input_channels, num_classes, gg_threshold, cons_threshold, load_net_path,
         load_net_path_2, model_name, model_name_2):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model_dict = {'baseline': Inf_Net_UNet, 'improved': Inf_Net_UNet_Improved}
    lung_model = model_dict[model_name](input_channels, num_classes).to(device)  # input_channels=3， n_class=3
    # lung_model.load_state_dict(torch.load('./Snapshots/save_weights/multi_baseline/unet_model_200.pkl', map_location=torch.device(device)))

    net_state_dict = torch.load(load_net_path, map_location=torch.device(device))
    net_state_dict = {k: v for k, v in net_state_dict.items() if k in lung_model.state_dict()}
    lung_model.load_state_dict(net_state_dict)
    lung_model.eval()

    all_metrics_information, background_np_total_test_dice, background_np_total_test_jaccard, background_np_total_test_sensitivity, background_np_total_test_precision, \
    gg_np_total_test_dice, gg_np_total_test_jaccard, gg_np_total_test_sensitivity, gg_np_total_test_precision, \
    cons_np_total_test_dice, cons_np_total_test_jaccard, cons_np_total_test_sensitivity, cons_np_total_test_precision \
        = calculate_metrics(test_dataloader, num_classes, load_net_path, lung_model, device, gg_threshold,
                            cons_threshold)

    # if there is not second network then it is done
    if load_net_path_2 is None:
        return all_metrics_information, None
    del lung_model
    lung_model = model_dict[model_name_2](input_channels, num_classes).to(device)
    net_state_dict = torch.load(load_net_path_2, map_location=torch.device(device))
    net_state_dict = {k: v for k, v in net_state_dict.items() if k in lung_model.state_dict()}
    lung_model.load_state_dict(net_state_dict)
    lung_model.eval()

    all_metrics_information_2, background_np_total_test_dice_2, background_np_total_test_jaccard_2, background_np_total_test_sensitivity_2, background_np_total_test_precision_2, \
    gg_np_total_test_dice_2, gg_np_total_test_jaccard_2, gg_np_total_test_sensitivity_2, gg_np_total_test_precision_2, \
    cons_np_total_test_dice_2, cons_np_total_test_jaccard_2, cons_np_total_test_sensitivity_2, cons_np_total_test_precision_2 \
        = calculate_metrics(test_dataloader, num_classes, load_net_path, lung_model, device, gg_threshold,
                            cons_threshold)

    background_dice_stats, background_dice_pvalue = mannwhitneyu(background_np_total_test_dice,
                                                                 background_np_total_test_dice_2)
    background_jaccard_stats, background_jaccard_pvalue = mannwhitneyu(background_np_total_test_jaccard,
                                                                       background_np_total_test_jaccard_2)
    background_sensitivity_stats, background_sensitivity_pvalue = mannwhitneyu(background_np_total_test_sensitivity,
                                                                               background_np_total_test_sensitivity_2)
    background_precision_stats, background_precision_pvalue = mannwhitneyu(background_np_total_test_precision,
                                                                           background_np_total_test_precision_2)

    gg_dice_stats, gg_dice_pvalue = mannwhitneyu(gg_np_total_test_dice,
                                                 gg_np_total_test_dice_2)
    gg_jaccard_stats, gg_jaccard_pvalue = mannwhitneyu(gg_np_total_test_jaccard,
                                                       gg_np_total_test_jaccard_2)
    gg_sensitivity_stats, gg_sensitivity_pvalue = mannwhitneyu(gg_np_total_test_sensitivity,
                                                               gg_np_total_test_sensitivity_2)
    gg_precision_stats, gg_precision_pvalue = mannwhitneyu(gg_np_total_test_precision,
                                                           background_np_total_test_precision_2)

    cons_dice_stats, cons_dice_pvalue = mannwhitneyu(cons_np_total_test_dice,
                                                     cons_np_total_test_dice_2)
    cons_jaccard_stats, cons_jaccard_pvalue = mannwhitneyu(cons_np_total_test_jaccard,
                                                           cons_np_total_test_jaccard_2)
    cons_sensitivity_stats, cons_sensitivity_pvalue = mannwhitneyu(cons_np_total_test_sensitivity,
                                                                   cons_np_total_test_sensitivity_2)
    cons_precision_stats, cons_precision_pvalue = mannwhitneyu(cons_np_total_test_precision,
                                                               background_np_total_test_precision_2)

    print('======================================================================')
    print('======================================================================')
    print('======================== calculate comparison ========================')
    print('===background===')
    print(f'background dice pvalue: {background_dice_pvalue}')
    print(f'background jaccard pvalue: {background_jaccard_pvalue}')
    print(f'background sensitivity pvalue: {background_sensitivity_pvalue}')
    print(f'background precision pvalue: {background_precision_pvalue}')
    print('===ground-glass opacities===')
    print(f'gg dice pvalue: {gg_dice_pvalue}')
    print(f'gg jaccard pvalue: {gg_jaccard_pvalue}')
    print(f'gg sensitivity pvalue: {gg_sensitivity_pvalue}')
    print(f'gg precision pvalue: {gg_precision_pvalue}')
    print('===consolidation===')
    print(f'cons dice pvalue: {cons_dice_pvalue}')
    print(f'cons jaccard pvalue: {cons_jaccard_pvalue}')
    print(f'cons sensitivity pvalue: {cons_sensitivity_pvalue}')
    print(f'cons precision pvalue: {cons_precision_pvalue}')
    return all_metrics_information, all_metrics_information_2


def cross_validation(arg):
    imgs_path = './Dataset/AllSet/MultiClassInfection-All/Imgs/'
    img_names = [imgs_path + f for f in os.listdir(imgs_path)]
    # NOTES: prior is borrowed from the object-level label of train split
    pseudo_path = './Dataset/AllSet/MultiClassInfection-All/Prior/'
    label_path = './Dataset/AllSet/MultiClassInfection-All/GT/'

    imgs_path = np.array(imgs_path)

    k_folds = KFold(5)
    for fold_index, (train_index, test_index) in enumerate(k_folds.split(imgs_path)):
        np.random.seed(arg.seed)
        random.seed(arg.seed)
        torch.manual_seed(arg.seed)
        torch.cuda.manual_seed(arg.seed)
        torch.random.manual_seed(arg.seed)

        train_img_names = img_names[train_index]
        test_img_names = img_names[test_index]

        training_dataset = IndicesLungDataset(
            img_names=train_img_names,
            pseudo_path=pseudo_path,
            label_path=label_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            is_data_augment=arg.is_data_augment, is_label_smooth=arg.is_label_smooth, random_cutout=arg.random_cutout
        )
        testing_dataset = IndicesLungDataset(
            img_names=test_img_names,
            pseudo_path=pseudo_path,
            label_path=label_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            is_test=False)

        model_dict = {'baseline': Inf_Net_UNet, 'improved': Inf_Net_UNet_Improved}
        lung_model = model_dict[arg.model_name](arg.input_channels, arg.num_classes)  # input_channels=3， n_class=3
        # lung_model.load_state_dict(torch.load('./Snapshots/save_weights/multi_baseline/unet_model_200.pkl', map_location=torch.device(device)))
        print(lung_model)
        lung_model = lung_model.to(arg.device)

        train(lung_model, training_dataset, testing_dataset, epo_num=arg.epoch,
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
              load_net_path=arg.load_net_path,
              model_name=arg.model_name,
              arg=arg)

        all_metrics_information, _ = eval(testing_dataset, arg.device, arg.pseudo_test_path, batch_size=1, input_channels=6, num_classes=3,
             gg_threshold=arg.gg_threshold, cons_threshold=arg.cons_threshold,
             load_net_path=arg.load_net_path,
             model_name=arg.model_name, load_net_path_2=None, model_name_2=None)

        # write the metrics
        os.makedirs(os.path.join(arg.metric_path, arg.save_path), exist_ok=True)
        filename = os.path.join(arg.metric_path, arg.save_path, f"metrics_{fold_index}.txt")
        with open(f'{filename}', 'a') as f:
            f.write(all_metrics_information['metrics_string'])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--graph_path', type=str, default='multi_graph_baseline')
    parser.add_argument('--metric_path', type=str, default='./metrics_log')
    parser.add_argument('--save_path', type=str, default='Semi-Inf-Net_UNet')
    parser.add_argument('--model_name', type=str, default='baseline')  # baseline or improved
    parser.add_argument('--pseudo_test_path', type=str)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--is_data_augment', type=bool, default=False)
    parser.add_argument('--is_label_smooth', type=bool, default=False)
    parser.add_argument('--random_cutout', type=float, default=0)
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--is_eval', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load_net_path', type=str)
    parser.add_argument('--load_net_path_2', type=str, default=None)
    parser.add_argument('--model_name_2', type=str, default='baseline')
    parser.add_argument('--gg_threshold', type=float)
    parser.add_argument('--cons_threshold', type=float)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--fold', default=0, type=int)

    parser.add_argument('--focal_loss', action='store_true')
    parser.add_argument('--lookahead', action='store_true')

    arg = parser.parse_args()

    if arg.is_eval:
        # evaluation
        start = time.time()
        # test dataset
        test_dataset = LungDataset(
            imgs_path='./Dataset/TestingSet/MultiClassInfection-Test/Imgs/',
            pseudo_path=arg.pseudo_test_path,  # NOTES: generated from Semi-Inf-Net
            label_path='./Dataset/TestingSet/MultiClassInfection-Test/GT/',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            is_test=False
        )
        eval(test_dataset, arg.device, arg.pseudo_test_path, batch_size=1, input_channels=6, num_classes=3,
             gg_threshold=arg.gg_threshold, cons_threshold=arg.cons_threshold,
             load_net_path=arg.load_net_path, load_net_path_2=arg.load_net_path_2,
             model_name=arg.model_name, model_name_2=arg.model_name_2)
        end = time.time()
        timer(start, end)
    else:
        # training

        # do cross validation
        if arg.fold > 0:
            cross_validation(arg)
        else:

            # just do single training if 0 folds
            np.random.seed(arg.seed)
            random.seed(arg.seed)
            torch.manual_seed(arg.seed)
            torch.cuda.manual_seed(arg.seed)
            torch.random.manual_seed(arg.seed)
            start = time.time()

            os.makedirs(f'./Snapshots/save_weights/{arg.save_path}/', exist_ok=True)

            train_dataset = LungDataset(
                imgs_path='./Dataset/TrainingSet/MultiClassInfection-Train/Imgs/',
                # NOTES: prior is borrowed from the object-level label of train split
                pseudo_path='./Dataset/TrainingSet/MultiClassInfection-Train/Prior/',
                label_path='./Dataset/TrainingSet/MultiClassInfection-Train/GT/',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                is_data_augment=arg.is_data_augment, is_label_smooth=arg.is_label_smooth,
                random_cutout=arg.random_cutout)

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

            # load model
            model_dict = {'baseline': Inf_Net_UNet, 'improved': Inf_Net_UNet_Improved}
            lung_model = model_dict[arg.model_name](arg.input_channels, arg.num_classes)  # input_channels=3， n_class=3
            # lung_model.load_state_dict(torch.load('./Snapshots/save_weights/multi_baseline/unet_model_200.pkl', map_location=torch.device(device)))
            print(lung_model)
            lung_model = lung_model.to(arg.device)
            best_loss, best_dice, best_jaccard, best_sensitivity, best_precision = train(
                lung_model,
                train_dataset,
                test_dataset,
                epo_num=arg.epoch,
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
                load_net_path=arg.load_net_path,
                model_name=arg.model_name,
                arg=arg)
            end = time.time()
            timer(start, time)
