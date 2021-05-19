# -*- coding: utf-8 -*-

"""Preview
Code for 'Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Scans'
submit to Transactions on Medical Imaging, 2020.

First Version: Created on 2020-05-13 (@author: Ge-Peng Ji)
"""

import torch
import math
import time
import random
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
import torch.utils.data
import os
import numpy as np
import argparse
from datetime import datetime
from Code.utils.dataloader_LungInf import get_loader, COVIDDataset
from Code.utils.utils import clip_gradient, adjust_lr, AvgMeter, timer
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, auc
import statistics
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold

from focal_loss import FocalLoss
from lookahead import Lookahead

import sys
sys.path.append('..')

from InfNet.Code.utils.dataloader_LungInf import test_dataset
from metric import dice_similarity_coefficient, jaccard_similarity_coefficient, sensitivity_similarity_coefficient, \
    specificity_similarity_coefficient, precision_similarity_coefficient

global_current_iteration = 0
best_loss = 1e9
focal_loss_criterion = FocalLoss(logits=True)


def joint_loss(pred, mask, opt):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    if opt.focal_loss:
        wbce = focal_loss_criterion(pred, mask)#, reduce='none')
    else:
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, test_loader, model, optimizer, epoch, train_save, device, opt):
    global global_current_iteration
    global best_loss
    global focal_loss_criterion

    if opt.lookahead:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)
    optimizer.zero_grad()
    focal_loss_criterion = focal_loss_criterion.to(device)

    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]    # replace your desired scale, try larger scale for better accuracy in small object
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        global_current_iteration += 1
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edges = pack
            images = Variable(images).to(device)
            gts = Variable(gts).to(device)
            edges = Variable(edges).to(device)
            # ---- rescaling the inputs (img/gt/edge) ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edges = F.upsample(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(images)
            # ---- loss function ----
            loss5 = joint_loss(lateral_map_5, gts, opt)
            loss4 = joint_loss(lateral_map_4, gts, opt)
            loss3 = joint_loss(lateral_map_3, gts, opt)
            loss2 = joint_loss(lateral_map_2, gts, opt)
            loss1 = BCE(lateral_edge, edges)
            loss = loss1 + loss2 + loss3 + loss4 + loss5

            train_writer.add_scalar('train/edge_loss', loss1.item(), global_current_iteration)
            train_writer.add_scalar('train/loss2', loss2.item(), global_current_iteration)
            train_writer.add_scalar('train/loss3', loss3.item(), global_current_iteration)
            train_writer.add_scalar('train/loss4', loss4.item(), global_current_iteration)
            train_writer.add_scalar('train/loss5', loss5.item(), global_current_iteration)
            scalar_total_loss = loss2.item() + loss3.item() + loss4.item() + loss5.item()
            train_writer.add_scalar('train/total_loss', scalar_total_loss, global_current_iteration)

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train logging ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-edge: {:.4f}, '
                  'lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record1.show(),
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
        # check testing error
        if global_current_iteration % 20 == 0:
            total_test_step = 0
            total_loss_5 = 0
            total_loss_4 = 0
            total_loss_3 = 0
            total_loss_2 = 0

            total_dice_5 = 0
            total_dice_4 = 0
            total_dice_3 = 0
            total_dice_2 = 0
            model.eval()
            for pack in test_loader:
                total_test_step += 1
                image, gt, _, name = pack
                image = Variable(image).to(device)
                gt = Variable(gt).to(device)
                # ---- forward ----
                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)
                # ---- loss function ----
                loss5 = joint_loss(lateral_map_5, gt, opt)
                loss4 = joint_loss(lateral_map_4, gt, opt)
                loss3 = joint_loss(lateral_map_3, gt, opt)
                loss2 = joint_loss(lateral_map_2, gt, opt)
                total_loss_5 += loss5.item()
                total_loss_4 += loss4.item()
                total_loss_3 += loss3.item()
                total_loss_2 += loss2.item()

                total_dice_5 += dice_similarity_coefficient(lateral_map_5.sigmoid(), gt, 0.5)
                total_dice_4 += dice_similarity_coefficient(lateral_map_4.sigmoid(), gt, 0.5)
                total_dice_3 += dice_similarity_coefficient(lateral_map_3.sigmoid(), gt, 0.5)
                total_dice_2 += dice_similarity_coefficient(lateral_map_2.sigmoid(), gt, 0.5)

            total_average_loss = (total_loss_2 + total_loss_3 + total_loss_4 + total_loss_5) / total_test_step / 4
            test_writer.add_scalar('test/loss2', total_loss_2/total_test_step, global_current_iteration)
            test_writer.add_scalar('test/loss3', total_loss_3/total_test_step, global_current_iteration)
            test_writer.add_scalar('test/loss4', total_loss_4/total_test_step, global_current_iteration)
            test_writer.add_scalar('test/loss5', total_loss_5/total_test_step, global_current_iteration)
            test_writer.add_scalar('test/total_loss', total_average_loss, global_current_iteration)
            test_writer.add_scalar('test/dice', (total_dice_2 + total_dice_3 + total_dice_4 + total_dice_5) / total_test_step / 4, global_current_iteration)
            model.train()

            if total_average_loss < best_loss:
                best_loss = total_average_loss
                # ---- save model_lung_infection ----
                save_path = './Snapshots/save_weights/{}/'.format(train_save)
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), save_path + 'Inf-Net-%d.pth' % (epoch + 1))
                print('[Saving Snapshot:]', save_path + 'Inf-Net-%d.pth' % (epoch + 1))


def eval(test_loader, model, device, load_net_path, threshold, opt):
    total_test_step = 0
    total_loss_5 = []
    total_loss_4 = []
    total_loss_3 = []
    total_loss_2 = []

    total_dice_5 = []
    total_dice_4 = []
    total_dice_3 = []
    total_dice_2 = []

    total_jaccard_5 = []
    total_jaccard_4 = []
    total_jaccard_3 = []
    total_jaccard_2 = []

    total_sens_5 = []
    total_sens_4 = []
    total_sens_3 = []
    total_sens_2 = []

    total_spec_5 = []
    total_spec_4 = []
    total_spec_3 = []
    total_spec_2 = []

    total_precision_2 = []

    total_auc_2 = []

    roc_2 = []
    ground_truth_list = []

    model.eval()
    for pack in test_loader:
        total_test_step += 1
        image, gt_cont, gt_roc, name = pack
        image = Variable(image).to(device)
        gt_cont = Variable(gt_cont).to(device)
        gt_roc = Variable(gt_roc).to(device)
        # ---- forward ----
        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)
        # ---- loss function ----
        # loss5 = joint_loss(lateral_map_5, gt)
        # loss4 = joint_loss(lateral_map_4, gt)
        # loss3 = joint_loss(lateral_map_3, gt)
        # loss2 = joint_loss(lateral_map_2, gt)
        # loss5 = torch.mean(torch.abs(lateral_map_5 - gt_cont))
        # loss4 = torch.mean(torch.abs(lateral_map_4 - gt_cont))
        # loss3 = torch.mean(torch.abs(lateral_map_3 - gt_cont))
        loss2 = torch.mean(torch.abs(lateral_map_2 - gt_cont))
        # total_loss_5.append(loss5.item())
        # total_loss_4.append(loss4.item())
        # total_loss_3.append(loss3.item())
        total_loss_2.append(loss2.item())

        # total_dice_5.append(dice_similarity_coefficient(lateral_map_5.sigmoid(), gt_cont))
        # total_dice_4.append(dice_similarity_coefficient(lateral_map_4.sigmoid(), gt_cont))
        # total_dice_3.append(dice_similarity_coefficient(lateral_map_3.sigmoid(), gt_cont))
        current_dice = dice_similarity_coefficient(lateral_map_2.sigmoid(), gt_roc, threshold)
        if not math.isnan(current_dice):
            total_dice_2.append(current_dice)

        # total_jaccard_5.append(jaccard_similarity_coefficient(lateral_map_5.sigmoid(), gt_cont))
        # total_jaccard_4.append(jaccard_similarity_coefficient(lateral_map_4.sigmoid(), gt_cont))
        # total_jaccard_3.append(jaccard_similarity_coefficient(lateral_map_3.sigmoid(), gt_cont))
        current_jaccard = jaccard_similarity_coefficient(lateral_map_2.sigmoid(), gt_roc, threshold)
        if not math.isnan(current_jaccard):
            total_jaccard_2.append(current_jaccard)

        roc_2 += lateral_map_2.sigmoid().detach().view(-1).cpu().numpy().tolist()
        ground_truth_list += gt_roc.detach().view(-1).cpu().numpy().tolist()

        # total_sens_5.append(sensitivity_similarity_coefficient(lateral_map_5.sigmoid(), gt_roc, threshold))
        # total_sens_4.append(sensitivity_similarity_coefficient(lateral_map_4.sigmoid(), gt_roc, threshold))
        # total_sens_3.append(sensitivity_similarity_coefficient(lateral_map_3.sigmoid(), gt_roc, threshold))
        current_sens = sensitivity_similarity_coefficient(lateral_map_2.sigmoid(), gt_roc, threshold)
        if not math.isnan(current_sens):
            total_sens_2.append(current_sens)

        # total_spec_5.append(specificity_similarity_coefficient(lateral_map_5.sigmoid(), gt_roc, threshold))
        # total_spec_4.append(specificity_similarity_coefficient(lateral_map_4.sigmoid(), gt_roc, threshold))
        # total_spec_3.append(specificity_similarity_coefficient(lateral_map_3.sigmoid(), gt_roc, threshold))
        current_precision = precision_similarity_coefficient(lateral_map_2.sigmoid(), gt_roc, threshold)
        if not math.isnan(current_precision):
            total_precision_2.append(current_precision)

        fpr, tpr, thresholds = roc_curve(gt_roc.view(-1).detach().cpu().numpy(),
                                         lateral_map_2.sigmoid().view(-1).detach().cpu().numpy())
        roc_auc_2 = auc(fpr, tpr)
        if not math.isnan(roc_auc_2):
            total_auc_2.append(roc_auc_2)

    fpr, tpr, thresholds = roc_curve(ground_truth_list, roc_2)

    roc_auc = auc(fpr, tpr)
    print(f'auc: {roc_auc}')
    # get threshold cutoff
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    print(f'optimal threshold: {optimal_threshold} , tpr: {tpr[optimal_idx]}, fpr: {fpr[optimal_idx]}')

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(optimal_fpr, optimal_tpr, 'go')
    plt.annotate(f'{optimal_threshold}', (optimal_fpr, optimal_tpr))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{load_net_path.split(os.sep)[-2]}')
    plt.legend(loc="lower right")
    plt.show()

    roc_dict = {'tpr': tpr, 'fpr': fpr, 'optimal_tpr': optimal_tpr, 'optimal_fpr': optimal_fpr,
                'optimal_threshold': optimal_threshold}
    save_roc_dict_dir = './roc_saves'
    os.makedirs(save_roc_dict_dir, exist_ok=True)
    save_roc_dict_filename = os.path.join(save_roc_dict_dir, load_net_path.split(os.sep)[-2])
    with open(save_roc_dict_filename, 'wb') as f:
        pickle.dump(roc_dict, f, pickle.HIGHEST_PROTOCOL)

    # accumulated_loss = (np.array(total_loss_2) + np.array(total_loss_3) + np.array(total_loss_4) + np.array(
    #     total_loss_5)) / 4
    accumulated_loss = np.array(total_loss_2)
    mean_loss = np.mean(accumulated_loss)
    error_loss = np.std(accumulated_loss) / np.sqrt(accumulated_loss.size) * 1.96

    # accumulated_dice = (np.array(total_dice_2) + np.array(total_dice_3) + np.array(total_dice_4) + np.array(
    #     total_dice_5)) / 4
    accumulated_dice = np.array(total_dice_2)
    mean_dice = np.mean(accumulated_dice)
    error_dice = np.std(accumulated_dice) / np.sqrt(accumulated_dice.size) * 1.96

    # accumulated_jaccard = (np.array(total_jaccard_2) + np.array(total_jaccard_3) + np.array(total_jaccard_4) + np.array(
    #     total_jaccard_5)) / 4
    accumulated_jaccard = np.array(total_jaccard_2)
    mean_jaccard = np.mean(accumulated_jaccard)
    error_jaccard = np.std(accumulated_jaccard) / np.sqrt(accumulated_jaccard.size) * 1.96

    # accumulated_sens = (np.array(total_sens_2) + np.array(total_sens_3) + np.array(total_sens_4) + np.array(
    #     total_sens_5)) / 4
    accumulated_sens = np.array(total_sens_2)
    mean_sens = np.mean(accumulated_sens)
    error_sens = np.std(accumulated_sens) / np.sqrt(accumulated_sens.size) * 1.96

    # accumulated_spec = (np.array(total_spec_2) + np.array(total_spec_3) + np.array(total_spec_4) + np.array(
    #     total_spec_5)) / 4
    accumulated_precision = np.array(total_precision_2)
    mean_precision = np.mean(accumulated_precision)
    error_precision = np.std(accumulated_precision) / np.sqrt(accumulated_precision.size) * 1.96

    accumulated_auc = np.array(total_auc_2)
    mean_auc = np.mean(accumulated_auc)
    error_auc = np.std(accumulated_auc) / np.sqrt(accumulated_auc.size) * 1.96

    with open('single_metric.txt', 'a') as f:
        f.write(load_net_path + '\n')
        for loss in accumulated_dice:
            f.write(str(loss) + '\n')

    metric_string = ""
    metric_string  += f'mean absolute loss: {mean_loss}\n'
    metric_string  += f'error absolute loss: {error_loss}\n'
    metric_string  += '=============================\n'
    metric_string  += f'mean dice: {mean_dice}\n'
    metric_string  += f'error dice: {error_dice}\n'
    metric_string  += '=============================\n'
    metric_string  += f'mean jaccard: {mean_jaccard}\n'
    metric_string  += f'error jaccard: {error_jaccard}\n'
    metric_string  += '=============================\n'
    metric_string  += f'mean sens: {mean_sens}\n'
    metric_string  += f'error sens: {error_sens}\n'
    metric_string  += '=============================\n'
    metric_string  += f'mean precision: {mean_precision}\n'
    metric_string  += f'error precision: {error_precision}\n'
    metric_string  += '=============================\n'
    metric_string  += f'mean auc: {mean_auc}\n'
    metric_string  += f'error auc: {error_auc}\n'

    return metric_string


def cross_validation(train_save, opt):
    image_root = '{}/Imgs/'.format(opt.all_path)
    gt_root = '{}/GT/'.format(opt.all_path)
    edge_root = '{}/Edge/'.format(opt.all_path)

    dataset = COVIDDataset(image_root, gt_root, edge_root, opt.trainsize, opt.is_data_augment, opt.random_cutout)

    k_folds = KFold(5)
    for fold_index, (train_index, test_index) in enumerate(k_folds.split(dataset)):
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.random.manual_seed(opt.seed)
        model, optimizer = create_model(opt)

        training_dataset = torch.utils.data.dataset.Subset(dataset, train_index)
        testing_dataset = torch.utils.data.dataset.Subset(dataset, test_index)
        train_loader = torch.utils.data.DataLoader(dataset=training_dataset,
                                                   batch_size=opt.batchsize,
                                                   shuffle=True,
                                                   num_workers=opt.num_workers,
                                                   pin_memory=True,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset=testing_dataset,
                                                   batch_size=opt.batchsize,
                                                   shuffle=True,
                                                   num_workers=opt.num_workers,
                                                   pin_memory=True,
                                                   drop_last=False)

        for epoch in range(1, opt.epoch):
            adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            train(train_loader, test_loader, model, optimizer, epoch, train_save, opt.device, opt)
            metric_string = eval(test_loader, model, opt.device, opt.load_net_path, opt.eval_threshold, opt)

            # write the metrics
            os.makedirs(os.path.join(opt.metric_path, opt.load_net_path), exist_ok=True)
            filename = os.path.join(opt.metric_path, opt.load_net_path, f"metrics_{fold_index}.txt")
            with open(f'{filename}', 'a') as f:
                f.write(metric_string)


def create_model(opt):
    model = Inf_Net(channel=opt.net_channel, n_class=opt.n_classes).to(opt.device)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    return model, optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('--folds', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100,
                        help='epoch number')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--batchsize', type=int, default=24,
                        help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='set the size of training sample')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='every n epochs decay learning rate')
    parser.add_argument('--is_thop', type=bool, default=True,
                        help='whether calculate FLOPs/Params (Thop)')
    parser.add_argument('--gpu_device', type=int, default=0,
                        help='choose which GPU device you want to use')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers in dataloader. In windows, set num_workers=0')
    parser.add_argument('--device', type=str, default='cpu')
    # model_lung_infection parameters
    parser.add_argument('--net_channel', type=int, default=32,
                        help='internal channel numbers in the Inf-Net, default=32, try larger for better accuracy')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='binary segmentation when n_classes=1')
    parser.add_argument('--backbone', type=str, default='ResNet50',
                        help='change different backbone, choice: VGGNet16, ResNet50, Res2Net50')
    # training dataset
    parser.add_argument('--train_path', type=str,
                        default='./Dataset/TrainingSet/LungInfection-Train/Doctor-label')
    parser.add_argument('--is_semi', type=bool, default=False,
                        help='if True, you will turn on the mode of `Semi-Inf-Net`')
    parser.add_argument('--is_pseudo', type=bool, default=False,
                        help='if True, you will train the model on pseudo-label')
    parser.add_argument('--train_save', type=str, default=None,
                        help='If you use custom save path, please edit `--is_semi=True` and `--is_pseudo=True`')
    parser.add_argument('--is_data_augment', type=bool, default=False)
    parser.add_argument('--random_cutout', type=float, default=0)

    # testing dataset
    parser.add_argument('--test_path', type=str, default="./Dataset/TestingSet/LungInfection-Test/")
    parser.add_argument('--val_path', type=str, default="./Dataset/ValSet/LungInfection-Val")
    parser.add_argument('--all_path', type=str, default="./Dataset/AllSet/LungInfection-All")
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--valsize', type=int, default=352, help='validation size')


    # load model path
    parser.add_argument('--load_net_path', type=str)

    # new techniques
    parser.add_argument('--focal_loss', action='store_true')
    parser.add_argument('--lookahead', action='store_true')

    # save log tensorboard
    parser.add_argument('--graph_path', type=str, default="./graph_log")
    parser.add_argument('--is_eval', type=bool, default=False)
    parser.add_argument('--metric_path', type=str, default='./metrics_log')
    parser.add_argument('--eval_threshold', type=float, help='Use for threshold the sigmoid to get 1 or 0')

    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(opt.gpu_device)

    if opt.backbone == 'Res2Net50':
        print('Backbone loading: Res2Net50')
        from Code.model_lung_infection.InfNet_Res2Net import Inf_Net
    elif opt.backbone == 'ResNet50':
        print('Backbone loading: ResNet50')
        from Code.model_lung_infection.InfNet_ResNet import Inf_Net
    elif opt.backbone == 'VGGNet16':
        print('Backbone loading: VGGNet16')
        from Code.model_lung_infection.InfNet_VGGNet import Inf_Net
    else:
        raise ValueError('Invalid backbone parameters: {}'.format(opt.backbone))

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.random.manual_seed(opt.seed)
    model, optimizer = create_model(opt)

    if opt.load_net_path:
        net_state_dict = torch.load(opt.load_net_path, map_location=torch.device(opt.device))
        net_state_dict = {k: v for k, v in net_state_dict.items() if k in model.state_dict()}
        model.load_state_dict(net_state_dict)

    # ---- load pre-trained weights (mode=Semi-Inf-Net) ----
    # - See Sec.2.3 of `README.md` to learn how to generate your own img/pseudo-label from scratch.
    if opt.is_semi and opt.backbone == 'Res2Net50':
        print('Loading weights from weights file trained on pseudo label')
        model.load_state_dict(torch.load('./Snapshots/save_weights/Inf-Net_Pseduo/Inf-Net_pseudo_100.pth'))
    else:
        print('Not loading weights from weights file')

    # weights file save path
    if opt.is_pseudo and (not opt.is_semi):
        train_save = 'Inf-Net_Pseudo'
    elif (not opt.is_pseudo) and opt.is_semi:
        train_save = 'Semi-Inf-Net'
    elif (not opt.is_pseudo) and (not opt.is_semi):
        train_save = 'Inf-Net'
    else:
        print('Use custom save path')
        train_save = opt.train_save
    train_save = opt.train_save

    # ---- calculate FLOPs and Params ----
    if opt.is_thop:
        from Code.utils.utils import CalParams
        x = torch.randn(1, 3, opt.trainsize, opt.trainsize).to(opt.device)
        CalParams(model, x)

    # ---- load training sub-modules ----
    BCE = torch.nn.BCEWithLogitsLoss()

    train_writer = SummaryWriter(logdir=os.path.join(opt.graph_path, 'training'))
    test_writer = SummaryWriter(logdir=os.path.join(opt.graph_path, 'testing'))

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root,
                              batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=opt.num_workers,
                              is_data_augment=opt.is_data_augment, random_cutout=opt.random_cutout)

    test_image_root = '{}/Imgs/'.format(opt.test_path)
    test_gt_root = '{}/GT/'.format(opt.test_path)
    test_data = test_dataset(test_image_root, test_gt_root, opt.testsize)
    test_loader = DataLoader(test_data, batch_size=opt.batchsize, num_workers=opt.num_workers)

    val_image_root = '{}/Imgs/'.format(opt.val_path)
    val_gt_root = '{}/GT/'.format(opt.val_path)
    val_data = test_dataset(val_image_root, val_gt_root, opt.valsize)
    val_loader = DataLoader(val_data, batch_size=opt.batchsize, num_workers=opt.num_workers)

    total_step = len(train_loader)

    # ---- start !! -----
    print("#"*20, "\nStart Training (Inf-Net-{})\n{}\nThis code is written for 'Inf-Net: Automatic COVID-19 Lung "
                  "Infection Segmentation from CT Scans', 2020, arXiv.\n"
                  "----\nPlease cite the paper if you use this code and dataset. "
                  "And any questions feel free to contact me "
                  "via E-mail (gepengai.ji@163.com)\n----\n".format(opt.backbone, opt), "#"*20)

    if opt.is_eval:
        start = time.time()
        metric_string = eval(test_loader, model, opt.device, opt.load_net_path, opt.eval_threshold, opt)
        end = time.time()
        timer(start, end)

        # write the metrics
        os.makedirs(os.path.join(opt.metric_path, opt.load_net_path), exist_ok=True)
        with open(f'{os.path.join(opt.metric_path, opt.load_net_path, "metrics.txt")}', 'a') as f:
            f.write(metric_string)
    else:

        if opt.folds == 0:
            for epoch in range(1, opt.epoch):
                start = time.time()
                adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
                train(train_loader, val_loader, model, optimizer, epoch, train_save, opt.device, opt)
                end = time.time()
                timer(start, end)
        else:
            del train_loader, val_loader
            cross_validation(train_save, opt)

