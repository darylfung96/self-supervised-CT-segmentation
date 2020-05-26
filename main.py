## import libraries
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import torch.optim as optim
import torch.nn.functional as F
from utils.dataloaders import context_inpainting_dataloader, segmentation_data_loader
from models import resnet18_encoderdecoder, resnet18_encoderdecoder_wbottleneck
from models import resnet18_coach_vae

import warnings
warnings.filterwarnings('ignore')

## fix seeds
torch.cuda.manual_seed(7)
torch.manual_seed(7)
np.random.seed(7)

dataset_root = '/Users/darylfung/programming/Self-supervision-for-segmenting-overhead-imagery/datasets/'
model_root = '/Users/darylfung/programming/Self-supervision-for-segmenting-overhead-imagery/model/'

dataset = 'potsdam'                                    #options are: spacenet, potsdam, deepglobe_roads, deepglobe_lands
architecture = 'resnet18_autoencoder_no_bottleneck'    #options are: resnet18_autoencoder, resnet18_encoderdecoder_wbottleneck
use_coach = True                                       #options are: True or Flase
self_supervised_split = 'train_crops'                  #options are: train_10crops, train_25crops, train_50crops, train_crops
supervised_split = 'train_10crops'                     #options are: train_10crops, train_25crops, train_50crops, train_crops

experiment = dataset + '_' + architecture                #model file suffix

if use_coach:
    experiment += '_' + 'use_coach'

mean_bgr = np.array([85.5517787014, 92.6691667083, 86.8147645556])       # mean BGR values of images
std_bgr = np.array([32.8860206505, 31.7342205253, 31.5361127226])        # standard deviation BGR values of images

### set data paths
splits = None
train_image_list_path = None
train_img_root = None
train_gt_root = None

val_image_list = None
val_img_root = None
val_gt_root = None

nClasses = None
ignore_class = None


if dataset == 'spacenet':
    train_img_root = dataset_root + 'spacenet/spacenet_processed/train/images/'
    train_gt_root = dataset_root + 'spacenet_processed/train/gt/'

    val_img_root = dataset_root + 'spacenet/spacenet_processed/val/images/'
    val_gt_root = dataset_root + 'spacenet/spacenet_processed/val/gt/'
    val_image_list = dataset_root + 'spacenet/splits/val_crops.txt'

    train_image_list_path = dataset_root + 'spacenet/splits/'

    nClasses = 2                ### number of classes for pixelwise classification
    out = 'seg'                 ### process ground-truth as binary segmentation

elif dataset == 'potsdam':
    train_img_root = dataset_root + 'potsdam/processed/train/images/'
    train_gt_root = dataset_root + 'potsdam/processed/train/gt/'

    val_img_root = dataset_root + 'potsdam/processed/val/images/'
    val_gt_root = dataset_root + 'potsdam/processed/val/gt/'
    val_image_list = dataset_root + 'potsdam/splits/val_crops.txt'

    train_image_list_path = dataset_root + 'potsdam/splits/'
    nClasses = 6                ### number of classes for pixelwise classification
    out = None                  ### do not process ground-truth

elif dataset == 'deepglobe_roads':
    train_img_root = dataset_root + 'deepglobe_roads/processed/train/images/'
    train_gt_root = dataset_root + 'deepglobe_roads/processed/train/gt/'

    val_img_root = dataset_root + 'deepglobe_roads/processed/val/images/'
    val_gt_root = dataset_root + 'deepglobe_roads/processed/val/gt/'
    val_image_list = dataset_root + 'deepglobe_roads/splits/val_crops.txt'

    train_image_list_path = dataset_root + 'deepglobe_roads/splits/'

    nClasses = 2                ### number of classes for pixelwise classification
    out = 'seg'                 ### process ground-truth as binary segmentation

elif dataset == 'deepglobe_lands':
    train_img_root = dataset_root + 'deepglobe_lands/processed/train/images/'
    train_gt_root = dataset_root + 'deepglobe_lands/processed/train/gt/'

    val_img_root = dataset_root + 'deepglobe_lands/processed/val/images/'
    val_gt_root = dataset_root + 'deepglobe_lands/processed/val/gt/'
    val_image_list = dataset_root + 'deepglobe_lands/splits/val_crops.txt'

    train_image_list_path = dataset_root + 'deepglobe_lands/splits/'
    nClasses = 7                ### number of classes for pixelwise classification
    out = None                  ### do not process ground-truth
    ignore_class = 6

erase_shape = [16, 16]         ### size of each block used to erase image
erase_count = 16               ### number of blocks to erase from image
rec_weight = 0.99            ### loss = rec_weight*loss_rec+ (1-rec_weight)*loss_con

train_loader = torch.utils.data.DataLoader(
    context_inpainting_dataloader(img_root = train_img_root, image_list = train_image_list_path+self_supervised_split+'.txt', suffix=dataset,
                                  mirror = True, resize=True, resize_shape=[256, 256], rotate = True,
                                  erase_shape = erase_shape, erase_count = erase_count),
    batch_size=128, num_workers=10, shuffle = True)

val_loader = torch.utils.data.DataLoader(
    context_inpainting_dataloader(img_root = val_img_root, image_list = val_image_list, suffix=dataset,
                                  mirror = False, resize=False, resize_shape=[256, 256], rotate = False,
                                  crop = True, erase_shape = erase_shape, erase_count = erase_count),
    batch_size=32, num_workers=10, shuffle = False)


def torch_to_np(input_, mask, target, output=None):
    input_ = np.asarray(input_.numpy().transpose(1, 2, 0) + mean_bgr[np.newaxis, np.newaxis, :], dtype=np.uint8)[:, :,
             ::-1]
    mask = np.asarray(mask[0].numpy(), dtype=np.uint8)
    target = np.asarray(3 * std_bgr * (target.numpy().transpose(1, 2, 0)) + mean_bgr[np.newaxis, np.newaxis, :],
                        dtype=np.uint8)[:, :, ::-1]

    if output is not None:
        output = np.asarray(3 * std_bgr * (output.numpy().transpose(1, 2, 0)) + mean_bgr[np.newaxis, np.newaxis, :],
                            dtype=np.uint8)[:, :, ::-1]
    return input_, mask, target, output


def visualize_self_sup(cols=3, net=None, coach=None, use_coach_masks=False):
    fig, axs = plt.subplots(nrows=4, ncols=cols, figsize=(9, 9))

    for batch_idx, (inputs_, masks, targets) in enumerate(val_loader):
        if coach is None:
            inputs_ = inputs_ * masks.float()
        else:
            masks, _, _ = coach.forward(inputs_.cuda(), alpha=100, use_coach=use_coach_masks)
            inputs_ = inputs_ * masks.float().cpu()

        outputs = None
        if cols == 4:
            outputs = net.forward(inputs_.cuda()).cpu().data
            input_, mask, target, output = torch_to_np(inputs_[0].cpu(), masks[0].cpu(), targets[0].cpu(),
                                                       outputs[0].cpu())
        else:
            input_, mask, target, _ = torch_to_np(inputs_[0].cpu(), masks[0].cpu(), targets[0].cpu())
        axs[batch_idx, 0].imshow(input_)
        axs[batch_idx, 1].imshow(mask, cmap='gray')
        axs[batch_idx, 2].imshow(target)
        if cols == 4:
            axs[batch_idx, 3].imshow(output)
        if batch_idx == 3:
            break

    axs[0, 0].set_title('input', fontsize=18)
    axs[0, 1].set_title('mask', fontsize=18)
    axs[0, 2].set_title('target', fontsize=18)
    if cols == 4:
        axs[0, 3].set_title('semantic inpainting', fontsize=18)
    fig.tight_layout()
    plt.show()

visualize_self_sup()

net = resnet18_encoderdecoder().cuda()
net_coach = None

if use_coach:
    net_coach = resnet18_coach_vae(drop_ratio=0.75).cuda()

net_optimizer = None
coach_optimizer = None
best_loss = 1e5
train_loss = []
val_loss = []
coach_loss = []


def train_context_inpainting(epoch, net, net_optimizer, coach=None, use_coach_masks=False):
    progbar = tqdm_notebook(total=len(train_loader), desc='Train')
    net.train()

    if coach is not None:
        coach.eval()

    train_loss.append(0)
    for batch_idx, (inputs_, masks, targets) in enumerate(train_loader):
        net_optimizer.zero_grad()
        inputs_, masks, targets = Variable(inputs_.cuda()), Variable(masks.cuda().float()), Variable(targets.cuda())

        if coach is not None:
            masks, _, _ = coach.forward(inputs_, alpha=100, use_coach=use_coach_masks)

        outputs_1 = net(inputs_ * masks)
        mse_loss = (outputs_1 - targets) ** 2
        mse_loss = -1 * F.threshold(-1 * mse_loss, -2, -2)
        loss_rec = torch.sum(mse_loss * (1 - masks)) / torch.sum(1 - masks)
        if coach is not None:
            loss_con = torch.sum(mse_loss * masks) / torch.sum(masks)
        else:
            outputs_2 = net(inputs_ * (1 - masks))
            mse_loss = (outputs_2 - targets) ** 2
            mse_loss = -1 * F.threshold(-1 * mse_loss, -2, -2)
            loss_con = torch.sum(mse_loss * masks) / torch.sum(masks)

        total_loss = rec_weight * loss_rec + (1 - rec_weight) * loss_con
        total_loss.backward()

        net_optimizer.step()

        train_loss[-1] += total_loss.data
        progbar.set_description('Train (loss=%.4f)' % (train_loss[-1] / (batch_idx + 1)))
        progbar.update(1)
    train_loss[-1] = train_loss[-1] / len(train_loader)


def train_coach(epoch, net, coach, coach_optimizer):
    progbar = tqdm_notebook(total=len(train_loader), desc='Coach')
    coach.train()
    net.eval()
    coach_loss.append(0)
    for batch_idx, (inputs_, masks, targets) in enumerate(train_loader):
        coach_optimizer.zero_grad()
        inputs_, targets = Variable(inputs_.cuda()), Variable(targets.cuda())

        masks, mu, logvar = coach.forward(inputs_, alpha=1)

        outputs = net(inputs_ * masks).detach()
        mse_loss = (outputs - targets) ** 2
        mse_loss = -1 * F.threshold(-1 * mse_loss, -2, -2)
        loss_rec = torch.sum(mse_loss * (1 - masks)) / (3 * torch.sum(1 - masks))

        mu = mu.mean(dim=2).mean(dim=2)
        logvar = logvar.mean(dim=2).mean(dim=2)

        KLD = 0
        try:
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        except:
            KLD = 0

        total_loss = 1 - loss_rec + 1e-6 * KLD

        total_loss.backward()
        coach_optimizer.step()

        coach_loss[-1] += total_loss.data
        progbar.set_description('Coach (loss=%.4f)' % (coach_loss[-1] / (batch_idx + 1)))
        progbar.update(1)
    coach_loss[-1] = coach_loss[-1] / len(train_loader)


def val_context_inpainting(iter_, epoch, net, coach=None, use_coach_masks=False):
    global best_loss
    progbar = tqdm_notebook(total=len(val_loader), desc='Val')
    net.eval()
    if coach is not None:
        coach.eval()
    val_loss.append(0)
    for batch_idx, (inputs_, masks, targets) in enumerate(val_loader):
        inputs_, masks, targets = Variable(inputs_.cuda()), Variable(masks.cuda().float()), Variable(targets.cuda())

        if coach is not None:
            masks, _, _ = coach.forward(inputs_, alpha=100, use_coach=use_coach_masks)

        outputs_1 = net(inputs_ * masks)
        mse_loss = (outputs_1 - targets) ** 2
        mse_loss = -1 * F.threshold(-1 * mse_loss, -2, -2)
        loss_rec = torch.sum(mse_loss * (1 - masks)) / torch.sum(1 - masks)

        outputs_2 = net(inputs_ * (1 - masks))
        mse_loss = (outputs_2 - targets) ** 2
        mse_loss = -1 * F.threshold(-1 * mse_loss, -2, -2)
        loss_con = torch.sum(mse_loss * masks) / torch.sum(masks)

        total_loss = rec_weight * loss_rec + (1 - rec_weight) * loss_con

        val_loss[-1] += total_loss.data
        progbar.set_description('Val (loss=%.4f)' % (val_loss[-1] / (batch_idx + 1)))
        progbar.update(1)

    val_loss[-1] = val_loss[-1] / len(val_loader)
    if best_loss > val_loss[-1]:
        best_loss = val_loss[-1]
        print('Saving..')
        state = {'context_inpainting_net': net, 'coach': coach}

        torch.save(state, model_root + experiment + str(iter_) + '.ckpt.t7')

use_coach_masks = False
epochs = []
lrs = []

if use_coach:
    epochs = [100, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    lrs = [[1e-1, 1e-2, 1e-3, 1e-4],
       [1e-5, 1e-5, 1e-5, 1e-5],
       [1e-5, 1e-5, 1e-5, 1e-5],
       [1e-5, 1e-5, 1e-5, 1e-5],
       [1e-5, 1e-5, 1e-5, 1e-5],
       [1e-5, 1e-5, 1e-5, 1e-5],
       [1e-5, 1e-5, 1e-5, 1e-5],
       [1e-5, 1e-5, 1e-5, 1e-5],
       [1e-5, 1e-5, 1e-5, 1e-5],
       [1e-5, 1e-5, 1e-5, 1e-5],
       [1e-5, 1e-5, 1e-5, 1e-5]]
else:
    epochs = [100]
    lrs = [[1e-1, 1e-2, 1e-3, 1e-4]]

progbar_1 = tqdm_notebook(total=len(epochs), desc='Iters')
for iter_ in range(0, len(epochs)):
    best_loss = 1e5

    if use_coach and iter_ > 0:
        use_coach_masks = True
        progbar_2 = tqdm_notebook(total=epochs[iter_], desc='Epochs')
        optimizer_coach = optim.Adam(net_coach.parameters(), lr=1e-5)

        for epoch in range(epochs[iter_]):
            train_coach(epoch, net=net, coach=net_coach, coach_optimizer=optimizer_coach)
            progbar_2.update(1)

    net_optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    progbar_2 = tqdm_notebook(total=epochs[iter_], desc='Epochs')
    for epoch in range(epochs[iter_]):
        if epoch % 10 == 0:
            if use_coach:
                visualize_self_sup(cols=4, net=net.eval(), coach=net_coach.eval(), use_coach_masks=use_coach_masks)
            else:
                visualize_self_sup(cols=4, net=net.eval(), coach=None, use_coach_masks=use_coach_masks)

        if epoch == 90:
            net_optimizer = optim.SGD(net.parameters(), lr=lrs[iter_][3], momentum=0.9, weight_decay=5e-4)
        if epoch == 80:
            net_optimizer = optim.SGD(net.parameters(), lr=lrs[iter_][2], momentum=0.9, weight_decay=5e-4)
        if epoch == 40:
            net_optimizer = optim.SGD(net.parameters(), lr=lrs[iter_][1], momentum=0.9, weight_decay=5e-4)
        if epoch == 0:
            net_optimizer = optim.SGD(net.parameters(), lr=lrs[iter_][0], momentum=0.9, weight_decay=5e-4)

        train_context_inpainting(epoch, net=net, net_optimizer=net_optimizer, coach=net_coach,
                                 use_coach_masks=use_coach_masks)
        val_context_inpainting(iter_, epoch, net=net, coach=net_coach, use_coach_masks=use_coach_masks)

        progbar_2.update(1)

    progbar_1.update(1)

from utils.printing import training_curves_loss
training_curves_loss(train_loss, val_loss)

del(net_coach)
del(net)
torch.cuda.empty_cache()

from models import FCNify_v2
iter_ = len(epochs) - 1   ### iter_ = 0 is semantic inpainting model, iter_ > 0 is trained against coach masks
net = torch.load(model_root + experiment + str(iter_) + '.ckpt.t7')['context_inpainting_net']
net_segmentation = FCNify_v2(net, n_class = nClasses).cuda()
optimizer_seg = None
del(net)

from loss import soft_iou
from metric import fast_hist, performMetrics
from utils.dataloaders import segmentation_data_loader

train_seg_loss = []
val_seg_loss = []
train_seg_iou = []
val_seg_iou = []
ITER_SIZE = 2    ### accumulate gradients over ITER_SIZE iterations
best_iou = 0.

train_seg_loader = torch.utils.data.DataLoader(
    segmentation_data_loader(img_root = train_img_root, gt_root = train_gt_root, image_list = train_image_list_path+supervised_split+'.txt',
                             suffix=dataset, out=out, crop = True, crop_shape = [256, 256], mirror = True),
                                           batch_size=32, num_workers=8, shuffle = True)

val_seg_loader = torch.utils.data.DataLoader(
    segmentation_data_loader(img_root = val_img_root, gt_root = val_gt_root, image_list = val_image_list,
                             suffix=dataset, out=out, crop = False, mirror=False),
                                           batch_size=8, num_workers=8, shuffle = False)


def train_segmentation(epoch, net_segmentation, seg_optimizer):
    global train_seg_iou
    progbar = tqdm_notebook(total=len(train_seg_loader), desc='Train')
    net_segmentation.train()

    train_seg_loss.append(0)
    seg_optimizer.zero_grad()
    hist = np.zeros((nClasses, nClasses))
    for batch_idx, (inputs_, targets) in enumerate(train_seg_loader):
        inputs_, targets = Variable(inputs_.cuda()), Variable(targets.cuda())

        outputs = net_segmentation(inputs_)

        total_loss = (1 - soft_iou(outputs, targets, ignore=ignore_class)) / ITER_SIZE
        total_loss.backward()

        if (batch_idx % ITER_SIZE == 0 and batch_idx != 0) or batch_idx == len(train_loader) - 1:
            seg_optimizer.step()
            seg_optimizer.zero_grad()

        train_seg_loss[-1] += total_loss.data

        _, predicted = torch.max(outputs.data, 1)
        correctLabel = targets.view(-1, targets.size()[1], targets.size()[2])
        hist += fast_hist(correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
                          predicted.view(predicted.size(0), -1).cpu().numpy(),
                          nClasses)

        miou, p_acc, fwacc = performMetrics(hist)

        progbar.set_description('Train (loss=%.4f, mIoU=%.4f)' % (train_seg_loss[-1] / (batch_idx + 1), miou))
        progbar.update(1)
    train_seg_loss[-1] = train_seg_loss[-1] / len(train_seg_loader)
    miou, p_acc, fwacc = performMetrics(hist)
    train_seg_iou += [miou]


def val_segmentation(epoch, net_segmentation):
    global best_iou
    global val_seg_iou
    progbar = tqdm_notebook(total=len(val_seg_loader), desc='Val')
    net_segmentation.eval()

    val_seg_loss.append(0)
    hist = np.zeros((nClasses, nClasses))
    for batch_idx, (inputs_, targets) in enumerate(val_seg_loader):
        inputs_, targets = Variable(inputs_.cuda()), Variable(targets.cuda())

        outputs = net_segmentation(inputs_)

        total_loss = 1 - soft_iou(outputs, targets, ignore=ignore_class)

        val_seg_loss[-1] += total_loss.data

        _, predicted = torch.max(outputs.data, 1)
        correctLabel = targets.view(-1, targets.size()[1], targets.size()[2])
        hist += fast_hist(correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
                          predicted.view(predicted.size(0), -1).cpu().numpy(),
                          nClasses)

        miou, p_acc, fwacc = performMetrics(hist)

        progbar.set_description('Val (loss=%.4f, mIoU=%.4f)' % (val_seg_loss[-1] / (batch_idx + 1), miou))
        progbar.update(1)
    val_seg_loss[-1] = val_seg_loss[-1] / len(val_seg_loader)
    val_miou, _, _ = performMetrics(hist)
    val_seg_iou += [val_miou]

    if best_iou < val_miou:
        best_iou = val_miou
        print('Saving..')
        state = {'net_segmentation': net_segmentation}

        torch.save(state, model_root + experiment + 'segmentation' + '.ckpt.t7')


progbar = tqdm_notebook(total=100, desc='Epochs')
for epoch in range(0, 100):
    if epoch == 90:
        seg_optimizer = optim.SGD(net_segmentation.parameters(), lr=1e-6, momentum=0.9, weight_decay=5e-4)
    elif epoch == 80:
        seg_optimizer = optim.SGD(net_segmentation.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
    elif epoch == 60:
        seg_optimizer = optim.SGD(net_segmentation.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    elif epoch == 0:
        seg_optimizer = optim.SGD(net_segmentation.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    train_segmentation(epoch, net_segmentation=net_segmentation, seg_optimizer=seg_optimizer)
    val_segmentation(epoch, net_segmentation=net_segmentation)
    progbar.update(1)

progbar = tqdm_notebook(total=100, desc='Epochs')
for epoch in range(0, 100):
    if epoch == 90:
        seg_optimizer = optim.SGD(net_segmentation.parameters(), lr=1e-6, momentum=0.9, weight_decay=5e-4)
    elif epoch == 80:
        seg_optimizer = optim.SGD(net_segmentation.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
    elif epoch == 60:
        seg_optimizer = optim.SGD(net_segmentation.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    elif epoch == 0:
        seg_optimizer = optim.SGD(net_segmentation.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    train_segmentation(epoch, net_segmentation=net_segmentation, seg_optimizer=seg_optimizer)
    val_segmentation(epoch, net_segmentation=net_segmentation)
    progbar.update(1)

from utils.printing import segmentation_training_curves_loss, apply_color_map
segmentation_training_curves_loss(train_seg_loss, val_seg_loss, train_seg_iou, val_seg_iou)

del(net_segmentation)
torch.cuda.empty_cache()

c_map = np.asarray([[128, 128, 128], [128, 128, 0], [0, 64, 0], [0, 128, 0], [128, 0, 0], [0, 0, 0]])


def visualize_segmentation(net_segmentation):
    val_seg_loader = torch.utils.data.DataLoader(
        segmentation_data_loader(img_root=val_img_root, gt_root=val_gt_root, image_list=val_image_list,
                                 suffix=dataset, out=out, crop=False, mirror=False),
        batch_size=1, num_workers=8, shuffle=False)
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(9, 9))
    for batch_idx, (inputs_, targets) in enumerate(val_seg_loader):
        inputs_, targets = Variable(inputs_.cuda()), Variable(targets.cuda())

        outputs = net_segmentation(inputs_)

        _, predicted = torch.max(outputs.data, 1)

        input_ = np.asarray(inputs_[0].cpu().numpy().transpose(1, 2, 0) + mean_bgr[np.newaxis, np.newaxis, :],
                            dtype=np.uint8)[:, :, ::-1]
        axs[batch_idx, 0].imshow(input_)
        axs[batch_idx, 1].imshow(apply_color_map(targets[0].cpu().data, c_map))
        axs[batch_idx, 2].imshow(apply_color_map(predicted[0].cpu().data, c_map))
        if batch_idx == 3:
            break

    axs[0, 0].set_title('input', fontsize=18)
    axs[0, 1].set_title('GT', fontsize=18)
    axs[0, 2].set_title('Pred', fontsize=18)
    fig.tight_layout()
    plt.show()


def evaluate_segmentation(net_segmentation):
    net_segmentation.eval()
    hist = np.zeros((nClasses, nClasses))
    val_seg_loader = torch.utils.data.DataLoader(
        segmentation_data_loader(img_root=val_img_root, gt_root=val_gt_root, image_list=val_image_list,
                                 suffix=dataset, out=out, crop=False, mirror=False),
        batch_size=1, num_workers=8, shuffle=False)

    progbar = tqdm_notebook(total=len(val_seg_loader), desc='Eval')

    hist = np.zeros((nClasses, nClasses))
    for batch_idx, (inputs_, targets) in enumerate(val_seg_loader):
        inputs_, targets = Variable(inputs_.cuda()), Variable(targets.cuda())

        outputs = net_segmentation(inputs_)

        _, predicted = torch.max(outputs.data, 1)
        correctLabel = targets.view(-1, targets.size()[1], targets.size()[2])
        hist += fast_hist(correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
                          predicted.view(predicted.size(0), -1).cpu().numpy(),
                          nClasses)

        miou, p_acc, fwacc = performMetrics(hist)
        progbar.set_description('Eval (mIoU=%.4f)' % (miou))
        progbar.update(1)

    miou, p_acc, fwacc = performMetrics(hist)
    print('\n mIoU: ', miou)
    print('\n Pixel accuracy: ', p_acc)
    print('\n Frequency Weighted Pixel accuracy: ', fwacc)

net = torch.load(model_root + experiment + 'segmentation' + '.ckpt.t7')['net_segmentation'].cuda().eval() ### load the best model
evaluate_segmentation(net)

visualize_segmentation(net)