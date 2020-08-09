import math
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def performMetrics(hist):
    
    pixel_accuracy = np.diag(hist).sum() / hist.sum()
    mean_accuracy = np.diag(hist) / hist.sum(1)
    mean_iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * mean_iou[freq > 0]).sum()
        
    return 100*np.nanmean(mean_iou), 100*pixel_accuracy, 100*fwavacc


def dice_similarity_coefficient(predicted_seg, ground_truth_seg, threshold):
    a = predicted_seg.contiguous().view(-1)
    b = ground_truth_seg.contiguous().view(-1)

    if threshold:
        a[a >= threshold] = 1
        a[a < threshold] = 0

    intersection = (a * b).sum()
    dice = ((2. * intersection) / (a.sum() + b.sum())).item()
    return dice


def jaccard_similarity_coefficient(predicted_seg, ground_truth_seg, threshold):
    a = predicted_seg.contiguous().view(-1)
    b = ground_truth_seg.contiguous().view(-1)

    if threshold:
        a[a >= threshold] = 1
        a[a < threshold] = 0

    intersection = (a * b).abs().sum()
    sum_ = torch.sum(a.abs() + b.abs())
    jaccard = ((intersection) / (sum_ - intersection)).item()
    return jaccard


def sensitivity_similarity_coefficient(predicted_seg, ground_truth_seg, threshold):
    a = predicted_seg.contiguous().view(-1).detach().cpu().numpy()
    b = ground_truth_seg.contiguous().view(-1).detach().cpu().numpy()

    if threshold:
        a[a >= threshold] = 1
        a[a < threshold] = 0

    tn, fp, fn, tp = confusion_matrix(b, a, labels=[0, 1]).ravel()
    sensitivity = tp / (fn + tp)
    # true_positive = (a * b).sum().detach().cpu().numpy()  # because ground truth is 1, and remove all the other false positive
    # false_negative = (b - a).detach().cpu().numpy()
    # false_negative[false_negative < 0] = 0
    # false_negative = false_negative.sum()
    # sensitivity = true_positive/(false_negative + true_positive + 1e-6)
    return sensitivity


def specificity_similarity_coefficient(predicted_seg, ground_truth_seg, threshold):
    a = predicted_seg.view(-1).contiguous().detach().cpu().numpy()
    b = ground_truth_seg.view(-1).contiguous().detach().cpu().numpy()

    if threshold:
        a[a >= threshold] = 1
        a[a < threshold] = 0

    tn, fp, fn, tp = confusion_matrix(b, a, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp)
    return specificity


def precision_similarity_coefficient(predicted_seg, ground_truth_seg, threshold):
    a = predicted_seg.contiguous().view(-1).detach().cpu().numpy()
    b = ground_truth_seg.contiguous().view(-1).detach().cpu().numpy()

    if threshold:
        a[a >= threshold] = 1
        a[a < threshold] = 0

    tn, fp, fn, tp = confusion_matrix(b, a, labels=[0, 1]).ravel()
    precision = tp / (tp + fp)
    return precision


