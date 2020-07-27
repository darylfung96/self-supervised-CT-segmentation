import numpy as np
import torch


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


def dice_similarity_coefficient(predicted_seg, ground_truth_seg):
    a = predicted_seg.view(-1)
    b = ground_truth_seg.view(-1)
    intersection = (a * b).sum()
    return ((2. * intersection) / (a.sum() + b.sum())).item()


def jaccard_similarity_coefficient(predicted_seg, ground_truth_seg):
    a = predicted_seg.view(-1)
    b = ground_truth_seg.view(-1)
    intersection = (a * b).abs().sum()
    sum_ = torch.sum(a.abs() + b.abs())
    jaccard = (intersection ) / (sum_ - intersection)
    return jaccard.item()


def sensitivity_similarity_coefficient(predicted_seg, ground_truth_seg):
    a = predicted_seg.view(-1)
    b = ground_truth_seg.view(-1)

    true_positive = (a * b).sum().detach().cpu().numpy()  # because ground truth is 1, and remove all the other false positive
    false_negative = (b - a).detach().cpu().numpy()
    false_negative[false_negative < 0] = 0
    false_negative = false_negative.sum()
    sensitivity = true_positive/(false_negative + true_positive + 1e-6)
    return sensitivity


def specificity_similarity_coefficient(predicted_seg, ground_truth_seg):
    a = predicted_seg.view(-1)
    b = ground_truth_seg.view(-1)

    inverted_b = 1 - b
    inverted_a = 1 - a

    true_negative = (inverted_a * inverted_b).sum().detach().cpu().numpy()
    false_positive = (inverted_b - inverted_a).detach().cpu().numpy()
    false_positive[false_positive < 0] = 0
    false_positive = false_positive.sum()
    specificity = true_negative / (false_positive + true_negative + 1e-6)
    return specificity


