import numpy as np

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
    smooth = 1.
    a = predicted_seg.view(-1).sigmoid()
    b = ground_truth_seg.view(-1)
    intersection = (a * b).sum()
    return 1 - ( (2. * intersection + smooth) / (a.sum() + b.sum() + smooth) )


    k = 1
    dice = np.sum(predicted_seg[ground_truth_seg==k]) * 2 / (np.sum(predicted_seg[predicted_seg==k]==k) + np.sum(ground_truth_seg[ground_truth_seg==k]==k))
    return dice
