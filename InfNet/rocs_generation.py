import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import roc_curve, auc

roc_saves_dir = './roc_saves'
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
plt.figure()
all_network_rocs = os.listdir(roc_saves_dir)

for index, roc_file in enumerate(all_network_rocs):
    roc_filename = os.path.join(roc_saves_dir, roc_file)

    with open(roc_filename, 'rb') as f:
        roc_dict = pickle.load(f)

    roc_auc = auc(roc_dict['fpr'], roc_dict['tpr'])
    lw = 2
    plt.plot(roc_dict['fpr'], roc_dict['tpr'], color=colors[index],
             lw=lw, label='ROC curve (area = %0.2f) %s' % (roc_auc, roc_file))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(roc_dict['optimal_fpr'], roc_dict['optimal_tpr'], 'go')
    plt.annotate(f'{round(roc_dict["optimal_threshold"], 5)}', (roc_dict['optimal_fpr'], roc_dict['optimal_tpr']))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

plt.show()