## Self-supervised Deep Learning Model for COVID-19 CT Lung Image Segmentation Highlighting Putative Causal Relationship among Age, Underlying Disease and COVID-19

    Daryl Fung Lerh Xing, Qian Liu, Judah Zammit, Carson Kai-Sang Leung, PingZhao Hu
    
### Getting Started

---

#### Process data for self-supervision
Prepare dataset for Self-Supervised learning from ICTCF. You can obtain the processed ICTCF dataset [here](https://mega.nz/file/b5AkQSoa#wk40ZPGB8IUCLQBU0S8dZl1SJBYeU9TtB3EDeBBbKP0).
    
Once you have downloaded the ```ictcf-dataset.zip```, move the ictcf-dataset to the directory of Self-supervised-CT-segmentatiobn. Then run 

    python process_ictcf_for_self.py -- input_folder ictcf-dataset --output_folder ./datasets/medseg
    
---

#### Training for self-supervision
After running the processing data for self-supervision, you can start training for self-supervised for InfNet.

To train for the single InfNet:

    python main_inf-net.py --save_path model/self-inf-net --graph_path graphs/self-inf-net --device cuda --seed 7 --batchsize 64
    
To train for the multi InfNet:

    python main_multi-inf-net.py --save_path model/self_multi_improved --graph_path graphs/graph_self_multi_improved --device cuda --seed 7


feel free to change the device to cpu if you do not want to use cuda.

You can visualise the training graph by running tensorboard on:

    tensorboard --logdir=./graphs
    
####

--- 

#### Preparing InfNet for training

W need to get the dataset for the InfNet first. The data can be found [here](https://mega.nz/file/e8hHSKgZ#VKAYJ0lVWT8a27sv4ObvPGuLf1YFE0sZfdTYmLCpiBY).

    

Once the data is obtained, go to InfNet directory and extract the zip file there. There should be a file called Dataset being created. Place that Dataset folder in InfNet directory.

##### <b style="color: red">Note: All the following code at the bottom will be ran inside InfNet folder</b> 

---

#### Combine Training, Validation, Testing Set 
Combine the different sets of dataset into one folder in order to carry out cross validation

```
python combine_dataset.py
```

---

#### Note that:

There is also a run-self-inf-net.sh and run-self-multi-inf-net.sh to see all the available scripts to run the code for training and evaluating

---

#### Training baseline InfNet

Navigate to the InfNet directory, then run

For the single InfNet: 

    python MyTrain_LungInf.py --train_save baseline-inf-net --random_cutout 0 --graph_path graphs/graph_baseline-inf-net --device cuda --epoch 500 --batchsize 8


For the multi InfNet:

    python MyTrainMulClsLungInf_UNet.py --train_save self-multi-inf-net --random_cutout 0 --graph_path graphs/graph_baseline-multi-inf-net --device cuda --epoch 500 --model_name baseline --batchsize 8
    
 

#### Training InfNet on self-supervised weights

Navigate to the InfNet directory, then run 

For the single InfNet:

    python MyTrain_LungInf.py --train_save self-inf-net --random_cutout 0 --graph_path graphs/graph_self-inf-net --load_net_path ../model/self-inf-net/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 500 --batchsize 8

For the multi InfNet:

    python MyTrain_MulClsLungInf_UNet.py --train_save self-multi-inf-net --random_cutout 0 --graph_path graphs/graph_self-multi-inf-net --load_net_path ../model/self_multi_improved/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 500 --model_name improved --batchsize 8


You can visualise the training graph by running tensorboard on:

    tensorboard --logdir=./graphs


### Running Cross-validation 
For baseline single InfNet:

```
python MyTrain_LungInf.py --train_save baseline-inf-net-cross-val --fold 5 --eval_threshold 0.5 --random_cutout 0 --seed 100 --graph_path graph_baseline-inf-net-cross-val --device cuda --epoch 500 --batchsize 8
```

For self-supervised single InfNet:

```buildoutcfg
python MyTrain_LungInf.py --train_save self-improved-cross-val --fold 5 --eval_threshold 0.5 --focal_loss --lookahead --is_data_augment True --random_cutout 0.5 --seed 100 --graph_path graph_self-improved-cross-val --load_net_path ../model/self-inf-net/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 500 --batchsize 8
```

For baseline multi InfNet:

```buildoutcfg
python MyTrain_MulClsLungInf_UNet.py --folds 5 --save_path multi-inf-net-cross-val --random_cutout 0  --graph_path graph_multi-inf-net-cross-val --device cuda --epoch 500 --batchsize 4
```

For self-supervised multi InfNet:

```buildoutcfg
python MyTrain_MulClsLungInf_UNet.py --folds 5 --save_path self-multi-improved-inf-net-cross-val --seed 100 --is_data_augment True --random_cutout 0.5 --is_label_smooth True --graph_path graph_self-multi-inf-net-cross-val --model_name improved --load_net_path ../model/self_multi_improved_new/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 500 --batchsize 8
```


### Evaluating models

For Single InfNet, you can run:
    
    python MyTrain_LungInf.py --is_eval True --eval_threshold 0.5 --load_net_path './Snapshots/save_weights/self-inf-net/{checkpoint_model_name}'

Feel free the change the threshold for the optimal value instead of 0.5 after the evaluation has occured, it will print out the best optimal threshold to use.

After evaluating the models, you could visualise the ROC curve for the models as they are saved in the roc_savces directory by default. You can run:

    python rocs_generation.py

It will generate a plot between the ROC curve for different models.

For the multi InfNet, you can run:

    python MyTrain_MulClsLungInf_UNet.py --is_eval True --load_net_path './Snapshots/save_weights/self-multi-inf-net/{checkpoint_model_name}' --model_name improved

Feel free to change the load_net_path argument to suits your need. If you want to evaluate on the baseline, you can change the load_net_Path to point to the baseline directory accordingly.

If you want to compare different models, for instance self InfNet vs baseline InfNet, then you can pass another argument called load_net_path_2 to the argument for multi class InfNet. For instance:

    python MyTrain_MulClsLungInf_UNet.py --is_eval True --load_net_path './Snapshots/save_weights/self-multi-inf-net/{checkpoint_model_name}' --model_name improved --load_net_path_2 './Snapshots/save_weights/baseline-multi-inf-net/{checkpoint_model_name}' --model_name_2 baseline

This will compare the models and calculate the p-value significant value between the models using Wilcox Test.

For all the evaluation, a folder will be created called "metrics_log". You will be able to see all the dice, jaccard, sensitivity, precision metrics there from the eval models ran.

### Generating Results

The pre-trained models can be found [here](https://mega.nz/file/b9p0GIAJ#qzbY8yhcQ_e3XDDwIvWHayT4TssZCynvqGhIosvotm4). Extract the zip file into this directory: ```./Snapshots/save_weights```

#### baseline InfNet
In order to generate results for the baseline Single InfNet, run:

    python MyTest_LungInf.py --pth_path './Snapshots/save_weights/baseline-inf-net/{model_checkpoint_name}' --save_path './Results/Lung infection segmentation/single_baseline-inf-net/'
    
The results for Single InfNet will be saved in './Results/Lung infection segmentation/single_self-inf-net'

In order to generate results for the baseline Multi InfNet run:

    python MyTest_MulClsLungInf_UNet.py --model_name baseline --save_path './Results/Lung infection segmentation/single_baseline-inf-net/' --pth_path './Snapshots/save_weights/baseline-multi-inf-net/{model_checkpoint_name}' --pseudo_path './Results/Lung infection segmentation/single_baseline-inf-net/'


#### Self-Supervised InfNet

In order to generate results for the self Single InfNet, run:

    python MyTest_LungInf.py --pth_path './Snapshots/save_weights/self-inf-net/{model_checkpoint_name}' --save_path './Results/Lung infection segmentation/single_self-inf-net/'
    
The results for Single InfNet will be saved in './Results/Lung infection segmentation/single_self-inf-net'

In order to generate results for the self Multi InfNet run:

    python MyTest_MulClsLungInf_UNet.py --model_name improved --save_path './Results/Lung infection segmentation/single_self-inf-net/' --pth_path './Snapshots/save_weights/self-multi-inf-net/{model_checkpoint_name}' --pseudo_path './Results/Lung infection segmentation/single_self-inf-net/' 

#### References:

Most of the code for the self-supervision is obtained from [self-supervision-for-segmenting-overhead-imagery](https://github.com/suriyasingh/Self-supervision-for-segmenting-overhead-imagery).
As for the InfNet, most of the code is obtained from [InfNet](https://github.com/DengPingFan/Inf-Net).

They are made with changes to improve on the COVID-19 CT lung images.
 
