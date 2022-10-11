# check again maybe batchsize can be changed
python main_multi-inf-net.py --save_path model/self_multi_improved --graph_path graph_self_multi_improved --device cuda --seed 100

# baseline
python MyTrain_MulClsLungInf_UNet.py --save_path multi-inf-net --random_cutout 0  --graph_path graph_multi-inf-net --device cuda --epoch 200 --batchsize 4
# data augment
python MyTrain_MulClsLungInf_UNet.py --save_path multi-inf-net04_random_cutout --is_data_augment True --random_cutout 0.4  --graph_path graph_multi-inf-net04_random_cutout --device cuda --epoch 200 --batchsize 4
python MyTrain_MulClsLungInf_UNet.py --save_path multi-inf-net05_random_cutout --is_data_augment True --random_cutout 0.5  --graph_path graph_multi-inf-net05_random_cutout --device cuda --epoch 200 --batchsize 4
# cross-val
python MyTrain_MulClsLungInf_UNet.py --folds 3 --save_path multi-inf-net-cross-val --random_cutout 0  --graph_path graph_multi-inf-net-cross-val --device cuda --epoch 200 --batchsize 4

#fcn

# cross-val
python MyTrain_MulClsLungInf_UNet.py --folds 3 --save_path multi-inf-net-cross-val --random_cutout 0  --graph_path graph_multi-inf-net-cross-val --device cuda --epoch 200 --batchsize 4

python

# self-supervised
python MyTrain_MulClsLungInf_UNet.py --load_net_path ../model/self_multi_improved/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --save_path self-multi-inf-net --random_cutout 0  --graph_path graph_self-multi-inf-net --device cuda --epoch 200 --batchsize 4
python MyTrain_MulClsLungInf_UNet.py --load_net_path ../model/self_multi_improved_new/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --save_path self-multi-inf-net_new --random_cutout 0  --graph_path graph_self-multi-inf-net_new --device cuda:1 --epoch 200 --batchsize 4

python MyTrain_MulClsLungInf_UNet.py --save_path self-multi-inf-net --seed 100 --random_cutout 0 --graph_path graph_self-multi-inf-net --load_net_path ../model/self_multi_improved/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 200 --batchsize 8
python MyTrain_MulClsLungInf_UNet.py --lookahead --focal_loss --save_path self-multi-improved-inf-net --graph_path graph_self-multi-improved-inf-net --seed 100 --is_data_augment True --random_cutout 0.5 --is_label_smooth True  --model_name improved --load_net_path ../model/self_multi_improved_new/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 200 --batchsize 8
# cross-val
python MyTrain_MulClsLungInf_UNet.py --folds 3 --lookahead --focal_loss --save_path self-multi-improved-inf-net-cross-val --seed 100 --is_data_augment True --random_cutout 0.5 --is_label_smooth True --graph_path graph_self-multi-inf-net-cross-val --model_name improved --load_net_path ../model/self_multi_improved_new/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 200 --batchsize 4

# ablation
# self-supervised only
python MyTrain_MulClsLungInf_UNet.py --save_path self-multi-improved-inf-net --seed 100 --is_data_augment True --random_cutout 0.5 --is_label_smooth True --graph_path graph_self-multi-inf-net --model_name improved --load_net_path ../model/self_multi_improved_new/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7--device cuda --epoch 200 --batchsize 4

# focal loss only
python MyTrain_MulClsLungInf_UNet.py --save_path self-multi-improved-inf-net-focal --seed 100 --focal_loss --is_data_augment True --random_cutout 0.5 --is_label_smooth True --graph_path graph_self-multi-inf-net-focal --model_name improved --load_net_path ../model/self_multi_improved_new/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 200 --batchsize 4

# lookahead only
python MyTrain_MulClsLungInf_UNet.py --save_path self-multi-improved-inf-net-lookahead --seed 100 --lookahead --is_data_augment True --random_cutout 0.5 --is_label_smooth True --graph_path graph_self-multi-inf-net-lookahead --model_name improved --load_net_path ../model/self_multi_improved_new/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 200 --batchsize 4


# evaluation
python MyTrain_MulClsLungInf_UNet.py --device cuda --is_eval True --load_net_path './Snapshots/save_weights/self-multi-inf-net/{checkpoint_model_name}' --model_name improved --train_path {train_path} --test_path {test_path} --pseudo_test_path {pseudo_test_path}

# generation
python MyTest_MulClsLungInf_UNet.py --pth_path Snapshots/save_weights/{model weights}--save_path "Results/Multi-class lung infection segmentation/" --pseudo_path {pseudo path} --input_channels 3 --model_name FCN

### (on their small dataset)
# baseline
python MyTrain_MulClsLungInf_UNet.py --save_path multi-inf-net-smalldataset --random_cutout 0  --graph_path graph_multi-inf-net-smalldataset --device cuda --epoch 200 --batchsize 4 --train_path "Dataset(small)/TrainingSet/MultiClassInfection-Train/" --test_path "Dataset(small)/TestingSet/MultiClassInfection-Test/"
# improved
python MyTrain_MulClsLungInf_UNet.py --save_path self-multi-improved-inf-net-smalldataset --graph_path graph_self-multi-improved-inf-net-smalldataset --seed 100 --is_data_augment True --random_cutout 0.5 --is_label_smooth True --model_name improved --load_net_path ../model/self_multi_improved_new/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 200 --batchsize 8 --train_path "Dataset(small)/TrainingSet/MultiClassInfection-Train/" --test_path "Dataset(small)/TestingSet/MultiClassInfection-Test/"