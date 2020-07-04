# check again maybe batchsize can be changed
python main_multi-inf-net.py --save_path model/self_multi_improved --graph_path graph_self_multi_improved --device cuda --seed 7

python MyTrain_MulClsLungInf_UNet.py --train_save self-multi-inf-net --is_data_augment False --random_cutout 0 --is_label_smooth False --graph_path graph_self-multi-inf-net --load_net_path ../model/self_multi_improved/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7--device cuda --epoch 500 --batchsize 12
python MyTrain_MulClsLungInf_UNet.py --train_save self-multi-inf-net --is_data_augment True --random_cutout 0.5 --is_label_smooth True --graph_path graph_self-multi-inf-net --load_net_path ../model/self_multi_improved/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7--device cuda --epoch 500 --batchsize 12

