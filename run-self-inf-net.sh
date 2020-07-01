python MyTrain_LungInf.py --train_save self-improved --is_data_augment True --graph_path graph_self-improved --device cuda --epoch 500 --batchsize 12 --is_data_augment True --random_cutout 0.4

python main_inf-net.py --save_path model/self-inf-net --graph_path graphs/self-inf-net --device cuda --seed 7 --batchsize 64
python MyTrain_LungInf.py --train_save self-improved --is_data_augment True --graph_path graph_self-improved --load_net_path ../model/self-inf-net/medseg_resnet18_autoencoder_no_bottleneck_use_coach0.net.best.ckpt.t7 --device cuda --epoch 500 --batchsize 12