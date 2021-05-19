python MyTrain_LungInf.py --train_save self-improved --is_data_augment True --graph_path graph_self-improved --device cuda --epoch 500 --batchsize 12 --is_data_augment True --random_cutout 0.4

# baseline
python MyTrain_LungInf.py --train_save baseline-inf-net --random_cutout 0 --graph_path graph_baseline-inf-net --device cuda --epoch 500 --batchsize 8
# baseline cross-val
python MyTrain_LungInf.py --train_save baseline-inf-net-cross-val --fold 5 --eval_threshold 0.5 --random_cutout 0 --seed 100 --graph_path graph_baseline-inf-net-cross-val --device cuda --epoch 500 --batchsize 8

# improved
python main_inf-net.py --save_path model/self-inf-net --graph_path graphs/self-inf-net --device cuda --seed 100 --batchsize 64
python MyTrain_LungInf.py --train_save self-inf-net --random_cutout 0 --graph_path graph_self-inf-net --seed 100 --load_net_path ../model/self-inf-net/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 500 --batchsize 8
python MyTrain_LungInf.py --train_save self-improved --focal_loss --lookahead --is_data_augment True --random_cutout 0.5 --seed 100 --graph_path graph_self-improved --load_net_path ../model/self-inf-net/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 500 --batchsize 8
#improved cross-val
python MyTrain_LungInf.py --train_save self-improved-cross-val --fold 5 --eval_threshold 0.5 --focal_loss --lookahead --is_data_augment True --random_cutout 0.5 --seed 100 --graph_path graph_self-improved-cross-val --load_net_path ../model/self-inf-net/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 500 --batchsize 8

# ablation studies
# self-supervised only
python MyTrain_LungInf.py --train_save self-inf-net --random_cutout 0 --graph_path graph_self-inf-net --seed 100 --load_net_path ../model/self-inf-net/medseg_resnet18_autoencoder_no_bottleneck_use_coach10.net.best.ckpt.t7 --device cuda --epoch 500 --batchsize 8
# focal loss only
python MyTrain_LungInf.py --train_save baseline-inf-net-focal --random_cutout 0 --focal_loss --seed 100 --graph_path graph_baseline-inf-net-focal --device cuda --epoch 500 --batchsize 8
# lookahead only
python MyTrain_LungInf.py --train_save baseline-inf-net-lookahead --random_cutout 0 --lookahead --seed 100 --graph_path graph_baseline-inf-net-lookahead --device cuda --epoch 500 --batchsize 8


# evaluation (use python rocs_generation.py to find the best eval_threshold)
python MyTrain_LungInf.py --is_eval True --eval_threshold 0.5 --load_net_path './Snapshots/save_weights/self-inf-net/{checkpoint_model_name}'
