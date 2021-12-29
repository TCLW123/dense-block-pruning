#python train.py --name 328 --gpus '1' --lr 0.1 --method 'supernet' --epochs 300 --bn_cal_batch_num 197 --pretrain ''
#python train_mp.py --name 332 --gpus '7' --epochs 300 --print_freq 10 --lr_arch 0.01 --lr 0.01
#python train_mp.py --name 347 -b 128 --lr 0.001 --lr_arch 0.001 --floss_type log_l1 --flop_loss_weight 0.1 --gpus '1' --print_freq 10
python train_mp.py --name 356 -b 256 --method 'sample' --sample_mode 'expected' --lr 0.0001 --epochs 300 --print_freq 100 --gpus '5' --mp_net '/home3/huxinyi/compression/experiments/densenet40_cifar10/345-mp-lr=0.01-arch_lr=0.01_2021-12-19-18:38:02/checkpoint.pth.tar'