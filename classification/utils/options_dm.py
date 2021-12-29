import argparse

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')

### slimmable cut
parser.add_argument('--name', default='329', type=str,
                    help='name of experiment')
parser.add_argument('--gpus', default='5', type=str,
                    help='gpus id')
parser.add_argument('--epochs', default=250, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='current learning rate')
parser.add_argument('--gammas', default=[0.1,0.1,0.1,0.1,0.1], type=float,
                    help='LR is multiplied by gamma on schedule, number of gammas '
                         'should be equal to schedule')
parser.add_argument('--schedule', default=[150,225,250,300,350], type=int,
                    help='Decrease learning rate at these epochs')
parser.add_argument('--manualSeed', default=None,
                    help='random seed')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', default=100, type=int,
                    help='')

### supernet
parser.add_argument('--width_range', default=[0.35,1],
                    help='Depth range of the layer')
parser.add_argument('--offset', default=0.025, type=float,
                    help='')
parser.add_argument('--num_sample', default=4,
                    help='slim network per epoch')
parser.add_argument('--supernet',
                    default='/home3/huxinyi/compression/experiments/densenet40_cifar10/345-mp-lr=0.01-arch_lr=0.01_2021-12-19-18:38:02/checkpoint.pth.tar'
                    # default='/home3/huxinyi/compression/checkpoints/densenet40_supernet_top1=94.21'
                    , type=str, help='path of supernet')
parser.add_argument('--mp_net',
                    default='../experiments/densenet40_cifar10'
                            '/368-sample-expected-lr=0.1_2021-12-23-13:02:31/checkpoint.pth.tar',
                    type=str,)

## markov process
parser.add_argument('--method', default='sample', type=str,
                    choices=['sample','mp'],help='gpus id')
parser.add_argument('--sample_mode', default='expected', type=str,
                    choices=['expected','direct'],help='')
parser.add_argument('--prob_type', default='sigmoid', type=str,
                    choices=['sigmoid','exp'],help='gpus id')
parser.add_argument('--floss_type', default='log_l1', type=str,
                    choices=['log_l1','l2','inverted_log_l1', 'l1'],help='')
parser.add_argument('--flop_loss_weight', default=0.1, type=float,
                    help='')
parser.add_argument('--target_flops', default=None, type=float,
                    help='')
parser.add_argument('--model_flops', default=None, type=float,
                    help='')
parser.add_argument('--prune_ratio', default=0.5, type=float,
                    help='')
parser.add_argument('--init_alpha', default=6, type=int,
                    help='')
parser.add_argument('--lr_arch', default=0.5, type=float,
                    help='')
parser.add_argument('--warm_up_step', default=0, type=int,
                    help='')
parser.add_argument('--train_freq', default=1, type=int,
                    help='')
parser.add_argument('--gammas_arch', default=[0.1,0.1,0.1,0.1,0.1], type=float,
                    help='LR is multiplied by gamma on schedule, number of gammas '
                         'should be equal to schedule')
parser.add_argument('--schedule_arch', default=[150,225,250,300,350], type=int,
                    help='Decrease learning rate at these epochs')

### densenet 40
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=1, type=float,
                    help='compression rate in transition stage (default: 0.5)')




parser.set_defaults(augment=True)

args = parser.parse_args()