import argparse

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')

### slimmable cut
parser.add_argument('--name', default='324', type=str,
                    help='name of experiment')
parser.add_argument('--gpus', default='5', type=str,
                    help='gpus id')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='current learning rate')

parser.add_argument('--method', default='supernet', type=str,
                    choices=['cut-inplace','distill','supernet','cut','train','test','cal_bn'],
                    help='method of experiment')
### supernet
parser.add_argument('--width_range', default=[0.35,1],
                    help='Depth range of the layer')
parser.add_argument('--offset', default=0.025, type=float,
                    help='')
parser.add_argument('--num_sample', default=4,
                    help='slim network per epoch')
parser.add_argument('--bn_cal_batch_num', default=100, type=int,
                    help='')

parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (  : 10)')
parser.add_argument('--pretrain', default='/home3/huxinyi/compression/experiments/densenet40_cifar10/325/checkpoint.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gammas', default=[0.1,0.1,0.1], type=float,
                    help='LR is multiplied by gamma on schedule, number of gammas '
                         'should be equal to schedule')
parser.add_argument('--schedule', default=[150,225,250], type=int,
                    help='Decrease learning rate at these epochs')

parser.add_argument('--epochs', default=250, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 64)')


parser.add_argument('--manualSeed', default=None,
                    help='random seed')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')


parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=1, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')


### random width
parser.add_argument('--num_subnet', default=50, type=int,
                    help='number of subnet')
parser.add_argument('--sub_epoch', default=3, type=int)
parser.add_argument('--lr_cut', default=0.1, type=float,
                    help='current learning rate')
parser.add_argument('--gammas_cut', default=[0.1,0.1,0.1], type=float,
                    help='LR is multiplied by gamma on schedule, number of gammas '
                         'should be equal to schedule')
parser.add_argument('--schedule_cut', default=[150,225,250], type=int,
                    help='Decrease learning rate at these epochs')
parser.add_argument('--target_ch', default=[348, 1500, 2500], type=int,
                    help='target')
parser.add_argument('--width', default=0.5, type=float,
                    help='Depth of the layer')
parser.add_argument('--width_list', default=[],
                    help='Depth range list of the layer')

parser.set_defaults(augment=True)

args = parser.parse_args()