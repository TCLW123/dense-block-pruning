import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn as nn
import numpy as np

# from config import args
from pruner.prune_engine import *
from pruner.cut_ops import *
from models import *
from models import densenet_alpha as dnh
# from models import densenet_us as dnh
from models import densenet as dn
from models.resnet import *
from utils import *

# connection = [0]*36

model = dnh.densenet40_alpha()
# model = dn.DenseNet3(depth=40, num_classes=10)
# model = resnet50()
model.cuda()

model_path = '/home3/huxinyi/compression/checkpoints/densenet40_supernet_top1=94.21'
# best_list_path = '/home3/huxinyi/compression/experiments/densenet40_cifar10/308-supernet-lr=0.1-[35, 65, 95]-pretrain_2021-11-30-19:47:08/history_list.npy'
# IF_update_row_col = True

# if model_path != '':
#     if os.path.isfile(model_path):
#         pretrain = torch.load(model_path)
#         print("=> loading pretrain model '{}'".format(model_path))
#         new_state_dict = change_model_keys(pretrain)
#         model = nn.DataParallel(model)
#         model.load_state_dict(new_state_dict)
#         # if isinstance(model, torch.nn.DataParallel):
#         #     model = model.module
#     else:
#         print("Pretrain path: {}".format(model_path + ' does not exit'))

# cut_connection = np.load(best_list_path)

# cut_connection = [1]*6 + [5]*6 + [0]*6 + [5]*6 + [0]*6 + [5]*6
# model = mask_by_connection(model, cut_connection=cut_connection.tolist())

# if IF_update_row_col:
#     UpdatePrunedRatio(model, "Col")
# from thop import profile
# input = torch.randn(1,3,224,224)
# flops, params = profile(model, (input,))
# print("flops:{}, params:{}".format(flops,params))


total_flops, last_flops = check_flops(model)
Prune_rate_compute(model, verbose = True)
print("==================== Flops Compress Rate ========================")
print('Total number of flops: {:.2f}M'.format(total_flops / 1e6))
print('Last number of flops: {:.2f}M'.format(last_flops / 1e6))
print("Final model speedup rate: {:.2f}".format(total_flops / last_flops))
print("=================================================================")

