import numpy as np
import torch.nn as nn
from models import densenet_alpha as dnh
# from models import densenet_us as dnh

def get_subnet_random(connection_list, target_ch, block):
    # input_ch = (144 * (block - 1) +24)
    ### generate random subnet
    # if block == 1:
    #     first = np.random.randint(10,70,size=[6])
    #     last = np.random.randint(20,60,size=[6])
    # elif block ==2:
    #     first = np.random.randint(10, 60, size=[6])
    #     last = np.random.randint(10, 60, size=[6])
    # elif block ==3:
    #     first = np.random.randint(10, 60, size=[6])
    #     last = np.random.randint(10, 50, size=[6])

    # subnet_ratio = 0.01 * np.concatenate([first, last])

    subnet_ratio = 0.01 * np.random.randint(10, 90, size=[12])

    connection_list = connection_list[(block-1)*12 + block : block*12 + block]

    cut_connection = np.floor(connection_list * subnet_ratio)

    return cut_connection

def get_channels(block, cut_connection):
    input_ch = (144 * (block - 1) + 24)
    a = np.clip(cut_connection, 0, 1) * input_ch
    b = np.clip((cut_connection - 1), 0, 10000) * 12
    cut_channels = a.sum() + b.sum()

    return int(cut_channels)

def get_net(model):
    width_list = []

    for k,item in enumerate(model.modules()):
        if isinstance(item, nn.Conv2d):
            width_list.append(item.in_channels)

    ### get current_connection
    connection_list = get_connection(width_list)
    connection_list_np = np.array(connection_list)

    return connection_list_np

def get_connection(list):
    block = []
    for i in range(3):
        list1 = list[i*12+1+i: (i+1)*12+1+i]
        connection_list = (np.array(list1) - 24 - i*144) // 12
        block = block + [0] + connection_list.tolist()
    return block

def sample_subnet(connection_list, target_ch):
    cut_connection_list = []
    cut_channels = 0

    for block in range(3):
        block = block + 1
        subnet_connection_list = get_subnet_random(connection_list, target_ch, block)
        cut_connection_list = cut_connection_list + subnet_connection_list.tolist()

        cut_channels_block = get_channels(block, subnet_connection_list)
        cut_channels += cut_channels_block

    return cut_connection_list, cut_channels

def mask_by_connection(model, cut_connection):
    index = 0
    for i, item in enumerate(model.modules()):
        if isinstance(item, dnh.BasicBlock_rd):
            index += 1
            block = 1 + (index - 1) // 12
            item.cut_connection = 0
            cut_channels = get_channels(block, cut_connection[index-1])
            item.conv1.weight.data[:,:cut_channels] = 0
    return model

def bn_calibration_init(m, switch=False):
    """ calculating post-statistics of batch normalization """
    # reset all values for post-statistics
    if isinstance(m,nn.BatchNorm2d):
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = switch
        # # if use cumulative moving average
        change_momentum = False
        if change_momentum:
            m.momentum = None

def update_width(model, connection_list):
    ### update cut_connection_list of model
    index = 0
    for i, item in enumerate(model.modules()):
        if isinstance(item, dnh.BasicBlock_rd):
            item.cut_connection = connection_list[index]
            index += 1
    return model
