import torch
import os
import shutil
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss

class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):
    """ label smooth """
    def forward(self, output, target):
        eps = 0
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def sample_model(args, model, save_path, log):
    if args.sample_mode == 'expected':
        sample_model_flops_per_layer, sample_model_flops = model.expected_sampling()
        print_log('\n* sampled flops={:6.4f}M\n'
                  'flops of layer {} '.format(sample_model_flops, sample_model_flops_per_layer), log)
        save_path = os.path.join(save_path, 'expected_ch')
        save_checkpoint({
            'epoch': -1,
            'state_dict': model.state_dict(),
            'prec1': 0,
        }, False, save_path=save_path)
        # set_model(model, save_path)
    elif args.sample_mode == 'direct':
        model.expected_sampling()
        save_path = os.path.join(save_path, 'sample')
        save_checkpoint({
            'epoch': -1,
            'state_dict': model.state_dict(),
            'prec1': 0,
        }, False, save_path=save_path)
        # set_model(model, save_path)

# def set_model(model, save_path):
#     ### 根据expected_ch设置每层宽度

def layer_flops_distribution(args, model):
    num_sample = args.num_model_sample

def flop_loss(args, model):
    input_size = 32
    loss_type = args.floss_type
    loss_weight = args.flop_loss_weight
    if args.model_flops is not None:
        args.target_flops = args.prune_ratio * args.model_flops
    target_flops = args.target_flops

    e_flops = model.expected_flops(input_size, input_size)

    if loss_type == 'l2':
        loss = torch.pow(e_flops - float(target_flops), 2)
    elif loss_type == 'inverted_log_l1':
        loss = -torch.log(1 / (torch.abs(e_flops - target_flops) + 1e-5))
    elif loss_type == 'log_l1':
        # piecewise log function
        ratio = 1.0
        if abs(e_flops.item() - target_flops) > 82:
            ratio = 0.1

        if e_flops < target_flops * 0.95:
            loss = torch.log(ratio * torch.abs(e_flops - target_flops))
        elif target_flops * 0.95 <= e_flops < target_flops:
            loss = e_flops * 0
        else:
            loss = torch.log(ratio * torch.abs(e_flops - (target_flops * 0.95)))
    elif loss_type == 'l1':
        loss = torch.abs(e_flops - float(target_flops))
    else:
        raise NotImplementedError

    return loss_weight * loss, e_flops

def conv_compute_flops(conv_layer, in_height, in_width, e_in_ch=None, e_out_ch=None):
    out_channel = conv_layer.out_channels
    in_channel = conv_layer.in_channels
    groups = conv_layer.groups

    if e_out_ch is not None:
        out_channel = e_out_ch
    if e_in_ch is not None:
        if groups == in_channel:
            groups = e_in_ch.detach().cpu().item()
        else:
            assert groups == 1, 'Unknown group number'
        in_channel = e_in_ch

    padding_height, padding_width = conv_layer.padding
    stride_height, stride_width = conv_layer.stride
    kernel_height, kernel_width = conv_layer.kernel_size
    assert conv_layer.dilation == (1, 1)  # not support deformable conv

    bias_ops = 1 if conv_layer.bias is not None else 0

    kernel_ops = kernel_height * kernel_width * (in_channel / groups)
    output_height = (in_height + padding_height * 2 - kernel_height) // stride_height + 1
    output_width = (in_width + padding_width * 2 - kernel_width) // stride_width + 1
    flops = (kernel_ops + bias_ops) * output_height * output_width * out_channel

    return flops, output_height, output_width

def fc_compute_flops(fc_layer):
    out_channel = fc_layer.out_features
    in_channel = fc_layer.in_features

    bias_ops = 1 if fc_layer.bias is not None else 0

    flops = (2 * in_channel + bias_ops) * out_channel

    return flops

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def change_model_keys(pretrain):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in pretrain['state_dict'].items():
        name = k
        for i in range(3):
            ori_name = 'dense' + str(i + 1)
            if ori_name in k:
                new_name = ori_name + '.layer'
                name = k.replace(ori_name, new_name)
        if 'bn.' in k:
            name = k.replace('bn.', 'bn1.')
        new_state_dict[name] = v
    return new_state_dict

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_path=None):
    """Saves checkpoint to disk"""
    # directory = "runs/%s/"%(args.name)
    directory = save_path
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + "/" + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + "/" + 'model_best.pth.tar')

def load_data_cifar10(args):
    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/home3/huxinyi/data/cifar10', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/home3/huxinyi/data/cifar10/', train=False, transform=transform_test),
        batch_size=100, shuffle=True, **kwargs)

    return train_loader, val_loader

def save_history_list(list, save_path=None, filename=None):
    directory = save_path
    if not os.path.exists(directory):
        os.makedirs(directory)
    a = np.array(list)
    if filename == None:
        filename = directory + "/history_list.npy"
    else:
        filename = directory + "/" + filename + ".npy"
    np.save(filename, a)  # 保存为.npy格式

def resume_model(resume_path, log, model, prec=None, strict=True, args=None):
    if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path)
            if prec == None:
                prec = 'best_prec1'
            best_prec1 = checkpoint[prec]
            model.load_state_dict(checkpoint['state_dict'], strict=strict)
            if args is not None:
                args.start_epoch = checkpoint['epoch']
            print_log("=> loaded checkpoint '{}' (epoch {}, acc {})"
                  .format(resume_path, checkpoint['epoch'], best_prec1), log)
            return model
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))