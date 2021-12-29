import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import setproctitle
from utils.utils import *
from pruner.cut_ops import *

best_prec1 = 0
best_prec1_cut = 0
history_width = []

def main():
    global args, best_prec1, best_prec1_cut, history_width
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.name = args.name + "-lr=" + str(args.lr) + "-lr_cut=" + str(args.lr_cut) + \
                "-sn=" + str(args.num_subnet) + "_" + str(args.sub_epoch)
    if args.pretrain:
        args.name = args.name + "-pretrain"
    setproctitle.setproctitle(args.name)
    real_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    save_path = "../experiments/densenet40_cifar10/%s" % (args.name) + "_" + str(real_time)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    log = open(os.path.join(save_path + "/" + 'logger.log'), 'w')

    # Data loading code
    train_loader, val_loader = load_data_cifar10(args)

    # creat model
    connection = [0]*36
    model = dnh.densenet_40_rd(args.layers, 10, args.growth, args.reduce,
                               args.droprate, connection)
    # print_log(model, log)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # soft_criterion = CrossEntropyLossSoft()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    # use pretrained model
    if args.pretrain:
        if os.path.isfile(args.pretrain):
            pretrain = torch.load(args.pretrain)
            print_log("=> loading pretrain model '{}'".format(args.pretrain), log)
            new_state_dict = change_model_keys(pretrain)
            model = nn.DataParallel(model)
            model.load_state_dict(new_state_dict)
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
        else:
            print_log("Pretrain path: {}".format(args.pretrain + ' does not exit'), log)

    model = model.cuda()
    cudnn.benchmark = True
    testF = open(os.path.join(save_path, 'test.csv'), 'w')
    subnetF = open(os.path.join(save_path, 'train_subnet.csv'), 'w')
    selectF = open(os.path.join(save_path, 'val_subnet.csv'), 'w')

    # cut_connection = [0] * 6 + [4] * 6 + [0] * 6 + [4] * 6 + [0] * 6 + [4] * 6
    #
    # model = update_width(model, cut_connection)
    # prec1 = validate(val_loader, model, criterion, epoch=-1, log=log, testF=None)
    # print_log(" accu width is: %.3f %%" % prec1, log)
    #
    # # model = update_width(model, connection)
    # model = mask_by_connection(model, cut_connection=cut_connection)
    # prec1 = validate(val_loader, model, criterion, epoch=-1, log=log, testF=None)
    # print_log(" accu mask is: %.3f %%" % prec1, log)

    net_connection_list = get_net(model)

    ### sample random width
    for epoch in range(args.num_subnet):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas_cut,
                                                     args.schedule_cut, args.lr_cut)

        ### sample subnet
        cut_connection_list, cut_channels = sample_subnet(net_connection_list,
                                                          args.target_ch)
        history_width.append(cut_connection_list)

        model = update_width(model, cut_connection_list)
        # prec1 = validate(val_loader, model, criterion, epoch=-1, log=log, testF=None)
        # print_log(" accu before is: %.3f %%" % prec1, log)

        ### train subnet
        for sub_epoch in range(args.sub_epoch):
            train(train_loader, model, criterion, optimizer, epoch, sub_epoch, log)

        prec1 = validate(val_loader, model, criterion, epoch=-1, log=log, testF=subnetF)
        # test prec@1 and save checkpoint
        print_log(
            '\n* [Sample] [SubNet={:03d}/{:03d}] [learning_rate={:6.4f}] [cut_channels={:04d}] [top1={:6.4f}]'.format(
                epoch+1, args.num_subnet, current_learning_rate, int(cut_channels), prec1), log)
    print_log("sample finished!", log)

    ### save subnet
    save_history_list(history_width, save_path=save_path)

    ### select best subnet
    for i in range(args.num_subnet):
        # cut_connection_list = history_width[i]

        ### sample subnet
        cut_connection_list, cut_channels = sample_subnet(net_connection_list, args.target_ch)
        history_width.append(cut_connection_list)

        update_width(model, cut_connection_list)

        prec1 = validate(val_loader, model, criterion, epoch=-1, log=log, testF=selectF)

        # eval subnet choose the best one
        if prec1 > best_prec1:
            best_list = cut_connection_list
        best_prec1 = max(prec1, best_prec1)

        print_log(
            '\n* [select best subnet] [SubNet={:03d}/{:03d}] [cut_channels={:04d}] [top1={:6.4f}]'.format(
                epoch + 1, args.num_subnet, int(cut_channels) ,prec1) \
                + ' [best : top1={:.3f}]'.format(best_prec1), log)
    print_log("select finished!", log)
    save_history_list(best_list, save_path=save_path, filnename="best")

    ### retrain
    update_width(model, best_list)
    best_prec1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas,
                                                     args.schedule, args.lr)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, sub_epoch=None, log=log)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, log, testF=testF)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        print_log(
            '\n* [Retrain] [Epoch={:03d}/{:03d}] [learning_rate={:6.4f}] [top1={:6.4f}]'.format(
                epoch, args.epochs, current_learning_rate, prec1) \
            + ' [best : top1={:.3f}]'.format(best_prec1), log)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, save_path=save_path)
    print_log("retrain finished!", log)

def train(train_loader, model, criterion, optimizer, epoch, sub_epoch=None, log=None):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if sub_epoch is not None:
                print_log('[Sample] SubNet: [{0}]\t'
                      'Epoch: [{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          epoch+1, sub_epoch,i, len(train_loader), batch_time=batch_time,
                          loss=losses, top1=top1), log)
            else:
                print_log('[Retrain]\t'
                          'Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=losses, top1=top1), log)

def validate(val_loader, model, criterion, epoch, log, testF=None):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1), log)
    if testF is not None:
        testF.write('{},{:.4f},{:.3f}\n'.format(epoch, losses.avg, top1.avg))
        testF.flush()

    return top1.avg

def adjust_learning_rate(optimizer, epoch, gammas, schedule, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()
