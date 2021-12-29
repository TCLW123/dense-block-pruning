import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from utils.options_dm import args
import time
import random
import setproctitle
from utils.utils import * #CrossEntropyLossSoft, CrossEntropyLossSmooth
from models.densenet_alpha import *
from models.densenet_us import *
from pruner.prune_engine import *

def main():
    global args, best_prec1
    best_prec1 = 0
    ## set random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    setup_seed(args.manualSeed)

    ## set gpu num
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    ## set name
    if args.method == 'sample':
        args.name = args.name + '-' + args.method + '-' + args.sample_mode + '-lr=' + str(args.lr)
    else:
        args.name = args.name + '-mp-lr=' + str(args.lr) + '-arch_lr=' + str(args.lr_arch)
    setproctitle.setproctitle(args.name)
    real_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    save_path = "../experiments/densenet40_cifar10/%s" % (args.name) + "_" + str(real_time)

    # create log
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    log = open(os.path.join(save_path+ "/" + 'logger.log'), 'w')

    # print args
    print_log('\n* mp args: \n'
              '* lr={}, gammas={}, schedule={}, manualSeed={}, prob_type={} \n'
              '* floss_type={}, flop_loss_weight={}, target_flops={}, prune_ratio={}, lr_arch={} \n'
              '* warm_up_step={}, gammas_arch={}, schedule_arch={} '.format(
        args.lr, args.gammas, args.schedule, args.manualSeed, args.prob_type,
        args.floss_type, args.flop_loss_weight, args.target_flops, args.prune_ratio, args.lr_arch,
        args.warm_up_step, args.gammas_arch, args.schedule_arch), log)

    # Data loading code
    train_loader, val_loader = load_data_cifar10(args)

    # create model
    model = densenet40_alpha(args.layers, 10, args.growth, args.reduce, args.droprate)
    print("Alpha")
    # model = densenet40_us(args.layers, 10, args.growth, args.reduce, args.droprate)
    # print("US")

    criterion = nn.CrossEntropyLoss().cuda()
    soft_criterion = CrossEntropyLossSoft()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    optimizer_arch = torch.optim.SGD(model.parameters(), args.lr_arch,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    # load supernet
    if args.method == 'mp':
        if args.supernet:
            model = resume_model(args.supernet,log=log,model=model, prec='prec1', strict=False, args=args)
    elif args.method == 'sample':
        if args.mp_net:
            model = resume_model(args.mp_net,log=log, prec='best_prec1', model=model, strict=True, args=args)
            model.set_alpha_training(True)

    model = model.cuda()
    # prec1 = validate(val_loader, model, criterion)
    # print_log('\n* supernet top1={:6.4f} '.format(prec1), log)
    args.model_flops, flops_of_layer = model.cal_flops()
    print_log('\n* supernet flops={:6.4f}M\n'
              'flops of layer {} '.format(args.model_flops, flops_of_layer), log)

    testF = open(os.path.join(save_path, 'test.csv'), 'w')

    ### init width_list
    width_list = np.round(np.arange(args.width_range[0], args.width_range[1] + 0.01, args.offset), 3).tolist()
    width_list.reverse()
    args.width_list = width_list

    ### fix weight update alpha
    if args.method == "mp":
        # get model FLOPs
        args.model_flops, flops_of_layer = model.cal_flops()
        # print_log('\n* supernet flops={:6.4f}M\n'
        #           'flops of layer {} '.format(args.model_flops, flops_of_layer), log)

        for epoch in range(args.start_epoch, args.epochs):
            current_lr = adjust_learning_rate(optimizer, epoch, args.gammas,
                                              args.schedule, args.lr)
            arch_lr = adjust_learning_rate(optimizer_arch, epoch, args.gammas_arch,
                                              args.schedule_arch, args.lr_arch)
            eflops = train_mp(train_loader, model, criterion, soft_criterion,
                     optimizer, optimizer_arch,epoch, log)

            prec1 = validate(val_loader, model, criterion, epoch, testF=testF)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            print_log(
                '\n* [MP] [Epoch={:03d}/{:03d}] [lr={:.6f}] [lr_arch={:.6f}] [eflops={:.2f}M/{:.2f}M] [top1={:.3f}]'
                .format(epoch, args.epochs, current_lr, arch_lr, eflops, args.target_flops, prec1) \
                + ' [best: top1={:.3f}]'.format(best_prec1), log)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'prec1': prec1,
            }, is_best, save_path=save_path)

        print_log("Update alpha finished!!!", log=log)
    elif args.method == "sample":
        ## direct sample
        print_log("sampling model...", log)
        sample_model(args, model, save_path, log)

        model.set_alpha_training(False)

        prec1 = validate(val_loader, model, criterion)
        print_log('\n* expected sampling model top1={:6.4f} '.format(prec1), log)

        best_prec1 = prec1

        for epoch in range(args.start_epoch, args.epochs):
            current_lr = adjust_learning_rate(optimizer, epoch, gammas=args.gammas,
                                              schedule=args.schedule, lr=args.lr)
            retrain(train_loader, model, criterion, optimizer, epoch, log)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, epoch, testF)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            print_log(
                '\n* [Retrain] [Epoch={:03d}/{:03d}] [learning_rate={:6.4f}] [top1={:6.4f}]'.format(
                    epoch, args.epochs, current_lr, prec1) \
                + ' [best : top1={:.3f}]'.format(best_prec1), log)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, save_path=save_path)

        print_log("retrain finished!!!", log=log)

        # print_log("draw layer flops distribution...", log)
        # layer_flops_distribution(args, model)

def retrain(train_loader, model, criterion, optimizer, epoch, log):
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
            print_log('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1), log)

def train_mp(train_loader, model, criterion, soft_criterion, optimizer, optimizer_arch, epoch, log):
    batch_time = AverageMeter()
    losses_arch = AverageMeter()
    flosses = AverageMeter()
    eflopses = AverageMeter()

    model.train()

    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.cuda()
        target = target.cuda()

        # warm up by sandwitch rule
        model.set_alpha_training(False)
        train_one_batch_supernet(model, input, target, optimizer,
                                                    criterion, soft_criterion)

        # train architecture params
        if i >= args.warm_up_step and i % args.train_freq == 0:
            model.set_alpha_training(True)
            arch_loss, floss , eflops = train_one_batch_mp(model, input,
                                        target, criterion, optimizer_arch)

            losses_arch.update(arch_loss.item(), input.size(0))
            flosses.update(floss.item(), input.size(0))
            eflopses.update(eflops.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)


        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f}\t'
                  'arch_loss: {arch_loss.val:.4f}\t'
                  'floss: {floss.avg:.4f}\t'
                  'eflops: {eflops:.3f}M'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      arch_loss=losses_arch, floss=flosses, eflops=eflopses.val), log)
    return eflopses.val

def train_one_batch_mp(model, input, target, criterion, optimizer_arch):
    model.apply(lambda m: setattr(m, 'width_mult', 1))
    arch_out = model(input)
    arch_loss = criterion(arch_out, target)
    floss, eflops = flop_loss(args, model)
    loss = arch_loss + floss

    optimizer_arch.zero_grad()
    loss.backward()
    optimizer_arch.step()

    return arch_loss, floss, eflops

def train_one_batch_supernet(model, input, target, optimizer, criterion, soft_criterion):
    ### sample width
    widths_train = []
    for _ in range(args.num_sample - 2):
        widths_train.append(random.choice(args.width_list))
    widths_train = [args.width_range[1], args.width_range[0]] + widths_train

    soft_target = None
    for width_mult in widths_train:
        model.apply(lambda m: setattr(m, 'width_mult', width_mult))
        output = model(input)
        if width_mult == args.width_range[1]:
            loss = criterion(output, target)
            soft_target = torch.nn.functional.softmax(output, dim=1)
        else:
            if soft_target is not None:
                loss = torch.mean(soft_criterion(output, soft_target.detach()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(val_loader, model, criterion, epoch=None, testF=None):
    """Perform validation on the validation set"""
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            top1.update(prec1.item(), input.size(0))

    if testF is not None:
        testF.write('{},{:.4f},{:.3f}\n'.format(epoch, loss, top1.avg))
        testF.flush()

    return top1.avg

def setup_seed(seed):
    torch.manual_seed(seed) #为gpu设置随机种子
    torch.cuda.manual_seed_all(seed) #为所有GPU设置随机种子
    np.random.seed(seed) #为python numpy随机数生成器设置种子
    random.seed(seed)
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True

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
