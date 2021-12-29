import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import setproctitle
from models.densenet import *
from models.densenet_us import densenet40_us
import random
from pruner.cut_ops import *
from utils.utils import *
from utils.options import args

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

best_prec1 = 0
best_prec1_cut = 0

def main():
    global args, best_prec1, best_prec1_cut

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    setup_seed(args.manualSeed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    args.name = args.name + "-" + args.method
    if args.method in ['cut', 'distill', 'cut-inplace']:
        args.name = args.name + "-width=" + str(args.width)
    if args.method == 'cal_bn':
        args.name = args.name + "-" + str(args.bn_cal_batch_num)
    else:
        args.name = args.name + "-" + str(args.lr) + "-" +str(args.schedule)
        if args.pretrain:
            args.name = args.name + "-pretrain"
    setproctitle.setproctitle(args.name)
    real_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    save_path = "../experiments/densenet40_cifar10/%s"%(args.name)+ "_" + str(real_time)

    if args.tensorboard:
        configure(save_path)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    log = open(os.path.join(save_path+ "/" + 'logger.log'), 'w')

    # Data loading code
    train_loader, val_loader = load_data_cifar10(args)

    # create model
    if args.method == 'train':
        model = DenseNet3()
    else:
        model = densenet40_us(args.layers, 10, args.growth, args.reduce, args.droprate)

    # print_log(model,log)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.method in ['cut-inplace', 'supernet', 'distill', 'cal_bn']:
        soft_criterion = CrossEntropyLossSoft()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    # use pretrained model
    if args.pretrain:
        if os.path.isfile(args.pretrain):
            pretrain = torch.load(args.pretrain)
            #load pretrain
            # new_state_dict = change_model_keys(pretrain)

            ### load resume
            new_state_dict = pretrain['state_dict']
            # model = torch.nn.DataParallel(model)

            model.load_state_dict(new_state_dict)
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            print_log("=> loading pretrain model '{}'".format(args.pretrain), log)
        else:
            print_log("Pretrain path: {}".format(args.pretrain + ' does not exit'), log)

    # get the number of model parameters
    print_log('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])),log)
    print_log('reduce: {}, lr: {}'.format(args.reduce, args.lr),log=log)

    model = model.cuda()
    prec1 = validate(val_loader, model, criterion, epoch=-1, log=log, testF=None)
    print_log('\n* pretrain model top1={:6.4f} '.format(prec1), log)

    # optionally resume from a checkpoint
    if args.resume:
        model = resume_model(args.resume,log=log,model=model)

    cudnn.benchmark = True
    testF = open(os.path.join(save_path, 'test.csv'), 'w')

    if args.method == 'cut-inplace':
        testF_cut = open(os.path.join(save_path, 'test_cut.csv'), 'w')
    elif args.method in ['cut', 'distill']:
        if args.method == 'distill':
            model_t = copy.deepcopy(model)
            # prec1_t = validate(val_loader, model_t, criterion, epoch=-1, log=log, testF=None)
            # print_log(" accu of before is: %.3f %%" % prec1_t, log)
        model.apply(lambda m: setattr(m, 'width_mult', args.width))
        # prec1 = validate(val_loader, model, criterion, epoch=-1, log=log, testF=None)
        # print_log(" accu after is: %.3f %%" % prec1, log)
    elif args.method == 'supernet' or args.method == 'cal_bn':
        history_width_list = []
        width_list = np.round(np.arange(args.width_range[0], args.width_range[1]+0.01, args.offset), 3).tolist()
        width_list.reverse()
        args.width_list = width_list

    for epoch in range(args.start_epoch, args.epochs):
        current_lr = adjust_learning_rate(optimizer, epoch, args.gammas,
                                          args.schedule, args.lr)

        if args.method in ['cut', 'train', 'distill']:
            if args.method in ['cut', 'train']:
                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch, log)
            else:
                train_distill(train_loader, model_t, model, criterion, optimizer,
                                       epoch, log, soft_criterion=soft_criterion)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, epoch, log, testF)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            print_log(
                '\n* [Epoch={:03d}/{:03d}] [learning_rate={:6.4f}] [top1={:6.4f}]'.format(
                    epoch, args.epochs, args.lr, prec1) \
                + ' [best : top1={:.3f}]'.format(best_prec1), log)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, save_path=save_path)
        # print_log('Best accuracy: {}'.format(best_prec1), log)

        elif args.method == 'cut-inplace':
            # 返回train时用到的width
            width_list = train_inplace(train_loader, model, criterion, optimizer,
                                       epoch, log, soft_criterion=soft_criterion)
            # 验证每个width的acc
            for width_mult in width_list:
                model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                if width_mult == 1:
                    prec1 = validate(val_loader, model, criterion, epoch, log, testF=testF)
                    # is_best = prec1 > best_prec1
                    best_prec1 = max(prec1, best_prec1)
                else:
                    prec1_cut = validate(val_loader, model, criterion, epoch, log, testF=testF_cut)
                    is_best_cut = prec1_cut > best_prec1_cut
                    best_prec1_cut = max(prec1_cut, best_prec1_cut)

            print_log(
                '\n* [Epoch={:03d}/{:03d}] [learning_rate={:6.4f}] [top1={:6.4f}] [top1_cut={:6.4f}]'.format(
                    epoch, args.epochs, args.lr, prec1, prec1_cut) \
                + ' [best: top1={:.3f}, top1_cut={:.3f}]'.format(best_prec1, best_prec1_cut), log)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1_cut,
            }, is_best_cut, save_path=save_path)
        elif args.method == 'supernet':
            width_list = train_supernet(train_loader, model, criterion, optimizer,
                                       epoch, log, soft_criterion=soft_criterion)
            # history_width_list.append(width_list)

            if epoch % 10 == 0:
                model_cal_bn = copy.deepcopy(model)

                train_bn(train_loader,model_cal_bn, width=1, batch_num=None)

                prec1 = validate(val_loader, model_cal_bn, criterion, epoch, log, testF=testF)

                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)

                print_log(
                    '\n* [supernet] [Epoch={:03d}/{:03d}] [learning_rate={:6.4f}] [top1={:6.4f}] [batch_num={}] '.format(
                        epoch, args.epochs, current_lr, prec1, args.bn_cal_batch_num) \
                    + ' [best: top1={:.3f}]'.format(best_prec1), log)

                # save_history_list(history_width_list, save_path=save_path)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, save_path=save_path)
        else:
            break

    print_log("train finished!!!", log=log)
    ### calibrate bn
    if args.method == 'supernet' or args.method == 'cal_bn':
        model.apply(lambda m: setattr(m, 'width_mult', 1))
        prec1 = validate(val_loader, model, criterion, epoch, log, testF=None)
        print_log('\n* [cal_bn before] [top1={:6.4f}] '.format(prec1), log)

        train_bn(train_loader, model, width=1, batch_num=None)

        prec1 = validate(val_loader, model, criterion, epoch, log, testF=None)
        print_log('\n* width={} top1={:6.4f} bn_update '.format(1, prec1), log)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'prec1': prec1,
        }, False, save_path=save_path, filename='densenet40_supernet_top1=' + str(prec1))

        if args.method == 'supernet':
            best_model = save_path + "/" + 'checkpoint.pth.tar'
            model = resume_model(resume_path=best_model, log=log, model=model)

        model.apply(lambda m: setattr(m, 'width_mult', 1))
        prec1 = validate(val_loader, model, criterion, epoch, log, testF=None)
        print_log('\n* [cal_bn before] [top1={:6.4f}] '.format(prec1), log)

        train_bn(train_loader, model, width=1, batch_num=None)

        prec1 = validate(val_loader, model, criterion, epoch, log, testF=None)
        print_log('\n* width={} top1={:6.4f} bn_update '.format(1, prec1), log)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'prec1': prec1,
        }, False, save_path=save_path, filename='densenet40_supernet_best_top1='+str(prec1))

        print_log("bn test finished!!!", log=log)

def train_bn(train_loader, model, width = None ,batch_num=None):
    # switch to eval mode
    model.eval()

    if batch_num is None:
        batch_num = args.bn_cal_batch_num
    ### calibrate bn
    # momentum_bk = None
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.training = True
            # if momentum_bk is None:
            #     momentum_bk = m.momentum
            # m.momentum = 1.0

    for i, (input, target) in enumerate(train_loader):
        if i == batch_num:
            break

        if width is not None:
            model.apply(lambda m: setattr(m, 'width_mult', width))
            with torch.no_grad():
                input = input.cuda()
                model(input)
        else:
            for width in args.width_list:
                model.apply(lambda m: setattr(m, 'width_mult', width))
                with torch.no_grad():
                    input = input.cuda()
                    model(input)

    ### calibrate bn
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # m.momentum = momentum_bk
            m.training = False

def train_supernet(train_loader, model, criterion, optimizer, epoch, log, soft_criterion=None):
    """run one epoch for supernet"""
    batch_time = AverageMeter()

    # switch to train mode
    model.train()

    # sample width
    max_width = args.width_range[1]
    min_width = args.width_range[0]

    widths_train = []
    for _ in range(args.num_sample - 2):
        widths_train.append(random.choice(args.width_list))
        # widths_train.append(random.uniform(min_width, max_width))
    widths_train = [max_width, min_width] + widths_train

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        soft_target = None

        end = time.time()

        for width_mult in widths_train:
            ### the sandwitch ruls
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            # compute output
            end1 = time.time()
            output = model(input_var)
            # inplace distill
            end2 = time.time()
            print("forward:", end2 - end1)
            if width_mult == max_width:
                loss = criterion(output, target_var)
                soft_target = torch.nn.functional.softmax(output, dim=1)
            else:
                if soft_target is not None:
                    loss = torch.mean(soft_criterion(output, soft_target.detach()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("backward:", time.time() - end2)

            # measure elapsed time
            batch_time.update(time.time() - end)


            # if i % args.print_freq == 0:
            #     print_log('Epoch: [{0}][{1}/{2}]\t'
            #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #               'Loss {loss:.4f}\t'
            #               'Width {width:.3f}'.format(
            #         epoch, i, len(train_loader), batch_time=batch_time,
            #         loss=loss, width=width_mult), log)
        print("supernet one batch (4 model) time:", time.time() - end)
    model.apply(lambda m: setattr(m, 'width_mult', 1))
    return widths_train

def train_distill(train_loader, model_t, model, criterion, optimizer, epoch, log, soft_criterion=None):
    """run one epoch for inplace distillation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    model_t.eval()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        output_t = model_t(input_var)
        soft_target = torch.nn.functional.softmax(output_t, dim=1)
        loss = torch.mean(soft_criterion(output, soft_target.detach()))

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

def train_inplace(train_loader, model, criterion, optimizer, epoch, log, soft_criterion=None):
    """run one epoch for inplace distillation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    losses_cut = AverageMeter()
    top1_cut = AverageMeter()

    # switch to train mode
    model.train()

    max_width = args.width_range[1]
    min_width = args.width_range[0]
    ### set width
    widths_list = [1]
    widths_list.append(args.width)

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        soft_target = None

        for width_mult in widths_list:
            # sandwitch rule
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))

            # compute output
            output = model(input_var) ### 看一下output形式 --debug hxy

            # inplace distillation
            if width_mult == max_width:
                loss = criterion(output, target_var)
                soft_target = torch.nn.functional.softmax(output, dim=1)
                losses.update(loss.item(), input.size(0))
                prec1 = accuracy(output.data, target, topk=(1,))[0]
                top1.update(prec1.item(), input.size(0))
            else:
                if soft_target is not None:
                    loss = torch.mean(soft_criterion(output, soft_target.detach()))
                losses_cut.update(loss.item(), input.size(0))
                prec1 = accuracy(output.data, target, topk=(1,))[0]
                top1_cut.update(prec1.item(), input.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss_cut.val:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1_cut.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, loss_cut=losses_cut, top1=top1, top1_cut=top1_cut), log)
    return widths_list

def train(train_loader, model, criterion, optimizer, epoch, log):
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

    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)

def validate(val_loader, model, criterion, epoch, log, testF):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            # # if i % 100 == 0:
            #     print_log('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #               i, len(val_loader), batch_time=batch_time, loss=losses,
            #               top1=top1), log)
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

def setup_seed(seed):
    torch.manual_seed(seed) #为gpu设置随机种子
    torch.cuda.manual_seed_all(seed) #为所有GPU设置随机种子
    np.random.seed(seed) #为python numpy随机数生成器设置种子
    random.seed(seed)
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()
