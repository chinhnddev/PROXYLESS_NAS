import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import ProxylessNAS
from utils import get_cifar10
from utils.utils import accuracy, AverageMeter, save_checkpoint, create_exp_dir, get_logger


def main():
    # Args
    class Args:
        def __init__(self):
            self.data = './data'
            self.batch_size = 128
            self.learning_rate = 0.025
            self.learning_rate_min = 0.001
            self.momentum = 0.9
            self.weight_decay = 3e-4
            self.report_freq = 50
            self.gpu = 0
            self.epochs = 600
            self.init_channels = 36
            self.layers = 20
            self.model_path = 'saved_models'
            self.cutout = True
            self.cutout_length = 16
            self.drop_path_prob = 0.2
            self.save = 'train-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), 'proxylessnas')
            self.seed = 2
            self.grad_clip = 5
            self.resume = None
            self.auxiliary = False
            self.auxiliary_weight = 0.4

    args = Args()

    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = ProxylessNAS(args.init_channels, 10, args.layers)
    model = model.cuda()

    logging = get_logger(os.path.join(args.save, 'log.txt'))
    logging.info('args = %s', args.__dict__)

    optimizer = optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_loader, valid_loader = get_cifar10(args.batch_size, cutout=args.cutout, cutout_length=args.cutout_length)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    create_exp_dir(args.save, scripts_to_save=['train_cifar.py'])

    writer = SummaryWriter(log_dir=os.path.join(args.save, 'runs'))

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(model, optimizer, args.resume)

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_acc, train_obj = train(train_loader, model, criterion, optimizer, epoch, args)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_loader, model, criterion, args)
        logging.info('valid_acc %f', valid_acc)

        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('valid/acc', valid_acc, epoch)
        writer.add_scalar('train/loss', train_obj, epoch)
        writer.add_scalar('valid/loss', valid_obj, epoch)

        if valid_acc > best_acc:
            best_acc = valid_acc
            save_checkpoint(model, optimizer, epoch, best_acc, os.path.join(args.save, 'best.pth.tar'))

        save_checkpoint(model, optimizer, epoch, best_acc, os.path.join(args.save, 'checkpoint.pth.tar'))

    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, args):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    for step, (input, target) in enumerate(train_loader):
        n = input.size(0)

        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            print('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_loader, model, criterion, args):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_loader):
            input = input.cuda()
            target = target.cuda()

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
