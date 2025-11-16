import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import ProxylessNASNetwork
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
            self.epochs = 50
            self.init_channels = 36
            self.layers = 20
            self.model_path = 'saved_models'
            self.cutout = False
            self.cutout_length = 16
            self.drop_path_prob = 0.2
            self.save = 'search-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), 'proxylessnas')
            self.seed = 2
            self.grad_clip = 5
            self.train_portion = 0.5
            self.unrolled = False
            self.arch_learning_rate = 3e-4
            self.arch_weight_decay = 1e-3

    args = Args()

    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = ProxylessNASNetwork(args.init_channels, 10, args.layers, criterion)
    model = model.cuda()

    logging = get_logger(os.path.join(args.save, 'log.txt'))
    logging.info('args = %s', args.__dict__)

    optimizer = optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    optimizer_a = optim.Adam(model.arch_parameters(),
                            lr=args.arch_learning_rate,
                            betas=(0.5, 0.999),
                            weight_decay=args.arch_weight_decay)

    train_loader, valid_loader = get_cifar10(args.batch_size, cutout=args.cutout, cutout_length=args.cutout_length)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    create_exp_dir(args.save, scripts_to_save=['search_cifar.py'])

    writer = SummaryWriter(log_dir=os.path.join(args.save, 'runs'))

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # training
        train_acc, train_obj = train(train_loader, valid_loader, model, architect, criterion, optimizer, optimizer_a, lr, epoch, args, writer)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_loader, model, criterion, args)
        logging.info('valid_acc %f', valid_acc)

        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('valid/acc', valid_acc, epoch)
        writer.add_scalar('train/loss', train_obj, epoch)
        writer.add_scalar('valid/loss', valid_obj, epoch)

        save_checkpoint(model, optimizer, epoch, valid_acc, os.path.join(args.save, 'checkpoint.pth.tar'))

    writer.close()


def train(train_loader, valid_loader, model, architect, criterion, optimizer, optimizer_a, lr, epoch, args, writer):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for step, (input, target) in enumerate(train_loader):
        model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda()

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_loader))
        input_search = input_search.cuda()
        target_search = target_search.cuda()

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

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


class Architect(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = optim.SGD(
            self.model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)

        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]

        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


import numpy as np


if __name__ == '__main__':
    main()
