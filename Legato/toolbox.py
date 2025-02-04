import os
import torch
from pathlib import Path


def setup_seed(seed):
    import numpy as np
    import random
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100 / batch_size))
    return res


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = "epoch=" + str(epoch) + "\tglobal_step=" + str(global_step)
    for key, value in kwargs.items():
        if type(value) == str:
            display = "\t" + key + "=" + value
        else:
            display += "\t" + str(key) + "=%.4f" % value
    display += "\ttime=%.2fit/s" % (1.0 / time_elapse)
    return display


def first_rank(feature, y):
    ranklist = torch.argsort(feature, descending=True)
    index = torch.where( y == 1)[0]
    ranks = []
    for k in index:
        rank = torch.where(ranklist == k)[0] + 1
        ranks.append(rank)
    # ranks = torch.tensor(ranks)
    return min(ranks)


def average_rank(feature, y):
    ranklist = torch.argsort(feature, descending=True)
    index = torch.where( y == 1)[0]
    ranks = []
    for k in index:
        rank = torch.where(ranklist == k)[0] + 1
        ranks.append(int(rank))
    # ranks = torch.tensor(ranks)
    return sum(ranks)/len(ranks)


def ACC(feature, y, topk):
    firstrank = first_rank(feature, y)
    acc = []
    for i in topk:
        if firstrank <= i:
            acc.append(1)
        else:
            acc.append(0)
    return acc


def multi_ACC(feature, y, topk):
    ranklist = torch.argsort(feature, descending=True)
    index = torch.where( y == 1 )[0]
    acc = []
    for i in topk:
        flag = True
        for j in range(i):
            rank = ranklist[j]
            for k in index:
                if rank == k:
                    acc.append(1)
                    flag = False
            if not flag:
                break
        if flag:
            acc.append(0)
    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
