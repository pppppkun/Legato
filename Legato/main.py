import torch
import argparse
import math
import json
import sys as _sys
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from pathlib import Path
from datetime import datetime
from toolbox import setup_seed
from dataset import SupervisedDataset, ClassificationDataset
from model import GraphConvClassifier, Classifier
from trainer import SemiSupervisedTrainer, ClassifierTrainer
from trainer import (SemiSupervisedTrainer)
from evaluator import Evaluator, ClassifierEvaluator
from tqdm import tqdm

dataset_path = Path('')

def parse_option():
    parser = argparse.ArgumentParser("argument for training")
    # training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--distributed", action='store_true', default=False)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--local-rank", default=-1, type=int, dest='local_rank')
    # dataloader
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset", type=str, help="root of dataset")
    # optimizer
    parser.add_argument("--optim", type=str, default="adam", help="type of optimizer")
    parser.add_argument(
        "--lr", "--learning_rate", type=float, default=1e-5, dest="learning_rate"
    )
    parser.add_argument(
        "--wd", "--weight_decay", type=float, default=1e-4, dest="weight_decay"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument("--cosine", action="store_true", default=True)
    parser.add_argument("--Tmax", type=int, default=100)
    # model
    parser.add_argument("--conv_layer", type=str, default="ggnn")
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--savepath", type=str)
    parser.add_argument(
        "--loadpath",
        type=str,
    )
    parser.add_argument("--head", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--drop_threshold", type=float, default=0.2)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', default=0.999)
    parser.add_argument('--multi', default=False, action='store_true')
    parser.add_argument('--lambda_u', type=float, default=1e-2)

    args = parser.parse_args()
    TIMESTAMP = "{0:%m_%dT%H_%M}".format(datetime.now())
    args.timestamp = TIMESTAMP
    if args.mode == 'semisupervised':
        args.T = 1.0
    args.command_line = _sys.argv[1:]
    return args


def init_distributed_mode(args):
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    device_ids = list(range(n_gpu))
    args.device_ids = device_ids
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    args.world_size = dist.get_world_size()


def init_optimizer(arg, model):
    if arg.optim == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=arg.learning_rate,
            weight_decay=arg.weight_decay,
            momentum=arg.momentum,
        )
    if arg.optim == "adam":
        optimizer = Adam(
            model.parameters(), lr=arg.learning_rate, weight_decay=arg.weight_decay
        )
    return optimizer


def log_experiment_dir(args):
    comment = args.comment + '_' if args.comment else ""
    if args.ablation == 0:
        return Path('experiment') / (comment + args.timestamp + f'_{args.trainer}_lr{args.learning_rate}_bs{args.batch_size}_rate{args.label_rate}_{args.dataset}')
    else:
        return Path('experiment') / (comment + args.timestamp + f'_{args.trainer}_lr{args.learning_rate}_bs{args.batch_size}_rate{args.label_rate}_var{args.ablation}_{args.dataset}')


def save_args(args):
    path = (log_experiment_dir(args) / "hyperparams.json").absolute()
    if args.distributed:
        if dist.get_rank() == 0:
            with open(path, 'w+', encoding='utf-8') as f:
                json.dump(
                    vars(args),
                    f,
                    ensure_ascii=True,
                    sort_keys=True,
                    indent=4,
                )
    else:
        with open(path, 'w+', encoding='utf-8') as f:
            json.dump(
                vars(args),
                f,
                ensure_ascii=True,
                sort_keys=True,
                indent=4,
            )


def record_hypherparams(arg):
    if 'semi' in arg.mode[0]:
        return f"e{arg.epochs}_bs{arg.batch_size}_lr{arg.learning_rate}_optim{arg.optim}_ema{arg.ema}_{arg.dataset}"
    else:
        return f"e{arg.epochs}_bs{arg.batch_size}_lr{arg.learning_rate}_wd{arg.weight_decay}_mom{arg.momentum}_cos{arg.cosine}_tmax{arg.Tmax}_{arg.dataset}"


def log_dir(arg):
    flag = arg.mode[0].upper()
    if 'semi' in arg.mode:
        flag = 'SE'
    if arg.mode == 'cla':
        flag = 'LC'
    return Path("runs") / (arg.timestamp + f"_{flag}_" + record_hypherparams(arg))


def semi_supervised_log(writer, payload, epoch):
    writer.add_scalar("train/loss_l", payload["train/loss_l"], epoch)
    writer.add_scalar("train/loss_u", payload["train/loss_u"], epoch)
    writer.add_scalar("train/acc", payload["train/acc"], epoch)
    writer.add_scalar("test/acc", payload["test/acc"], epoch)
    writer.add_scalar('test/acc3', payload['test/acc3'], epoch)
    writer.add_scalar('test/acc5', payload['test/acc5'], epoch)
    writer.add_scalar('test/mfr', payload['test/mfr'], epoch)
    writer.add_scalar('test/mar', payload['test/mar'], epoch)
    writer.add_scalar("test/loss", payload["test/loss"], epoch)


def supervised_log(writer, payload, epoch):
    writer.add_scalar("train/loss", payload["train/loss"], epoch)
    writer.add_scalar("train/acc", payload["train/acc"], epoch)
    writer.add_scalar("test/acc", payload["test/acc"], epoch)
    writer.add_scalar("test/loss", payload["test/loss"], epoch)
    writer.add_scalar('test/acc3', payload['test/acc3'], epoch)
    writer.add_scalar('test/acc5', payload['test/acc5'], epoch)
    writer.add_scalar('test/mfr', payload['test/mfr'], epoch)
    writer.add_scalar('test/mar', payload['test/mar'], epoch)


def semi_supervised_learning(args):
    if args.local_rank not in [-1, 0]:
        dist.barrier()

    if not args.dataset:
        raise Exception("Must specify a semi dataset.")
    labelset = torch.load(dataset_path / args.dataset)
    unlabelset = torch.load(dataset_path / 'unlabel_' + args.dataset)
    testset = torch.load(dataset_path / 'test_' + args.dataset)
    labelset = SupervisedDataset(labelset)
    unlabelset = SupervisedDataset(unlabelset)
    testset = SupervisedDataset(testset)
    print(f'train size: {len(labelset)} test size: {len(testset)}')

    if args.local_rank == 0:
        dist.barrier()

    if args.distributed:
        device = torch.device('cuda', args.local_rank)
        labelloader = DataLoader(labelset, batch_size=args.batch_size, sampler=DistributedSampler(labelset), num_workers=4)
        unlabelloader = DataLoader(unlabelset, batch_size=args.batch_size, sampler=DistributedSampler(unlabelset), num_workers=4)
        # testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    else:
        device = 'cuda:1'
        labelloader = DataLoader(labelset, batch_size=args.batch_size, shuffle=True)
        unlabelloader = DataLoader(unlabelset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    model = GraphConvClassifier(args).to(device)

    optimizer = init_optimizer(args, model)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Tmax)
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(log_dir=log_experiment_dir(args))
        save_args(args)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank) 


    trainer = SemiSupervisedTrainer(labelloader, unlabelloader, args)

    evaluator = Evaluator(testloader, args)

    best_model = None
    best_acc = -1

    for epoch in tqdm(range(args.epochs)):
        if args.distributed:
            labelloader.sampler.set_epoch(epoch)
            unlabelloader.sampler.set_epoch(epoch)
        payload = {}
        train_payload = trainer.train(epoch, model, optimizer)
        test_model = model
        if args.local_rank in [-1, 0]:    
            test_payload = evaluator.evaluate(epoch, test_model)
            payload.update(train_payload)
            payload.update(test_payload)
            semi_supervised_log(writer, payload, epoch)
            if payload['test/acc'] > best_acc:
                best_acc = payload['test/acc']
                torch.save({"model": model.state_dict()}, log_experiment_dir(args) / "best_model.zip")
            # torch.save({"model": model.state_dict()}, log_dir(args) / f"model_{epoch}.zip")
        if args.cosine:
            scheduler.step()
    if args.local_rank in [-1, 0]:             
        writer.flush()
        writer.close()
        print(f'best acc/1 is: {best_acc}')
        torch.save({"model": model.state_dict()}, log_experiment_dir(args) / "final_model.zip")
    

def classification(args):
    # if args.dataset:
        # dataset = torch.load(dataset_path / args.dataset)
    # trainset, testset = split_dataset(dataset, 0)
    trainset = ClassificationDataset()
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # testloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=32)
    model = Classifier(args).to('cuda:1')
    # model = torch.compile(model, mode='reduce-overhead')
    optimizer = init_optimizer(args, model)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Tmax)
    writer = SummaryWriter(log_dir=log_dir(args))
    trainer = ClassifierTrainer(trainloader, args)
    # evaluator = LogClassifierEvaluator(testloader, args)
    save_args(args)
    best_model = None
    best_acc = 0
    for epoch in tqdm(range(args.epochs)):
        payload = {}
        train_payload = trainer.train(epoch, model, optimizer)
        # test_payload = evaluator.evaluate(epoch, model)
        payload.update(train_payload)
        # payload.update(test_payload)
        payload['test/acc'] = payload['train/acc']
        payload['test/loss'] = payload['train/loss']
        supervised_log(writer, payload, epoch)
        if payload['test/acc'] > best_acc:
            best_acc = payload['test/acc']
            best_model = model.state_dict()
        # torch.save({"model": model.state_dict()}, create_log_dir(arg) / f"model_{epoch}.zip")
        if args.cosine:
            scheduler.step()
    writer.flush()
    writer.close()
    torch.save({"model": best_model}, log_dir(args) / "model.zip")    



def main():
    args = parse_option()
    setup_seed(args.seed)
    if args.distributed:
        init_distributed_mode(args)
    if args.mode == 'semisupervised':
        semi_supervised_learning(args)
    if args.mode == 'cla':
        classification(args)


if __name__ == "__main__":
    main()