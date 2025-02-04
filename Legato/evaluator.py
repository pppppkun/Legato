from toolbox import AverageMeter, first_rank, ACC, multi_ACC, average_rank, accuracy
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

class Evaluator:
    def __init__(self, dataloader, arg):
        self.arg = arg
        self.dataloader = dataloader
        if self.arg.multi:
            self.acc_f = multi_ACC
        else:
            self.acc_f = ACC

    def _reset_stats(self):
        self.acc1_meters = AverageMeter()
        self.acc3_meters = AverageMeter()
        self.acc5_meters = AverageMeter()
        self.loss_meters = AverageMeter()
        self.firstrank_meters = AverageMeter()
        self.reciprocalrank_meters = AverageMeter()
        self.mar = AverageMeter()

    def evaluate(self, epoch, model):
        self._reset_stats()
        model.eval()
        # model.to(device)
        device = next(model.parameters()).device
        for i, data in enumerate(self.dataloader):
            data.to(device)
            with torch.no_grad():
                node_feature = model(data.x, data.edge_index, data.batch, data.mask)
                graph_ids = torch.unique(data.batch)
                loss = 0
                for gid in graph_ids:
                    nodes_in_target_graph = (data.batch == gid).nonzero(as_tuple=True)[
                        0
                    ]
                    feature = node_feature[nodes_in_target_graph]
                    y = data.y[nodes_in_target_graph]
                    loss += torch.sum(
                        -torch.log(feature.clamp(min=1e-10, max=1)) * y, dim=-1
                    )
                    acc = self.acc_f(feature, y, [1, 3, 5])
                    _first_rank = first_rank(feature, y)[0]
                    _average_rank = average_rank(feature, y)
                    self.acc1_meters.update(acc[0], 1)
                    self.acc3_meters.update(acc[1], 1)
                    self.acc5_meters.update(acc[2], 1)
                    self.firstrank_meters.update(_first_rank, 1)
                    self.reciprocalrank_meters.update(1 / _first_rank, 1)
                    self.mar.update(_average_rank, 1)

        self.loss_meters.update(loss.item(), len(graph_ids))
        payload = {
            "test/acc": self.acc1_meters.avg,
            "test/acc3": self.acc3_meters.avg,
            "test/acc5": self.acc5_meters.avg,
            "test/loss": self.loss_meters.avg,
            "test/mfr": self.firstrank_meters.avg,
            "test/mrr": self.reciprocalrank_meters.avg,
            "test/mar": self.mar.avg
        }
        return payload


class ClassifierEvaluator:
    def __init__(self, dataloader, args):
        self.args = args
        self.dataloader = dataloader
        self.criterion = CrossEntropyLoss()

    def _reset_stats(self):
        self.acc_meters = AverageMeter()
        self.loss_meters = AverageMeter()

    def evaluate(self, epoch, model):
        self._reset_stats()
        model.eval()
        # model.to(device)
        device = next(model.parameters()).device
        for i, data in tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=False):
            graph = data[0]
            labels = data[1]
            graph.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                logits = model(graph.x, graph.edge_index, graph.batch)
                loss = self.criterion(logits, labels)
                acc = accuracy(logits, labels, topk=(1, ))
                self.acc_meters.update(acc[0], labels.shape[0])
                self.loss_meters.update(loss.item(), labels.shape[0])
        payload = {
            "test/acc": self.acc_meters.avg,
            "test/loss": self.loss_meters.avg,
        }
        return payload