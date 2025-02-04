from random import random
import torch
import torch.nn.functional as F
import numpy as np
from toolbox import AverageMeter, ACC, multi_ACC, accuracy
from tqdm import tqdm
from torch import nn
from torch.nn import CrossEntropyLoss
from augmentation import augmentation_based_on_attention
from model import AdapGAT, Classifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


def gmm_bic_score(estimator, X):
    return -estimator.bic(X)

class ClassifierTrainer:
    def __init__(self, dataloader, args):
        self.dataloader = dataloader
        self.args = args
        self.criterion = CrossEntropyLoss()

    def _reset_stats(self):
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()


    def train(self, epoch, model, optimizer):
        # device = torch.device("cuda:1")
        self._reset_stats()
        model.train()
        device = next(model.parameters()).device
        for i, data in tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=False):
            graph = data[0]
            labels = data[1]
            graph.to(device)
            labels = labels.to(device)
            logits = model(graph.x, graph.edge_index, graph.batch)
            loss = self.criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = accuracy(logits, labels, topk=(1, ))
            self.acc_meters.update(acc[0], labels.shape[0])
            self.loss_meters.update(loss.item(), labels.shape[0])
        payload = {
            "train/acc": self.acc_meters.avg,
            "train/loss": self.loss_meters.avg,
        }
        return payload


class SemiSupervisedTrainer:
    def __init__(self, labeled_trainloader, unlabeled_trainloader, args) -> None:
        self.ltl = labeled_trainloader
        self.utl = unlabeled_trainloader
        self.param_grid = {
            'n_components': [1, 2],
            'covariance_type': ['spherical', 'tied', 'diag', 'full']
        }
        self.args = args
        self.attention_model = Classifier(self.args)
        self.attention_model.load_state_dict(torch.load('model.zip')['model'])
        self.attention_model.attention_available()
        if self.args.multi:
            self.acc_f = multi_ACC
        else:
            self.acc_f = ACC
   
    
    def _reset_stats(self):
        self.losses_l = AverageMeter()
        self.losses_u = AverageMeter()
        self.mask_probs = AverageMeter()
        self.acc = AverageMeter()

    def train(self, epoch, model, optimizer):
        self._reset_stats()
        labeled_iter = iter(self.ltl)
        unlabeled_iter = iter(self.utl)
        if self.args.distributed:
            device = torch.device("cuda", self.args.local_rank)
        else:
            device = next(model.parameters()).device
        self.attention_model.to(device)
        eval_step = max(len(self.ltl.dataset) / self.args.batch_size, len(self.utl.dataset) / self.args.batch_size)
        eval_step = int(eval_step)
        for i in range(11):
            if pow(2, i) > eval_step:
                eval_step = pow(2, i)
                break
        # print(eval_step)
        for batch_idx in range(eval_step):
            try:
                inputs_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(self.ltl)
                inputs_l = next(labeled_iter)
            
            try:
                inputs_uw = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.utl)
                inputs_uw = next(unlabeled_iter)

            inputs_l.to(device)
            inputs_uw.to(device)
            # edge_index = augmentation_unlabeled(self.args, inputs_uw)
            edge_index = augmentation_based_on_attention(self.args, self.attention_model, inputs_uw)
            # supervised
            # edge_index.to('cuda:1')
            node_feature = model(inputs_l.x, inputs_l.edge_index, inputs_l.batch, inputs_l.mask)
            label_graph_ids = torch.unique(inputs_l.batch)
            loss = 0
            for gid in label_graph_ids:
                nodes_in_target_graph = (inputs_l.batch == gid).nonzero(as_tuple=True)[0]
                feature = node_feature[nodes_in_target_graph]
                y = inputs_l.y[nodes_in_target_graph]
                loss += torch.sum(
                     -torch.log(feature.clamp(min=1e-10, max=1))
                    * inputs_l.y[nodes_in_target_graph],
                    dim=-1,
                )
                # acc = torch.argmax(feature) == torch.argmax(inputs_l.y[nodes_in_target_graph])
                acc = self.acc_f(feature, y, [1])
                self.acc.update(acc[0], 1)
            self.losses_l.update(loss.item(), len(label_graph_ids))
            if self.args.distributed:
                n1 = model.module.logits(inputs_uw.x, inputs_uw.edge_index, inputs_uw.batch, inputs_uw.mask)
            else:
                n1 = model.logits(inputs_uw.x, inputs_uw.edge_index, inputs_uw.batch, inputs_uw.mask)
            n2 = model(inputs_uw.x, edge_index, inputs_uw.batch, inputs_uw.mask)
            loss_u = torch.sum(torch.tensor([0.0])).to(device)
            unlabel_graph_ids = torch.unique(inputs_uw.batch)
            for gid in unlabel_graph_ids:
                nodes_in_target_graph = (inputs_uw.batch == gid).nonzero(as_tuple=True)[0]
                graph_mask = (inputs_uw.mask[nodes_in_target_graph] == 1).nonzero(as_tuple=True)[0]
                pseudo_label = F.softmax(n1[nodes_in_target_graph].detach()/self.args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                grid_search = GridSearchCV(GaussianMixture(), param_grid=self.param_grid, scoring=gmm_bic_score)
                grid_search.fit(pseudo_label[graph_mask].cpu().numpy().reshape(-1, 1))
                gmm = grid_search.best_estimator_
                if gmm.n_components != 2:
                    continue
                indexs = np.where(gmm.predict(pseudo_label[graph_mask].cpu().numpy().reshape(-1, 1)) == gmm.means_.argmax())[0]
                fake_y = inputs_uw.y[nodes_in_target_graph].float()
                fake_y[graph_mask[indexs]] = pseudo_label[graph_mask[indexs]]
                # mask = max_probs.ge(self.args.threshold).float()
                loss_u += torch.sum(
                    -torch.log(n2[nodes_in_target_graph].clamp(min=1e-10, max=1))
                    * fake_y,
                    dim=-1
                )
            self.losses_u.update(loss_u.item(), len(unlabel_graph_ids))
            loss = loss + self.args.lambda_u * loss_u
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        payload = {
            "train/acc": self.acc.avg,
            "train/loss_l": self.losses_l.avg,
            "train/loss_u": self.losses_u.avg
        }
        return payload