import torch
import torch.nn.functional as F
from typing import Final, Union, Tuple
from copy import deepcopy
from attention_nn import GATv3Conv
from torch import nn as nn
from torch_geometric.nn import GatedGraphConv, Sequential
from torch_geometric.nn.models import GAT, GCN, GIN
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.aggr import MeanAggregation




class AdapGAT(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATv3Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout.p, **kwargs)    
    
    def set_to_save_attention(self):
        self.convs[-1].log_attention = True
    
    def get_last_layer_attention(self):
        # if self.convs[-1].attention
        if hasattr(self.convs[-1], 'attention'):
            return self.convs[-1].attention
        else:
            raise ValueError("do not record attention")


class Classifier(torch.nn.Module):
    def __init__(self, args) -> None:
        super(Classifier, self).__init__()
        self.args = args
        self.graph = AdapGAT(
                in_channels=args.embed_dim,
                hidden_channels=args.embed_dim,
                out_channels=args.embed_dim,
                num_layers=args.num_layers,
                dropout=0.2,
                heads=args.head,            
            )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(args.embed_dim, args.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.embed_dim, 2),
        )
        self.readout = MeanAggregation()


    def attention_available(self):
        self.graph.set_to_save_attention()


    def forward(self, x, edge_index, batch):
        x = self.graph(x, edge_index)
        graph = self.readout(x, batch)
        out = self.linear(graph)
        return out

    def attention(self, x, edge_index, batch):
        x = self.graph(x, edge_index)
        attention = self.graph.get_last_layer_attention()
        return attention, batch

class GraphConvClassifier(torch.nn.Module):
    def __init__(self, arg) -> None:
        super(GraphConvClassifier, self).__init__()
        self.arg = arg
        # self.embedding = torch.nn.Embedding(vocab_size, arg.embed_dim, padding_idx=pad_idx)
        if arg.conv_layer == "ggnn":
            self.graph = GatedGraphConv(
                out_channels=arg.embed_dim,
                num_layers=arg.num_layers,
            )
        elif arg.conv_layer == "gat":
            self.graph = GAT(
                in_channels=arg.embed_dim,
                hidden_channels=arg.embed_dim,
                out_channels=arg.embed_dim,
                num_layers=arg.num_layers,
                dropout=0.2,
                v2=True,
                heads=arg.head,
            )
        elif arg.conv_layer == "gcn":
            self.graph = GCN(
                in_channels=arg.embed_dim,
                hidden_channels=arg.embed_dim,
                out_channels=arg.embed_dim,
                num_layers=arg.num_layers,
                dropout=0.2,
            )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(arg.embed_dim, arg.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(arg.embed_dim, 1),
        )

    def forward(self, x, edge_index, batch, mask):
        x = self.graph(x, edge_index)
        x = self.linear(x).squeeze(-1).masked_fill(mask == 0, -1e9)
        for gid in torch.unique(batch):
            nodes_in_target_graph = (batch == gid).nonzero(as_tuple=True)[0]
            # may need mask to control the meanless code.
            x[nodes_in_target_graph] = F.softmax(x[nodes_in_target_graph], dim=-1)
        return x


    def logits(self, x, edge_index, batch, mask):
        x = self.graph(x, edge_index)
        x = self.linear(x).squeeze(-1).masked_fill(mask == 0, -1e9)
        return x        

    def active_learning(self, x, edge_index, batch, mask):
        x = self.graph(x, edge_index)
        logits = self.linear(x)
        # node_features = x[mask==1]
        # logits = logits[mask==1]
        return x, logits

    def encode(self, x, edge_index, batch, mask):
        x = self.graph(x, edge_index)
        return x

