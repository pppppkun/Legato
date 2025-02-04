from random import random
import torch

def drop_edge(edge_index, edge_weights, p, threshold):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(
        edge_weights < threshold, torch.ones_like(edge_weights) * threshold
    )
    sel_mask = torch.bernoulli(1.0 - edge_weights).to(torch.bool)
    return edge_index[:, sel_mask]



def augmentation_based_on_attention(args, model, data):
    model.eval()
    with torch.no_grad():
        attention, batch = model.attention(data.x, data.edge_index, data.batch)
        edge_shape = data.edge_index.shape[1]
        attention = attention[:edge_shape, :]
        attention = attention.cpu()
    attention = torch.clip(1 - attention, min=0, max=1).reshape(-1)
    edge_index = drop_edge(data.edge_index, attention, args.drop_rate, args.drop_threshold)
    return edge_index