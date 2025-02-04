import torch
from collections import defaultdict
from toolbox import *
from torch_geometric.data import Dataset, Data, Batch
from pathlib import Path
from tqdm import tqdm
from random import shuffle
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class SupervisedDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        # self.data = torch.load(datafolders)
        self.data = self.load_dataset(dataset)

    def load_dataset(self, dataset):
        new = []
        for index, cp, graph, is_pass in dataset:
            if not is_pass:
                new.append(graph)
        return new

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


class ClassificationDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.cache_size = 3000
        self.load_dataset('dataset/binary_graph')

    def load_dataset_multi_threading(self, datafold):
        data_path = Path(datafold)
        self.meta = []
        self.paths = list(data_path.iterdir())
        self.loaded = []
        self.unloaded = []
        count = 0
        for p in data_path.iterdir():
            if count < self.cache_size:
                self.loaded.append(p)
                count += 1
            else:
                self.unloaded.append(p)
        self.unloaded = self.unloaded[:3000]
        with tqdm(total=len(self.loaded + self.unloaded)) as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                for ret in executor.map(torch.load, self.loaded + self.unloaded):
                    pbar.update(1)
                    self.meta.append(ret)
    

    def load_dataset(self, datafold):
        data_path = Path(datafold)
        self.meta = []
        self.loaded = []
        self.paths = list(data_path.iterdir())
        self.unloaded = []
        self.cache_list = torch.load('cache_graph.list')
        self.cache_list = list(map(lambda x: x[0], self.cache_list))
        count = 0
        for p in data_path.iterdir():
            if p in self.cache_list:
                self.loaded.append(p)
            else:
                self.unloaded.append(p)
        self.unloaded = self.unloaded[:3000]
        with tqdm(total=len(self.loaded + self.unloaded)) as pbar:
            for ret in map(torch.load, self.loaded + self.unloaded):
                self.meta.append(ret)
                pbar.update(1)
    
    def len(self):
        return len(self.loaded + self.unloaded)

    def get(self, idx):
        graph, is_pass = self.meta[idx]
        label = 0 if is_pass else 1
        return graph, label
