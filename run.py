import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from graphdataset import CodeGraphDataset

train_dataset = CodeGraphDataset('dataset/python/valid.jsonl')
stats = train_dataset.get_stats()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)