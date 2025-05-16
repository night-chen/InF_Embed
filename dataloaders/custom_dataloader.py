import torch
from torch.utils.data import Dataset
from typing import List, Dict

class BaseDataset(Dataset):
    """
    Base class for contrastive learning datasets.
    It expects a `self.samples` list, each element describing:
      - tokenized query
      - tokenized positive doc
      - tokenized negative docs (list of length `num_hard_negatives`)
    """
    def __init__(self, data=[], num_hard_negatives: int = 1):
        super().__init__()
        self.num_hard_negatives = num_hard_negatives
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def select(self, indices):
        # Create a new instance without calling __init__
        new_dataset = self.__class__.__new__(self.__class__)
        # Shallow copy the current __dict__
        new_dataset.__dict__ = self.__dict__.copy()
        # Replace data with only the selected items
        new_dataset.data = [self.data[i] for i in indices]
        return new_dataset


class IndexedDataset(BaseDataset):
    def __init__(self, data=[]):
        self.data = data
        # Store the index of each sample
        self.indices = list(range(len(data)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "text": self.data[idx],
            "indices": self.indices[idx]  # or just return (self.data[idx], idx)
        }