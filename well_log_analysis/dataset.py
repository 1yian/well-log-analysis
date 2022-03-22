import lasio
import torch
import sys
import numpy as np
import copy
import pandas as pd
class WellDataset(torch.utils.data.Dataset):

    def __init__(self, path, return_sites=[], clean_sites=[], sequence_length=1):
        self.df = lasio.read(path).df().reset_index()[[site for site in return_sites]].dropna(subset=clean_sites)
        self.return_sites = return_sites
        self.sequence_length = sequence_length
    
        self.map = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx not in self.map:
            ret = np.array([self.df.iloc[max(idx, 0)][self.return_sites] for idx in range(idx - self.sequence_length + 1, idx + 1)])
            self.map[idx] = ret
        return self.map[idx]

    def __add__(self, other_dataset):

        if type(other_dataset) != type(self):
            raise TypeError("Must be added with another WellDataset")

        if set(other_dataset.return_sites) != set(self.return_sites):
            raise TypeError("Datasets must have the same return sites")

        new_dataset = copy.deepcopy(self)
        new_dataset.df = pd.concat([self.df, other_dataset.df])
        return new_dataset
    
    def mean(self):
        
        return np.mean(np.array(self.df[self.return_sites]), axis=0)
    
    def std(self):
        return np.std(np.array(self.df[self.return_sites]), axis=0)
