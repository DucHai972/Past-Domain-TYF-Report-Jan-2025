from torch.utils.data import Dataset
from .preprocessing import stack_variables, load_statistics
import random
import xarray as xr
import pandas as pd
import numpy as np
import os

class MerraDataset(Dataset):
    def __init__(self, data, pos_ind, norm_type='new', small_set=False):
        self.data = data
        self.pos_ind = pos_ind
        
        # self.labels = self._assign_labels()
        self.labels = self.data['Label']
        self.paths = self.data['Path'].values
        if norm_type == 'new':
            self.stats_file = "/N/slate/tnn3/HaiND/11-17_newPast/data_statistics.xlsx"
        elif norm_type == 'old':
            self.stats_file = "/N/slate/tnn3/DucHGA/TC/DataMerra/Output/data_statistics.xlsx"
        else:
            self.stats_file = None
        # self.mean = mean
        # self.std = std
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        
        # check if path exist else raise error
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file or directory '{path}' does not exist.")

        ds = xr.open_dataset(path)
        
        # Subset by latitude and longitude ranges
        latitude_range = ds.coords['latitude'].data[100:161]
        longitude_range = ds.coords['longitude'].data[64:145]

        # latitude_range = ds.coords['latitude'].data[110:141]
        # longitude_range = ds.coords['longitude'].data[64:129]

        ds = ds.sel(latitude=latitude_range, longitude=longitude_range)
        
        if self.stats_file:
        # Stack variables
            data = stack_variables(ds, statistics=load_statistics(self.stats_file), small_set=False)
        # data = stack_variables(ds)
        else:
            data = stack_variables(ds, small_set=False)

        return data, label

