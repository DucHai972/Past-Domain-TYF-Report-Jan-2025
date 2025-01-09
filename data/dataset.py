from torch.utils.data import Dataset
from .preprocessing import stack_variables, load_statistics
import random
import xarray as xr
import pandas as pd
import numpy as np

class MerraDataset(Dataset):
    def __init__(self, data, pos_ind, norm_type='new'):
        # self.data = data
        self.data = data[data['Label'] != -1].reset_index(drop=True)
        self.pos_ind = pos_ind
        
        # self.labels = self._assign_labels()
        self.labels = self.data['Label'].values
        self.paths = self.data['Path'].values
        if norm_type == 'new':
            self.stats_file = "/N/slate/tnn3/HaiND/11-17_newPast/data_statistics.xlsx"
        elif norm_type == 'old':
            self.stats_file = "/N/slate/tnn3/DucHGA/TC/DataMerra/Output/data_statistics.xlsx"
        else:
            self.stats_file = None
        # self.mean = mean
        # self.std = std

    def _assign_labels(self):
        """
        Assign labels based on the 'Domain' and 'Step' values.

        Returns:
        - A NumPy array of labels (1 for POSITIVE or specified Past, 0 otherwise).
        """
        labels = []
        for _, row in self.data.iterrows():
            if row['Domain'] == 'POSITIVE':
                labels.append(1)
            elif row['Domain'] == 'Past' and 1 <= row['Step'] <= self.pos_ind:
                labels.append(1)
            else:
                labels.append(0)
        return pd.Series(labels).values
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        

        ds = xr.open_dataset(path)
        
        # Subset by latitude and longitude ranges
        # latitude_range = ds.coords['latitude'].data[100:161]
        # longitude_range = ds.coords['longitude'].data[64:145]

        latitude_range = ds.coords['latitude'].data[110:141]
        longitude_range = ds.coords['longitude'].data[64:129]

        ds = ds.sel(latitude=latitude_range, longitude=longitude_range)
        
        if self.stats_file:
        # Stack variables
            data = stack_variables(ds, statistics=load_statistics(self.stats_file))
        # data = stack_variables(ds)
        else:
            data = stack_variables(ds)

        return data, label

