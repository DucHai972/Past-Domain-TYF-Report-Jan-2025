from torch.utils.data import Dataset
from .preprocessing import stack_variables, load_statistics
import random
import xarray as xr
import pandas as pd
import numpy as np
import os
import torch

class MerraDataset(Dataset):
    def __init__(self, 
                 data, 
                 pos_ind, 
                 norm_type='new', 
                 small_set=False, 
                 preprocessed_dir="/N/scratch/tnn3/data_fullmap"):
        
        self.data = data
        self.pos_ind = pos_ind
        
        # self.labels = self._assign_labels()
        self.labels = self.data['Label']
        self.paths = self.data['Path'].values
        self.filenames = self.data['Filename'].values
        self.norm_type = norm_type
        self.small_set = small_set
        self.preprocessed_dir = preprocessed_dir

        if self.norm_type == 'new':
            self.stats_file = "/N/slate/tnn3/HaiND/01-06_report/csv/data_statistics.xlsx"
        else:
            self.stats_file = None
        # self.mean = mean
        # self.std = std
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # path = self.paths[idx]
        # # label = self.labels[idx]
        # label = self.labels.iloc[idx]
        # # check if path exist else raise error
        # if not os.path.exists(path):
        #     raise FileNotFoundError(f"The file or directory '{path}' does not exist.")

        # ds = xr.open_dataset(path)
        
        # # Subset by latitude and longitude ranges
        # latitude_range = ds.coords['latitude'].data[100:161]
        # longitude_range = ds.coords['longitude'].data[64:145]

        # # latitude_range = ds.coords['latitude'].data[110:141]
        # # longitude_range = ds.coords['longitude'].data[64:129]

        # ds = ds.sel(latitude=latitude_range, longitude=longitude_range)
        
        # if self.stats_file:
        # # Stack variables
        #     data = stack_variables(ds, statistics=load_statistics(filepath=self.stats_file, norm_type=self.norm_type), small_set=self.small_set)
        # # data = stack_variables(ds)
        # else:
        #     data = stack_variables(ds, small_set=self.small_set)

        # return data, label

        filename = self.filenames[idx].replace(".nc", ".pt")  # Convert .nc filename to .pt
        label = self.labels.iloc[idx]

        file_path = os.path.join(self.preprocessed_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        # Load preprocessed tensor
        data = torch.load(file_path)

        # Normalised
        if self.stats_file:
            stat_df = pd.read_excel(self.stats_file)

            means = stat_df["Mean"].values.astype(np.float32)
            stds = stat_df["Std"].values.astype(np.float32)
            means = means.reshape(-1, 1, 1)
            stds = stds.reshape(-1, 1, 1)
            mean_tensor = torch.tensor(means, device=data.device, dtype=data.dtype)
            std_tensor = torch.tensor(stds, device=data.device, dtype=data.dtype)

            # Normalize the data: (data - mean) / std
            data = (data - mean_tensor) / std_tensor

        return data, label

