import numpy as np
import pandas as pd


def load_statistics(filepath, norm_type = 'new'):
    """
    Load mean and std for each variable and level from the Excel file.
    """

    df = pd.read_excel(filepath)
    statistics = {}
    
    if norm_type == 'old':
        for _, row in df.iterrows():
            var_name = row['variable']
            level = int(row['level']) if not pd.isna(row['level']) else 0  # Level 0 for single-level vars
            key = f"{var_name}_{level}" if level != 0 else var_name
            statistics[key] = {'mean': row['mean'], 'std': row['std']}
    else:
        for _, row in df.iterrows():
            var_name = row['Variable']
            key = var_name
            statistics[key] = {'mean': row['mean'], 'std': row['std']}

    return statistics

def stack_variables(ds, statistics=None, ff=False, chanh=True):
    """
    Stack selected variables from the dataset and normalize using the provided statistics.
    """
    # Define specific isobaric levels
    selected_isobaric_levels = [1000, 975, 950, 925, 900, 875, 850, 825, 
               800, 775, 750, 725, 700, 650, 600, 550, 
               500, 450, 400, 350, 300, 250, 200, 150, 100]

    if ff:
        mul_vars = ['H', 'OMEGA', 'QI', 'QL', 'QV', 'RH', 'T', 'U', 'V']
    else:
        mul_vars = ['H', 'OME   GA', 'RH', 'T', 'U', 'V']

    if chanh:
        # mul_vars = ['RH', 'T', 'OMEGA', 'U', 'V']
        # mul_vars = ['H', 'OMEGA', 'RH', 'T', 'U', 'V']
        mul_vars = ['H']
        
    stacked_vars = []

    for var_name in mul_vars:
        variable_data = []
        
        if chanh:
            # if var_name == 'RH':
            #     selected_isobaric_levels = [750]
            # elif var_name == 'T':
            #     selected_isobaric_levels = [900, 500]
            # elif var_name == 'OMEGA':
            #     selected_isobaric_levels = [500]
            # elif var_name == 'U':
            #     selected_isobaric_levels = [800, 200]
            # elif var_name == 'V':
            #     selected_isobaric_levels = [800, 200]
            
            selected_isobaric_levels = [800, 200]

        for level in selected_isobaric_levels:
            key = f"{var_name}_{level}"
            level_data = ds[var_name].sel(isobaricInhPa=level).values
            if statistics and key in statistics:
                level_data = (level_data - statistics[key]['mean']) / statistics[key]['std']
            level_data = np.nan_to_num(level_data)
            variable_data.append(level_data)
        stacked_vars.append(np.stack(variable_data, axis=-1))
    

    single_vars = ['PHIS', 'PS', 'SLP']
    if chanh:
        single_vars = ['PHIS']

    single_vars_data = []
    for var_name in single_vars:
        key = var_name
        single_data = ds[var_name].values
        if statistics and key in statistics:
            single_data = (single_data - statistics[key]['mean']) / statistics[key]['std']
        single_data = np.nan_to_num(single_data)
        single_vars_data.append(single_data)
    
    
    if len(np.array(single_vars_data).shape) == 3:
        single_vars_data = np.expand_dims(single_vars_data, axis=-1)
    
    data = np.concatenate([single_vars_data, stacked_vars], axis=-1)
 
    # Rearrange axes for PyTorch compatibility: [C, H, W]
    data = np.squeeze(data, axis=0).transpose(2, 0, 1)
    

    return data
