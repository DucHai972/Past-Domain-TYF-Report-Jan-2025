from data.dataset import MerraDataset
import pandas as pd
import time

def save_dataset_to_csv(dataset, file_name):
    """
    Save the dataset to a CSV file with 'Path' and 'Label' columns.

    Args:
    - dataset: MerraDataset object
    - file_name: Name of the CSV file to save
    """
    data_dict = {
        'Path': dataset.paths,
        'Step': dataset.data['Step'],
        'Label': dataset.labels,
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(file_name, index=False)
    print(f"Dataset saved to {file_name}")

# Split the dataset
def split_and_normalize(csv_file, pos_ind, neg_remaining=True, mul_pos=True, norm_type='new'): 
    # Load the CSV and filter based on criteria
    data = pd.read_csv(csv_file)
    
    train_data = data[data['Year'].between(2008, 2014)]
    val_data = data[data['Year'].between(2015, 2017)]
    test_data = data[data['Year'].between(2018, 2021)]

    
    # Create dataset objects
    train_dataset = MerraDataset(train_data, pos_ind=pos_ind, norm_type=norm_type)
    val_dataset = MerraDataset(val_data, pos_ind=pos_ind, norm_type=norm_type)
    test_dataset = MerraDataset(test_data, pos_ind=pos_ind, norm_type=norm_type)

    # save_dataset_to_csv(train_dataset, "/N/slate/tnn3/HaiND/11-17_newPast/train_dataset_2.csv")
    # save_dataset_to_csv(val_dataset, "/N/slate/tnn3/HaiND/11-17_newPast/val_dataset_2.csv")
    # save_dataset_to_csv(test_dataset, "/N/slate/tnn3/HaiND/11-17_newPast/test_dataset_2.csv")

    return train_dataset, val_dataset, test_dataset


