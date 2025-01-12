from data.dataset import MerraDataset
import pandas as pd
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

def save_dataset_to_csv(dataset, file_name):
    """
    Save the dataset to a CSV file with 'Path' and 'Label' columns.

    Args:
    - dataset: MerraDataset object
    - file_name: Name of the CSV file to save
    """
    data_dict = {
        'Path': dataset.paths,
        'Filename': dataset.data['Filename'],
        'Year': dataset.data['Year'],
        'Label': dataset.labels,
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(file_name, index=False)
    print(f"Dataset saved to {file_name}")

def convert_timestamp_to_filename(timestamp, time_steps_back=0):
    try:
        # Parse the timestamp
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Calculate the new timestamp by going back the specified number of time steps
        # Each time step is 3 hours
        dt -= timedelta(hours=time_steps_back * 3)
        
        # Format into the desired filename
        filename = f"merra2_{dt.strftime('%Y%m%d_%H_00')}.nc"
        return filename
    except ValueError as e:
        print(f"Error parsing timestamp: {timestamp}. Ensure it is in 'YYYY-MM-DD HH:MM:SS' format.")
        return None

def undersample_data(data, label_column="Label", ratio=10):

    data_minority = data[data[label_column] == 1]
    data_majority = data[data[label_column] == 0]
    
    target_majority_count = len(data_minority) * ratio
    data_majority_sampled = data_majority.sample(n=target_majority_count, random_state=42)
    
    undersampled_data = pd.concat([data_minority, data_majority_sampled], axis=0)
    undersampled_data = undersampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return undersampled_data

# Split the dataset
def split_and_normalize(csv_file, pos_ind, small_set, norm_type='new'): 
    # Load the CSV and filter based on criteria
    data = pd.read_csv(csv_file)
    
    data["Path"] = data["Path"].str.replace("nasa-merra2", "nasa-merra2.old")
    data = data[data['Label'] != -1].reset_index(drop=True)
    data['Label'] = 0
    
    ibtracs_file = '/N/slate/tnn3/HaiND/01-06_report/csv/FIRST_MERRA2_IBTRACS.csv'
    ibtracs_data = pd.read_csv(ibtracs_file)



    ibtracs_filenames = set(
    ibtracs_data['ISO_TIME'].apply(lambda x: convert_timestamp_to_filename(x, time_steps_back=pos_ind))
    )

    
    data['Label'] = data['Filename'].isin(ibtracs_filenames).astype(int)

    # train_data = data[data['Year'].between(2008, 2014)]
    # val_data = data[data['Year'].between(2015, 2017)]
    # test_data = data[data['Year'].between(2018, 2021)]

    if not small_set:
        train_data = data[data['Year'].between(1980, 2016)]
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
        test_data = data[data['Year'].between(2017, 2023)]
    else:
        train_data = data[data['Year'].between(2008, 2012)]
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
        test_data = data[data['Year'] == 2018]

    # Undersampling
    train_data = undersample_data(train_data, label_column="Label", ratio=10)
    val_data = undersample_data(val_data, label_column="Label", ratio=10)
    test_data = undersample_data(test_data, label_column="Label", ratio=10)
    
    # Create dataset objects
    train_dataset = MerraDataset(train_data, pos_ind=pos_ind, norm_type=norm_type, small_set=small_set)
    val_dataset = MerraDataset(val_data, pos_ind=pos_ind, norm_type=norm_type, small_set=small_set)
    test_dataset = MerraDataset(test_data, pos_ind=pos_ind, norm_type=norm_type, small_set=small_set)

    save_dataset_to_csv(train_dataset, "/N/slate/tnn3/HaiND/01-06_report/csv/train_dataset.csv")
    save_dataset_to_csv(val_dataset, "/N/slate/tnn3/HaiND/01-06_report/csv/val_dataset.csv")
    save_dataset_to_csv(test_dataset, "/N/slate/tnn3/HaiND/01-06_report/csv/test_dataset.csv")

    return train_dataset, val_dataset, test_dataset


