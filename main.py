from config import parse_arguments
from data.dataset import MerraDataset
from data.split_set import split_and_normalize
from data.data_loader import create_dataloader
from models.resnet import Resnet
from utils.class_weight import class_weight
from utils.training import train_model

import time
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    config = parse_arguments()


    train_dataset, val_dataset, test_dataset = split_and_normalize(
                                                    csv_path=config.csv_path,
                                                    pos_ind=config.pos_ind,
                                                    neg_remaining=config.neg_remaining,
                                                    mul_pos=config.mul_pos
                                                )

    train_loader, val_loader, test_loader = create_dataloader(
                                                    train_dataset,
                                                    val_dataset,
                                                    test_dataset
                                                )

    start_time = time.time()
    # Example usage
    for batch_data, batch_labels in train_loader:
        print("Train batch data shape:", batch_data.shape)
        print("Train batch labels shape:", batch_labels.shape)
        break
    end_time = time.time()
    print("Time taken for one epoch:", end_time - start_time)

    num_neg, num_pos = (train_dataset.labels == 0).sum(), (train_dataset.labels == 1).sum()
    print(f"Number of 0 labels: {num_neg}")
    print(f"Number of 1 labels: {num_pos}")
    print(f"Number of 0 labels: {(val_dataset.labels == 0).sum()}")
    print(f"Number of 1 labels: {(val_dataset.labels == 1).sum()}")
    print(f"Number of 0 labels: {(test_dataset.labels == 0).sum()}")
    print(f"Number of 1 labels: {(test_dataset.labels == 1).sum()}")

    inp_channel = batch_data.shape[1]
    model = Resnet(inp_channels=inp_channel,
              num_residual_block=[2, 2, 2, 2],
              num_class=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weight = class_weight(num_pos, num_neg)
    criterion = nn.BCEWithLogitsLoss(pos_weight=(class_weight[1]).float()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    train_model(config, model, criterion, optimizer, train_loader, val_loader, device)

if __name__ == "__main__":
    main()
