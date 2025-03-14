import pandas as pd
import numpy as np
from data.dataset import TimeSeriesDataset
from torch.utils.data import DataLoader, random_split
import sys
import os

def create_dataloader(df:pd.DataFrame, config:dict):
    
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['Date'], inplace=True)

    dataset = TimeSeriesDataset(df.to_numpy().astype(np.float32), config['seq_len'], config['pred_len']) 
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Perform random split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    dataloaders = {
        'train_dataloader':train_dataloader,
        'val_dataloader':val_dataloader,
        'test_dataloader':test_dataloader,
    }

    return dataloaders