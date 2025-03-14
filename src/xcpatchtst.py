import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath('/Users/engchongyock/Desktop/XCPatchTST'))
from config import get_config
from utils.helper import create_crosschannel_patchtst, create_dlinear,create_patchtst
from data.data_loader import create_dataloader
from utils.train import train

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def select_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    
def load_dataset(config):
    return pd.read_csv(config['dataset_dir'])

def create_model(model_name, device, in_channels, config):
    if model_name == 'xcpatchtst':
        return create_crosschannel_patchtst(
            device=device,
            in_channels=in_channels,
            pred_len=config['pred_len'],
            patch_size=config['patch_size'],
            stride=config['stride'],
            d_model=config['d_model'],
            kernel_size=config['kernel_size']
        )
    elif model_name == 'patchtst':
        return create_patchtst(
            device=device,
            in_channels=in_channels,
            pred_len=config['pred_len'],
            patch_size=config['patch_size'],
            stride=config['stride'],
            d_model=config['d_model'],
            kernel_size=config['kernel_size']
        )
    else:
        return create_dlinear(
            device=device,
            input_len=config['seq_len'],
            output_len=config['pred_len'],
            feature_dim=in_channels,
        )

if __name__ == "__main__":
    # Select device dynamically
    device = select_device()
    print(f"Using device: {device}")

    # Load config and dataset
    config = get_config()
    df = load_dataset(config)

    # Get the number of input channels from the dataset
    in_channels = df.shape[1]
    print(f"Input channels: {in_channels}")

    # Model selection and creation
    model_name = config['model_name']  
    model = create_model(model_name, device, in_channels, config)

    # Create dataloaders
    dataloaders = create_dataloader(df, config)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    train(model=model, train_dataloader=dataloaders['train_dataloader'], 
                test_dataloader=dataloaders['test_dataloader'], 
                val_dataloader=dataloaders['val_dataloader'], 
                optimizer=optimizer, 
                loss_fn=loss_fn, 
                epochs=config['epochs'], 
                device=device)
