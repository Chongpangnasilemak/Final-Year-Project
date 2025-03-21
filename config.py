import argparse
import os

def get_config():
    # Default Config
    config = {
        'epochs': 10,
        'model_name':'xcpatchtst',
        'seq_len': 64,
        'pred_len': 1,
        'batch_size': 8,
        'stride': 4,
        'patch_size': 8,
        'd_model': 128,
        'kernel_size': 3,
        'tickers': ["AAPL", "MSFT", "AMZN"],
        'raw_dataset_dir': 'data/raw',
        'processed_dataset_dir': 'data/processed',
        'ticker_threshold': 4000,
        'dataset_dir': f"/Users/engchongyock/Desktop/XCPatchTST/data/dataset/dataset_2025-03-16.csv"
    }

    parser = argparse.ArgumentParser(description="Training Configurations")
    parser.add_argument("--epochs", type=int, default=config['epochs'], help="Number of training epochs")
    parser.add_argument("--seq_len", type=int, default=config['seq_len'], help="Sequence length")
    parser.add_argument("--pred_len", type=int, default=config['pred_len'], help="Prediction length")
    parser.add_argument("--batch_size", type=int, default=config['batch_size'], help="Batch size")
    parser.add_argument("--stride", type=int, default=config['stride'], help="Stride size")
    parser.add_argument("--patch_size", type=int, default=config['patch_size'], help="Patch size")
    parser.add_argument("--d_model", type=int, default=config['d_model'], help="Model dimension")
    parser.add_argument("--kernel_size", type=int, default=config['kernel_size'], help="Kernel size")
    parser.add_argument("--dataset_dir", type=str, default=config['dataset_dir'], help="Dataset directory")
    parser.add_argument("--model_name", type=str, choices=['xcpatchtst', 'patchtst', 'dlinear'], default=config['model_name'], help="Model type to use")
    parser.add_argument("--tickers", type=str, default="AAPL", help="Comma-separated list of tickers")
        
    args = parser.parse_args()

    # Update config with argparse values
    config.update(vars(args))

    config['tickers'] = [ticker.strip() for ticker in config['tickers'].split(',')]

    return config

if __name__ == "__main__":
    config = get_config()
    
    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")