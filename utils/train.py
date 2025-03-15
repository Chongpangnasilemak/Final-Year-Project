import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.early_stopper import EarlyStopper
from utils.test import test_step
from datetime import datetime
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
import os 

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
  
    """Trains a PyTorch model for a single epoch.
    
    Args:
        model: PyTorch model.
        dataloader: DataLoader for training data.
        loss_fn: Loss function (e.g., MSELoss).
        optimizer: Optimizer (e.g., Adam).
        device: Device to run the model on (GPU or CPU).

    Returns:
        A tuple of training loss, mean absolute error, mean squared error, and R-squared score.
    """
  
    model.train()
    train_loss = 0.0
    train_smape = 0.0
    all_preds = []
    all_labels = []
    epsilon = 1e-8  # Small value to avoid division by zero
    
    for _, (X, y) in enumerate(dataloader):

                    
            # Move data to the device (GPU/CPU)
            X, y = X.to(device), y.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X)  # Model predictions
            
            # Compute the loss (assuming using MSE for regression)
            loss = loss_fn(y_pred, y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for logging
            train_loss += loss.item()
        
            smape = torch.mean(2 * torch.abs(y - y_pred) / (torch.abs(y) + torch.abs(y_pred) + epsilon)) * 100
            train_smape += smape.item()

            # Convert the outputs and labels to numpy and reshape
            y_pred_flat = y_pred.reshape(-1).cpu().detach().numpy()  # Flatten outputs using reshape
            y_flat = y.reshape(-1).cpu().detach().numpy()   # Flatten labels using reshape

            # Collect flattened predictions and labels for MAE, MSE, and R2
            all_preds.extend(y_pred_flat)
            all_labels.extend(y_flat)

    # Calculate average values for metrics
    avg_train_loss = train_loss / len(dataloader)
    avg_train_smape = train_smape / len(dataloader)
    
    # Calculate MAE, MSE, and R2
    avg_train_mae = mean_absolute_error(all_labels, all_preds)
    avg_train_mse = mean_squared_error(all_labels, all_preds)
    avg_train_r2 = r2_score(all_labels, all_preds)
    
    return avg_train_loss, avg_train_smape, avg_train_mae, avg_train_mse, avg_train_r2


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          shuffle=False,
          model_name: str = "model",  # Allow model name to be passed as a parameter
          save_dir: str = "models/saved_models"
          ):

    save_dir = os.path.join(os.getcwd(), save_dir)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    model_save_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pth")

    log_dir = os.path.join(os.getcwd(), "runs", timestamp)
    os.makedirs(log_dir, exist_ok=True) 
    
    print(f"Model will be saved to: {model_save_path}")
    print(f"Logs will be stored in: {log_dir}")
    
    # Initialize TensorBoard writer with dynamic log directory
    writer = SummaryWriter(log_dir=log_dir)

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_smape": [],
        "train_mae": [],
        "train_mse": [],
        "train_r2": [],
        "test_loss": [],
        "test_smape": [],
        "test_mae": [],
        "test_mse": [],
        "test_r2": []
    }

    early_stopper = EarlyStopper(patience=3, min_delta=1)
    
    for epoch in tqdm(range(epochs)):
        # Train step
        train_loss, train_smape, train_mae, train_mse, train_r2 = train_step(
            model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device
        )

        # Test step
        test_loss, test_smape, test_mae, test_mse, test_r2 = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        # Validation step
        validation_loss = test_step(model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device)

        # Early stopping
        if early_stopper.early_stop(validation_loss[0]):             
            break
        
        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Loss/Validation", validation_loss[0], epoch)

        writer.add_scalar("Metrics/Train_SMAPE", train_smape, epoch)
        writer.add_scalar("Metrics/Train_MAE", train_mae, epoch)
        writer.add_scalar("Metrics/Train_MSE", train_mse, epoch)
        writer.add_scalar("Metrics/Train_R2", train_r2, epoch)

        writer.add_scalar("Metrics/Test_SMAPE", test_smape, epoch)
        writer.add_scalar("Metrics/Test_MAE", test_mae, epoch)
        writer.add_scalar("Metrics/Test_MSE", test_mse, epoch)
        writer.add_scalar("Metrics/Test_R2", test_r2, epoch)

        # Print training progress
        print(
            f"Epoch: {epoch+1} | "
            f"Train Loss: {train_loss:.3f} | Train SMAPE: {train_smape:.3f} | Train MAE: {train_mae:.3f} | Train MSE: {train_mse:.3f} | Train R2: {train_r2:.3f} | "
            f"Test Loss: {test_loss:.3f} | Test SMAPE: {test_smape:.3f} | Test MAE: {test_mae:.3f} | Test MSE: {test_mse:.3f} | Test R2: {test_r2:.3f}"
        )

        # Save results
        results["train_loss"].append(train_loss)
        results["train_smape"].append(train_smape)
        results["train_mae"].append(train_mae)
        results["train_mse"].append(train_mse)
        results["train_r2"].append(train_r2)
        results["test_loss"].append(test_loss)
        results["test_smape"].append(test_smape)
        results["test_mae"].append(test_mae)
        results["test_mse"].append(test_mse)
        results["test_r2"].append(test_r2)

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    # Close the TensorBoard writer
    writer.close()

    return results

