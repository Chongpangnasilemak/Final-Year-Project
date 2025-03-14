import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
  
    """Tests a PyTorch model for a single epoch (Regression).
    
    Args:
        model: PyTorch model.
        dataloader: DataLoader for test data.
        loss_fn: Loss function (e.g., MSELoss).
        device: Device to run the model on (GPU or CPU).

    Returns:
        A tuple of testing loss, mean absolute error, mean squared error, and R-squared score.
    """
  
    model.eval() 
    val_loss = 0.0
    val_smape = 0.0
    all_preds = []
    all_labels = []
    epsilon = 1e-8
    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()

            smape = torch.mean(2 * torch.abs(y - y_pred) / (torch.abs(y) + torch.abs(y_pred) + epsilon)) * 100
            val_smape += smape.item()

            # Convert the outputs and labels to numpy and reshape
            y_pred_flat = y_pred.reshape(-1).cpu().numpy()  # Flatten outputs using reshape
            y_flat = y.reshape(-1).cpu().numpy()  # Flatten labels using reshape

            # Collect flattened predictions and labels for MAE, MSE, and R2
            all_preds.extend(y_pred_flat)
            all_labels.extend(y_flat)

    # Calculate average values for metrics
    avg_val_loss = val_loss / len(dataloader)
    avg_val_smape = val_smape / len(dataloader) 
    
    # Calculate MAE, MSE, and R2
    avg_val_mae = mean_absolute_error(all_labels, all_preds)
    avg_val_mse = mean_squared_error(all_labels, all_preds)
    avg_val_r2 = r2_score(all_labels, all_preds)


    return avg_val_loss, avg_val_smape, avg_val_mae, avg_val_mse, avg_val_r2