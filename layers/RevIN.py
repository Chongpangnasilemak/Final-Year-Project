import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class RevIN(nn.Module):
    def __init__(self,in_channels, affine=True):
        super().__init__()
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, in_channels))  # Scale factor
            self.beta = nn.Parameter(torch.zeros(1, 1, in_channels))  # Shift factor
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x, denorm=False, mean=None, std=None):
        x = x.permute(0, 2, 1)  # Convert to [Batch, Steps, Channels]
        
        if not denorm:  # Normalization
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + 1e-5
            x = (x - mean) / std
            if self.affine:
                x = x * self.gamma + self.beta  # Apply learnable transformation
            self._saved_mean = mean  # Store for later denorm
            self._saved_std = std
        else:  # Denormalization
            mean = self._saved_mean
            std = self._saved_std
            if self.affine:
                x = (x - self.beta) / (self.gamma + 1e-5)  # Reverse affine transformation
            x = (x * std) + mean  # Reverse normalization

        x = x.permute(0, 2, 1)  # Convert back to [Batch, Channels, Steps]
        return x  # Always return only the tensor
    
    def visualize_distribution(self, x):
        x = x.permute(0, 2, 1).detach().cpu().numpy()
        plt.figure(figsize=(10, 5))
        plt.hist(x.flatten(), bins=50, alpha=0.6, label='Original')
        normalized_x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-5)
        plt.hist(normalized_x.flatten(), bins=50, alpha=0.6, label='Normalized')
        plt.legend()
        plt.title("Effect of RevIN Normalization")
        plt.show()