import torch
import torch.nn as nn
class DLinear(nn.Module):
    def __init__(self, input_len, output_len, feature_dim):
        super(DLinear, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.feature_dim = feature_dim
        
        # Two separate linear layers for trend and seasonal components
        self.trend_fc = nn.Linear(input_len, output_len, bias=False)
        self.seasonal_fc = nn.Linear(input_len, output_len, bias=False)

    def forward(self, x):
        """
        x: (batch_size, input_len, feature_dim)
        """
        batch_size, input_len, feature_dim = x.shape

        # Ensure input length matches expected input_len
        assert input_len == self.input_len, f"Expected input length {self.input_len}, but got {input_len}"

        # Decomposition: moving average or identity (simplified here)
        trend_part = x
        seasonal_part = x

        # Apply separate linear layers across feature dimension
        trend_out = self.trend_fc(trend_part.permute(0, 2, 1))  # (batch_size, feature_dim, output_len)
        seasonal_out = self.seasonal_fc(seasonal_part.permute(0, 2, 1))  # (batch_size, feature_dim, output_len)

        # Sum components
        output = trend_out + seasonal_out  # (batch_size, feature_dim, output_len)

        # Permute to match expected output shape (batch_size, output_len, feature_dim)
        return output.permute(0, 2, 1)