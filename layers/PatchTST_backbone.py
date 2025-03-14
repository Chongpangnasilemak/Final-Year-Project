import torch
import torch.nn as nn

class Patching(nn.Module):
    def __init__(self, patch_size, stride):
        super(Patching, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
    
    def forward(self, x):
        batch_size, channels, seq_len = x.shape  # Expecting input of shape [Batch, Channel, Sequence Length]
        
        # Calculate output size and required padding
        output_size = (seq_len - self.patch_size) // self.stride + 1
        padding = (output_size - 1) * self.stride + self.patch_size - seq_len
        
        # Apply padding to the input tensor
        x_padded = torch.nn.functional.pad(x, (0, padding), mode='constant', value=0)
        
        # Apply unfold operation
        unfolded_tensor = x_padded.unfold(2, self.patch_size, self.stride)  # Shape: [Batch, Channel, Patch, Patch Data]
        
        # Permute to get the shape [Batch, Channel, Patch Data, Patch]
        permuted_tensor = unfolded_tensor.permute(0, 1, 3, 2)
        
        return permuted_tensor

class TSTiEncoder(nn.Module):
    def __init__(self, patch_data_size, d_model, n_heads=8, num_layers=2, dropout=0.1, max_len=500):
        super().__init__()
        self.linear_projection = nn.Linear(patch_data_size, d_model)  # Step 2: Linear projection
        self.dropout = nn.Dropout(dropout)

        # Positional encoding (learnable parameters) initialization
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))  # Learnable positional encoding

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Step 1: Permute to [Batch, Channel, Patch, Patch Data]
        x = x.permute(0, 1, 3, 2)  

        # Step 2: Linear projection
        x = self.linear_projection(x)  
        # Step 3: Reshape for transformer input: [Batch * Channel, Patch, Patch Data]
        batch_size, channels, patches, d_model = x.shape
        u = x.view(batch_size * channels, patches, d_model)  

        # Step 4: Add positional encoding
        u = u + self.positional_encoding[:, :patches, :]  # Adding positional encoding to the input

        # Step 5: Apply dropout
        u = self.dropout(u)

        # Step 6: Transformer Encoder
        z = self.transformer_encoder(u)  # Output shape: [Batch * Channel, Patch, Transformer Encoding]

        # Step 7: Reshape back to [Batch, Channel, Patch, Transformer Encoding]
        z = z.view(batch_size, channels, patches, d_model)

        # Step 8: Permute to [Batch, Channel, Transformer Encoding, Patch]
        z = z.permute(0, 1, 3, 2)

        return z

class FlattenHead(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        
        # Apply Adaptive Average Pooling to reduce from [128, 12] to [pred_len]
        self.pool = nn.AdaptiveAvgPool2d((pred_len,1))  # Target shape: (pred_len, 1)
    
    def forward(self, x):
        # Pooling across the 128 dimension and collapse 12 to 1
        x = self.pool(x)
        x = x.squeeze(-1)  # Remove the last dimension (12 -> 1)
        return x

class CrossChannelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(CrossChannelConv, self).__init__()
        # A convolutional layer that spans across multiple channels
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x):
        # Assuming x shape is [Batch, Channels, Sequence Length]
        return self.conv(x)







        
