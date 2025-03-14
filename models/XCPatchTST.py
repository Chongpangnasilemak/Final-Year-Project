import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.PatchTST_backbone import Patching,TSTiEncoder,FlattenHead, CrossChannelConv
class XCPatchTST(nn.Module):
    def __init__(self,in_channels, pred_len, patch_size, stride, d_model,kernel_size):
        super().__init__()
        self.revin = RevIN(in_channels)
        self.cross_channel_conv = CrossChannelConv(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size)
        self.patching = Patching(patch_size, stride)
        self.encoder = TSTiEncoder(patch_size, d_model)
        self.flatten = FlattenHead(pred_len)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.revin(x)
        x = self.cross_channel_conv(x)  # Cross-channel convolution operation
        x = self.patching(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.revin(x, denorm=True)
        x = x.permute(0, 2, 1)
        return x