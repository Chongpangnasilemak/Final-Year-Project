import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.PatchTST_backbone import Patching,TSTiEncoder,FlattenHead

class PatchTST(nn.Module):
    def __init__(self,in_channels, pred_len, patch_size, stride, d_model):
        super().__init__()
        self.revin = RevIN(in_channels)
        self.patching = Patching(patch_size, stride)
        self.encoder = TSTiEncoder(patch_size, d_model)
        self.flatten = FlattenHead(pred_len)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.revin(x)
        x = self.patching(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.revin(x, denorm=True)
        x = x.permute(0, 2, 1)
        
        return x