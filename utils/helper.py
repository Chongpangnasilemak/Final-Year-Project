import torch
import torch.nn as nn
from models.DLinear import DLinear
from models.PatchTST import PatchTST
from models.XCPatchTST import XCPatchTST

def create_crosschannel_patchtst(device:torch.nn.Module, in_channels:int, pred_len:int, patch_size:int, stride:int, d_model:int, kernel_size:int):
    model = XCPatchTST(in_channels=in_channels-1, pred_len=pred_len, patch_size=patch_size, stride=stride, d_model=d_model, kernel_size=kernel_size)
    model.name = "Cross-Channel PatchTST"
    print(f"[INFO] Created {model.name} model with the following parameters:")
    print(f"       - in_channels: {in_channels}")
    print(f"       - pred_len: {pred_len}")
    print(f"       - patch_size: {patch_size}")
    print(f"       - stride: {stride}")
    print(f"       - d_model: {d_model}")
    print(f"       - kernel_size: {kernel_size}")
    model.to(device)
    return model

def create_patchtst(device:torch.nn.Module, in_channels:int, pred_len:int, patch_size:int, stride:int, d_model:int):
    model = PatchTST(in_channels=in_channels-1, pred_len=pred_len, patch_size=patch_size, stride=stride, d_model=d_model)
    model.name = "PatchTST"
    print(f"[INFO] Created new {model.name} model.")
    print(f"[INFO] Created {model.name} model with the following parameters:")
    print(f"       - in_channels: {in_channels}")
    print(f"       - pred_len: {pred_len}")
    print(f"       - patch_size: {patch_size}")
    print(f"       - stride: {stride}")
    print(f"       - d_model: {d_model}")
    model.to(device)
    return model

def create_dlinear(device:torch.nn.Module, input_len:int, output_len:int, feature_dim:int):
    model = DLinear(input_len, output_len, feature_dim)
    model.name = "DLinear"
    print(f"[INFO] Created new {model.name} model.")
    print(f"[INFO] Created {model.name} model with the following parameters:")
    print(f"       - input_len: {input_len}")
    print(f"       - feature_dim: {feature_dim}")
    print(f"       - output_len: {output_len}")
    model.to(device)
    return model