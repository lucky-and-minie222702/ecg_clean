from typing import *
from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split, TensorDataset

# conv = convolution
# bn = batch norm
# act = activation
# ln = linear
# do = dropout
# att = attention
# lstm = LSTM
# em = embedding
# pool = pooling

def get_padding(kernel_size, stride = 2, dilation = 1):
    effective_kernel = dilation * (kernel_size - 1) + 1
    padding = (stride - 1 + effective_kernel - 1) // 2
    return padding

class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels //  reduction_ratio, bias = False),
            nn.SiLU(inplace=True),
            nn.Linear(channels //  reduction_ratio, channels, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        shape = x.size()
        # batch_size, channels 
        b, c = shape[:2:]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class EfficientNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expansion_ratio, use_residual = True, se_reduction_ratio = 4, downsample = False):
        super(EfficientNetBlock, self).__init__()
        mid_channels = in_channels * expansion_ratio
        self.use_residual = use_residual
        self.downsample = downsample
        stride = 1
        if downsample:
            stride = 2
            
        layers = []
        if expansion_ratio > 1:
            layers.extend([
                nn.Conv1d(in_channels, mid_channels, 1, padding = "same", bias = False),
                nn.BatchNorm1d(mid_channels),
                nn.SiLU(inplace=True),
            ])

        layers.extend([
            # depthwise conv
            nn.Conv1d(mid_channels, mid_channels, kernel_size, stride = stride, bias = False, groups = mid_channels, padding = get_padding(kernel_size, stride)),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU(inplace=True),
            # squeeze and excite
            SEBlock(mid_channels, reduction_ratio = se_reduction_ratio),
            # connection + projection conv
            nn.Conv1d(mid_channels, out_channels, 1, padding = "same", bias = False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True),
        ])
        
        self.main_conv = nn.Sequential(*layers)
        
        # project to match output
        self.p_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride = stride, bias = False, padding = get_padding(kernel_size, stride))

        self.p_bn_act = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
        )
        
    def forward(self, x):
        identity = x
        x = self.main_conv(x)
        if self.use_residual:
            # projection
            identity = self.p_conv(identity)
            identity = self.p_bn_act(identity)
            return x + identity
        return x


class TransposeEfficientNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expansion_ratio, use_residual = True, se_reduction_ratio = 4):
        super(TransposeEfficientNetBlock, self).__init__()
        mid_channels = in_channels * expansion_ratio
        self.use_residual = use_residual
            
        layers = []
        if expansion_ratio > 1:
            # shallow normal conv
            layers.extend([
                nn.Conv1d(in_channels, mid_channels, 1, padding = "same", bias = False),
                nn.BatchNorm1d(mid_channels),
                nn.SiLU(inplace=True),
            ])

        layers.extend([
            # depthwise conv
            nn.ConvTranspose1d(mid_channels, mid_channels, kernel_size, stride = 2, bias = False, groups = mid_channels, padding = get_padding(kernel_size, 2)),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU(inplace=True),
            # squeeze and excite
            SEBlock(mid_channels, reduction_ratio = se_reduction_ratio),
            # connection + projection conv
            nn.Conv1d(mid_channels, out_channels, 1, padding = "same", bias = False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True),
        ])
        
        self.main_conv = nn.Sequential(*layers)
        
        # project to match output
        self.p_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride = 2, bias = False, padding = get_padding(kernel_size, 2))

        self.p_bn_act = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
        )
        
    def forward(self, x):
        identity = x
        x = self.main_conv(x)
        if self.use_residual:
            # projection
            identity = self.p_conv(identity)
            identity = self.p_bn_act(identity)
            return x + identity
        return x

        
class FusedENetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expansion_ratio, use_residual, downsample = False):
        super(FusedENetBlock, self).__init__()
        mid_channels = in_channels * expansion_ratio
        self.use_residual = use_residual
        stride = 1
        if downsample:
            stride = 2

        layers = [
            nn.Conv1d(in_channels, mid_channels, kernel_size, stride = stride, bias = False, padding = get_padding(kernel_size, stride)),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU(),
        ]
        if expansion_ratio > 1:
            layers.append(nn.Conv1d(mid_channels, out_channels, kernel_size = 1, padding = "same", bias = False))

        self.blocks = nn.Sequential(*layers)
        
        self.bn_act = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )
        
        # projection
        self.proj = nn.Conv1d(in_channels, out_channels, 1 if not downsample else kernel_size, stride = stride, padding = get_padding(kernel_size, stride))
        
    def forward(self, x):
        out = self.blocks(x)
        if self.use_residual:
            out += self.proj(x)
            out = self.bn_act(out)
            return out
        return out

class Visualization:
    def format_number_short(n):
        if n < 1_000:
            return str(n)
        elif n < 1_000_000:
            return f"{n / 1_000:.1f}k".rstrip('0').rstrip('.')
        elif n < 1_000_000_000:
            return f"{n / 1_000_000:.1f}m".rstrip('0').rstrip('.')
        elif n < 1_000_000_000_000:
            return f"{n / 1_000_000_000:.1f}b".rstrip('0').rstrip('.')
        else:
            return f"{n / 1_000_000_000_000:.1f}t".rstrip('0').rstrip('.')
        
    def convert_bytes(byte_size: int) -> str:
        units = ["bytes", "KB", "MB", "GB", "TB", "PB", "EB"]
        size = byte_size
        unit_index = 0
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        return f"{size:.2f} {units[unit_index]}"
        
    def count_param(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def show_param(model):
        total, trainable = Visualization.count_param(model)
        print(f"Total: {Visualization.format_number_short(total)}, trainable: {Visualization.format_number_short(trainable)}")