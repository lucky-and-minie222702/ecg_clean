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
            # shallow normal conv
            layers.extend([
                nn.Conv1d(in_channels, mid_channels, 1, padding = "same", bias = False),
                nn.BatchNorm1d(mid_channels),
                nn.SiLU(inplace=True),
            ])

        layers.extend([
            # depthwise conv
            nn.Conv1d(mid_channels, mid_channels, kernel_size, stride = stride, bias = False, groups = mid_channels, padding = "same" if stride == 1 else "valid"),
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
        self.p_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride = stride, bias = False, padding = "same" if stride == 1 else "valid"),
        )
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
    
    def __init__(self, in_channels, out_channels, kernel_size, expansion_ratio, downsample = False):
        super(FusedENetBlock, self).__init__()
        mid_channels = in_channels * expansion_ratio
        stride = 1
        if downsample:
            stride = 2

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size, stride = stride, bias = False, padding = "same" if stride == 1 else "valid"),
            nn.BatchNorm1d(mid_channels),
            nn.SiLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size = 1, padding = "same", bias = False),
        )
        
        self.bn_act = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )

        self.use_residual = (stride == 1 and in_channels == out_channels)
        


    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out += x
            out = self.bn_act(out)
            return out
        return out
        
        
    
class CosineContrastiveLoss(nn.Module):
    
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        

    def forward(self, output1, output2):
        batch_size = output1.shape[0]
        
        cos_sim = F.cosine_similarity(output1, output2) 
        # temperature
        cos_sim /= self.temperature
        # softmax normalized
        cos_sim = F.softmax(cos_sim)
        
        label = torch.arange(batch_size)
        loss = F.cross_entropy(cos_sim, label)
        
        return loss.mean()