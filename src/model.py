from model_lib import *
from data_lib import *

# model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel).__init__()
        
        # down sample using conv
        self.ds_conv = nn.Sequential([
            nn.Conv1d(12, 32, 7, stride = 2),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace = True),
            nn.MaxPool1d(3, stride = 2)
        ])
        
        self.blocks = nn.Sequential(
            FusedENetBlock(32, 32, 5),
            FusedENetBlock(32, 48, 3, expansion_ratio = 4, downsample = True),
            FusedENetBlock(48, 64, 3, expansion_ratio = 4, downsample = True),

            EfficientNetBlock(64, 128, 3, expansion_ratio = 4,downsample = True),
            EfficientNetBlock(64, 128, 3, expansion_ratio = 4,downsample = True),
            EfficientNetBlock(64, 128, 3, expansion_ratio = 4,downsample = True),
            
            EfficientNetBlock(128, 160, 3, expansion_ratio = 6),
            EfficientNetBlock(128, 160, 3, expansion_ratio = 6),
            
            EfficientNetBlock(160, 256, 3, expansion_ratio = 6, downsample = True),
            EfficientNetBlock(160, 256, 3, expansion_ratio = 6, downsample = True),
        )
        
        vec_out = 1024
        self.head = nn.Sequential(
            nn.Conv1d(256, vec_out, 1, bias = False),
            nn.BatchNorm1d(vec_out),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )