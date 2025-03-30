from model_lib import *
from data_lib import *

# model
class MyModel(nn.Module):
    def __init__(self, quality_factor):
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
            
            EfficientNetBlock(128, 160, 3, expansion_ratio = 6),
            EfficientNetBlock(128, 160, 3, expansion_ratio = 6),
            
            nn.Conv1d(160, 64, 1, bias = False),
            nn.LSTM(64, 64, num_layers = 2, bidirectional = True),
            nn.Conv1d(160, 256, 1, bias = False),
            
            EfficientNetBlock(160, 256, 3, expansion_ratio = 6, downsample = True),
        )

        # config
        self.head1 = nn.Sequential(
            nn.Conv1d(16, 16, 1, bias = False),
            nn.AdaptiveAvgPool1d(quality_factor),
            nn.LSTM(16, 16, num_layers = 2, bidirectional = True),
            nn.Conv1d(32, 5, 1, bias = False),
        )
        
        # time
        self.head2 = nn.Sequential(
            nn.Conv1d(16, 16, 1, bias = False),
            nn.AdaptiveAvgPool1d(quality_factor),
            nn.LSTM(16, 16, num_layers = 2, bidirectional = True),
            nn.Conv1d(32, 1, 1, bias = False),
            nn.Softplus(beta = 1, threshold = 20)
        )
        

    def forward(self, x):
        x = self.ds_conv(x)
        x = self.blocks(x)
        conf = self.head1(x)
        t = self.head2(x)

        return t, conf
        
    