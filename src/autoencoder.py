from model_lib import *
from data_lib import *


# model
class BuildEncoder(nn.Module):
    def __init__(self):
        super(BuildEncoder, self).__init__()
        
        self.embedd = nn.Sequential(
            nn.Conv1d(1, 24, 11, stride = 2, bias = False),
            nn.BatchNorm1d(24),
            nn.SiLU(inplace = True),
            nn.MaxPool1d(3, stride = 2),
        )
        
        self.blocks = nn.Sequential(
            FusedENetBlock(24, 24, 7, expansion_ratio = 1, use_residual = False),
            FusedENetBlock(24, 48, 3, expansion_ratio = 4, use_residual = True),
            FusedENetBlock(48, 64, 3, expansion_ratio = 4, use_residual = True, downsample = True),
            
            EfficientNetBlock(64, 64, 3, expansion_ratio = 4),
            EfficientNetBlock(64, 64, 3, expansion_ratio = 4),
            EfficientNetBlock(64, 128, 3, expansion_ratio = 4),
            
            EfficientNetBlock(128, 128, 3, expansion_ratio = 6, downsample = True),
            EfficientNetBlock(128, 128, 3, expansion_ratio = 6),
            EfficientNetBlock(128, 128, 3, expansion_ratio = 6),
            EfficientNetBlock(128, 128, 3, expansion_ratio = 6),
            EfficientNetBlock(128, 160, 3, expansion_ratio = 6),
            
            EfficientNetBlock(160, 160, 3, expansion_ratio = 6, downsample = True),
            EfficientNetBlock(160, 160, 3, expansion_ratio = 6),
            EfficientNetBlock(160, 160, 3, expansion_ratio = 6),
            EfficientNetBlock(160, 160, 3, expansion_ratio = 6),
            EfficientNetBlock(160, 160, 3, expansion_ratio = 6),
            EfficientNetBlock(160, 160, 3, expansion_ratio = 6),
            EfficientNetBlock(160, 256, 3, expansion_ratio = 6),
        )
        
        # projection head
        self.proj_head = nn.Sequential(
            nn.Conv1d(256, 256, 1, bias = False),
            nn.BatchNorm1d(256),
        )
        
    def forward(self, x):
        x = self.embedd(x)
        x = self.blocks(x)
        x = self.proj_head(x)
        return x
    
class BuildDecoder(nn.Module):
    def __init__(self):
        super(BuildDecoder, self).__init__()

        self.blocks = nn.Sequential(
            TransposeEfficientNetBlock(256, 160, 3, expansion_ratio = 6),
            EfficientNetBlock(160, 160, 3, expansion_ratio = 6),
            EfficientNetBlock(160, 160, 3, expansion_ratio = 6),
            
            TransposeEfficientNetBlock(160, 128, 3, expansion_ratio = 6),
            EfficientNetBlock(128, 128, 3, expansion_ratio = 6),
            EfficientNetBlock(128, 128, 3, expansion_ratio = 6),
            EfficientNetBlock(128, 128, 3, expansion_ratio = 6),
            EfficientNetBlock(128, 128, 3, expansion_ratio = 6),
            
            TransposeEfficientNetBlock(128, 64, 3, expansion_ratio = 4),
            EfficientNetBlock(64, 64, 3, expansion_ratio = 4),
            EfficientNetBlock(64, 64, 3, expansion_ratio = 4),
            EfficientNetBlock(64, 64, 3, expansion_ratio = 4),
            EfficientNetBlock(64, 64, 3, expansion_ratio = 4),
            EfficientNetBlock(64, 64, 3, expansion_ratio = 4),
            EfficientNetBlock(64, 64, 3, expansion_ratio = 4),
            
            TransposeEfficientNetBlock(64, 48, 3, expansion_ratio = 1),
            TransposeEfficientNetBlock(48, 24, 5, expansion_ratio = 1),
        )

        # projection head
        self.proj_head = nn.Sequential(
            nn.ConvTranspose1d(24, 12, 7, bias = False),
            nn.BatchNorm1d(12),
            nn.SiLU(),
            
            nn.ConvTranspose1d(12, 12, 4, bias = False),
            nn.BatchNorm1d(12),
        )
        
    def forward(self, x):
        # x = self.embedd(x)
        x = self.blocks(x)
        # x = self.proj_head(x)
        return x


encoder = BuildEncoder()
decoder = BuildDecoder()
Visualization.show_param(encoder)
Visualization.show_param(decoder)

ecg = EcgDataLoader("database")
ecg = ecg.load_raw_data(100, 101)[0][::, :1:].flatten()
ecg = MySignal.clean_ecg(ecg)

with torch.no_grad():
    x = torch.tensor(ecg.copy(), dtype = torch.float).unsqueeze(0).unsqueeze(0)
    x = encoder(x)
    print(x.shape)
    x = decoder(x)
    print(x.shape)