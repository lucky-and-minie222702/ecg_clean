from model_lib import *
from data_lib import *

# model
class MyModel(nn.Module):
    def __init__(self, resolution, noise_factor):
        super(MyModel, self).__init__()
        self.resolution = resolution
        self.noise_factor = noise_factor
        
        # down sample using conv
        self.ds_conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2),    
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(3, stride = 2),
        )
        
        self.blocks = nn.Sequential(
            FusedENetBlock(32, 32, 5, expansion_ratio = 1),
            FusedENetBlock(32, 48, 3, expansion_ratio = 4, downsample = True),
            FusedENetBlock(48, 64, 3, expansion_ratio = 4),

            EfficientNetBlock(64, 96, 3, expansion_ratio = 4, downsample = True),
            EfficientNetBlock(96, 128, 3, expansion_ratio = 4),

            EfficientNetBlock(128, 160, 3, expansion_ratio = 6, downsample = True),
            EfficientNetBlock(160, 192, 3, expansion_ratio = 6),
            
            EfficientNetBlock(192, 224, 3, expansion_ratio = 6, downsample = True),
            EfficientNetBlock(224, 256, 3, expansion_ratio = 6),
        )

        # time
        self.time_head = nn.Sequential(
            nn.Conv1d(16, 16, 1, bias = False),
            nn.AdaptiveAvgPool1d(resolution),
            nn.Conv1d(16, 1, 1, bias = False),
            nn.Flatten(),
            nn.Softplus(beta = 1, threshold = 20)
        )

        # config
        self.config_head = nn.Sequential(
            nn.Conv1d(16, 16, 1, bias = False),
            nn.AdaptiveAvgPool1d(self.resolution),
            nn.Conv1d(16, 5, 1, bias = False),
        )
        
        # noise
        self.noise_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )         
        
        # base line
        self.baseline_head1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )         
        
        # base line
        self.baseline_head2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )     
        

    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.ds_conv(x)
        x = self.blocks(x)
        
        # high channels streaming
        b_noise = self.noise_head(x)
        b_baseline = self.baseline_head1(x)
        b_baseline_percentage = self.baseline_head2(x)
        
        # low channels streaming (longer sequences)
        x = x.view(batch_size,  16, -1)
        b_t_offset = self.time_head(x)
        b_config = self.config_head(x)
        
        out_clean_ecgs = []
        for i in range(batch_size):
            t_offset = b_t_offset[i].numpy()
            t_offset = np.cumsum(t_offset, axis = -1)
            
            config = b_config[i].numpy()
            config = np.transpose(config, (1, 0))
            
            noise = np.random.normal(0, b_noise[i] / self.noise_factor)
            
            duration = 10
            sampling_rate = 500
            clean_ecg = np.full(duration * sampling_rate, b_baseline[i] * b_baseline_percentage)  # base line
            
            t = np.linspace(0, 10, duration * sampling_rate)
            
            for idx, offset in enumerate(t_offset):
                clean_ecg += ECGGen.generate_beat(t, offset, config[idx])
            clean_ecg += noise
            
            out_clean_ecgs.append(clean_ecg)
            out_clean_ecgs = np.array(out_clean_ecgs)
    
        return torch.tensor(out_clean_ecgs)
    
# ecgs = np.load(path.join("data", "ecgs.npy"))
ecg_loader = EcgDataLoader("database")
ecg = ecg_loader.load_raw_data(100, 101)[0][::, :1:].flatten()
ecg = MySignal.clean_ecg(ecg)
model = MyModel(resolution = 100, noise_factor = 300)
model.eval()
with torch.no_grad():
    output = model(torch.tensor(ecg.copy(), dtype=torch.float).unsqueeze(0).unsqueeze(0))

print(output.squeeze())