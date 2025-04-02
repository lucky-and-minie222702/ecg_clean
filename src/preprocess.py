from os import path
from data_lib import *


data_loader = EcgDataLoader("database")
ecgs = data_loader.load_raw_data(1, 2)
ecgs = np.transpose(ecgs, (0, 2, 1))

ecgs = np.array([
   [MySignal.clean_ecg(e) for e in e12] for e12 in ecgs
])

# ecgs = ecgs.astype(np.float32)
# np.save(path.join("data", "ecgs"), ecgs)