from os import path
from data_lib import *


data_loader = EcgDataLoader("database")
ecgs = data_loader.load_raw_data()

ecgs = np.array([
    MySignal.clean_ecg(e) 
        for e12 in ecgs
            for e in e12
])

ecgs = ecgs.astype(np.float32)
np.save(path.join("data", "ecgs"), ecgs)