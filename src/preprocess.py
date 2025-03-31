from os import path
from data_lib import *



data_loader = EcgDataLoader("database")

ecgs = data_loader.load_raw_data()
ecgs = np.transpose(ecgs, (0, 2, 1))
ecgs = ecgs.reshape(-1, 5000)

ecgs = np.array([
    Signal.clean_ecg(e) 
    for e in ecgs
])
ecgs = ecgs
np.save(path.join("data", "ecgs"), ecgs)