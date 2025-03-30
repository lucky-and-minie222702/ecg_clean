import pandas as pd
import numpy as np
import wfdb
import ast
from os import path
from data_lib import *


def load_raw_data(df, folder):
    data = [wfdb.rdsamp(path.join(folder, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


folder = "database"
Y = pd.read_csv(path.join(folder, 'ptbxl_database.csv'), index_col = 'ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

ecgs = load_raw_data(Y, folder).astype(np.float16)
ecgs = np.transpose(ecgs, (0, 2, 1))
ecgs = ecgs.reshape(-1, 5000)

ecgs = np.array([
    Signal.clean_ecg(e) 
    for e in ecgs
])
ecgs = ecgs
np.save(path.join("data", "ecgs"), ecgs)