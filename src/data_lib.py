import pandas as pd
import numpy as np
from os import path
from scipy import signal
from scipy.stats import norm
from typing import *
import wfdb
import ast

class MySignal:
    def highpass(sig, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, sig)
    
    def lowpass(sig, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, sig)
        	
    def bandpass(sig, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return signal.filtfilt(b, a, sig)
    
    def notch_filter(sig, freq, fs, Q):
        b, a = signal.iirnotch(freq / (fs / 2), Q)
        return signal.filtfilt(b, a, sig)
    
    def clean_ecg(sig):
        sig -= signal.medfilt(sig, 201)  # baseline removal
        sig = MySignal.lowpass(sig, 40, 500, 4)
        return sig
    
class EcgDataLoader:
    def __init__(self, folder):
        self.folder = folder
        self.Y = pd.read_csv(path.join(self.folder, 'ptbxl_database.csv'), index_col = 'ecg_id')
        self.Y.scp_codes = self.Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        
    def load_raw_data(self, start = None, end = None):
        df = self.Y[start:end]
        data = [wfdb.rdsamp(path.join(self.folder, f)) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data
    
def pad_or_crop(x, target_len):
    x = np.asarray(x)
    current_len = len(x)

    if current_len == target_len:
        return x
    elif current_len < target_len:
        # pad 
        return np.pad(x, (0, target_len - current_len), mode='constant')
    else:
        # crop
        return x[:target_len]

class NoiseGen:
    def generate_waveform(
            wave_type = 'sine', 
            amplitude = 1.0, 
            frequency = 1.0, 
            sampling_rate = 1000, 
            duration = 1.0, 
            phase = 0.0):

        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        omega = 2 * np.pi * frequency

        if wave_type == 'sine':
            y = amplitude * np.sin(omega * t + phase)
        elif wave_type == 'cosine':
            y = amplitude * np.cos(omega * t + phase)
        else:
            raise ValueError("wave_type must be 'sine' or 'cosine'")
        
        return t, y
  
    def generate_noise(std, mean = 0.0):
        return np.random.normal(mean, std)