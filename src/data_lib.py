import pandas as pd
import numpy as np
from os import path
from scipy import signal
from scipy.stats import norm
from typing import *

class Signal:
    
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
        sig = Signal.bandpass(sig, 0.5, 40, 500, 4)
        sig = Signal.notch_filter(sig, 50, 500, 30)
        return sig
    

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


class ECGGen:
    
    def skewed_gaussian(t, mu, sigma, alpha):
        # time, center, width, skewness
        phi = norm.pdf((t - mu) / sigma)
        Phi = norm.cdf(alpha * (t - mu) / sigma)
        return 2 * phi * Phi


    def generate_beat(t, offset, config, noise = None, modify_noise = False):
        y = np.zeros_like(t)
        beat_duration = config[0]

        # scale 0 -> 1 (percentage)
        t_rel = np.linspace(0, 1, np.sum((t >= offset) & (t < offset + beat_duration)), endpoint = False)

        # mu, sigma need to between 0 - 1
        beat = config[1] * ECGGen.skewed_gaussian(
            t_rel, mu = config[2], sigma = [3], alpha = config[4]
        )
        
        mask = (t >= offset) & (t < offset + beat_duration)
        if noise is not None:
            if modify_noise:
                noise = pad_or_crop(noise, len(beat))
            beat += noise
        y[mask] = beat
        return y