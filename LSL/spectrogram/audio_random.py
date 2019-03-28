from __future__ import print_function

import random
import numpy as np
import scipy as sp
from scipy.integrate import simps
import time

NUM_SAMPLES = 1024
SAMPLING_RATE = 44100
MAX_FREQ = SAMPLING_RATE / 2
FREQ_SAMPLES = NUM_SAMPLES / 8
TIMESLICE = 100  # ms
NUM_BINS = 16

data = {'values': None}


def get_audio_data():
    
    while True:
        try:
            q=np.arange(NUM_SAMPLES)
            raw_data=(0.02*np.sin(q))
            noise = np.random.normal(0,1,NUM_SAMPLES)
            signal = raw_data + noise*0.01
            fft = sp.fft(signal)
            spectrum = abs(fft)[:NUM_SAMPLES/2]
            power = spectrum**2
            bins = [simps(a) for a in np.split(power, NUM_BINS)]
            data['values'] = signal, spectrum, bins
            return (data['values'])
            
        except:
            continue
