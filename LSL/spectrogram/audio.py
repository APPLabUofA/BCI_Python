from __future__ import print_function
from pylsl import StreamInlet, resolve_stream

import numpy as np
import scipy as sp
from scipy.integrate import simps

NUM_SAMPLES = 512
SAMPLING_RATE = 2000
MAX_FREQ = SAMPLING_RATE / 2
FREQ_SAMPLES = NUM_SAMPLES / 8
TIMESLICE = 100  # ms
NUM_BINS = 16

data = {'values': None}

def _get_audio_data():
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])

    while True:
        try:
            raw_data = np.empty(0, np.int16)
            while len(raw_data) < NUM_SAMPLES:
                sample = inlet.pull_sample()
                raw_data.append(sample[0])

            signal = raw_data / 32768.0
            fft = sp.fft(signal)
            spectrum = abs(fft)[:NUM_SAMPLES/2]
            power = spectrum**2
            bins = simps(np.split(power, NUM_BINS))
            data['values'] = signal, spectrum, bins
        except:
            continue
