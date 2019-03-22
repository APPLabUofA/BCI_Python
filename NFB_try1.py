
"""
@author: S3
Created on Tue Mar  5 11:09:36 2019

"""

from time import time

import numpy as np # Module that simplifies computations on matrices

from scipy.signal import butter, lfilter, lfilter_zi

from pylsl import StreamInlet, resolve_stream # LSL python code


#==============================================================================
#------------------------------------------------------------------------------
#==============================================================================



#//////////////////////////////////////////////////////////////////////////////
# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG', timeout=2)
if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

# Set active EEG stream to inlet and apply time correction
print("Start acquiring data")
inlet = StreamInlet(streams[0], max_chunklen=12) # create a new inlet to read from the stream
eeg_time_correction = inlet.time_correction()


# Get the stream info and description
info = inlet.info()
description = info.desc()
n_channels = info.channel_count()


# Get names of all channels
ch = description.child('channels').first_child()
ch_names = [ch.child_value('label')]
for i in range(1, n_channels):
    ch = ch.next_sibling()
    ch_names.append(ch.child_value('label'))


# Get the sampling frequency
# This is an important value that represents how many EEG data points are
# collected in a second. This influences our frequency band calculation.
fs = int(info.nominal_srate())


#//////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////
""" SET EXPERIMENTAL PARAMETERS """

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
buffer_length = 15

# Length of the epochs used to compute the FFT (in seconds)
epoch_length = 1

# Amount of overlap between two consecutive epochs (in seconds)
overlap_length = 0.8

# Amount to 'shift' the start of each next consecutive epoch
shift_length = epoch_length - overlap_length

# Index of the channel (electrode) to be used
# 0 = left ear
index_channel = [0, 1, 2, 3]

# Name of our channel for plotting purposes
ch_names = [ch_names[i] for i in index_channel]
n_channels = len(index_channel)


#//////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////
""" INITIALIZE BUFFERS """

# Initialize raw EEG data buffer (for plotting)
eeg_buffer = np.zeros((int(fs * buffer_length), n_channels))
filter_state = None  # for use with the notch filter

# Compute the number of epochs in "buffer_length" (used for plotting)
n_win_test = int(np.floor((buffer_length - epoch_length) /
                          shift_length + 1))


# Initialize the plots
plotter_eeg = BCIw.DataPlotter(fs * buffer_length, ch_names, fs)
plotter_feat = BCIw.DataPlotter(n_win_test, feature_names,
                                1 / shift_length)


#//////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////
""" 3. GET DATA """

#------------------------------------------------------------------------------
# The try/except structure allows to quit the while loop by aborting the script with <Ctrl-C>
print('Press Ctrl-C in the console to break the while loop.')

try:


    while True:
        
        """ 3.1 ACQUIRE DATA """
        # get a new sample
#        chunk, timestamps = inlet.pull_chunk()
        # Obtain EEG data from the LSL stream
        eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples = int(shift_length * fs))

        # Only keep the channel we're interested in
        ch_data = np.array(eeg_data)[:, index_channel]

        # Update EEG buffer
        eeg_buffer, filter_state = BCIw.update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)






except KeyboardInterrupt:    
    print('Closing!')
    
    
    
    
    
    
    
    
    