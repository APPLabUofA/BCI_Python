# -*- coding: utf-8 -*-
"""
Auxiliary Tools for Auditory NFB

Created on Sat Mar 23 14:15:36 2019

@author: ssshe


Adapted from https://github.com/NeuroTechX/bci-workshop

"""

# Most code comes from bci_workshop_tools


import matplotlib.pyplot as plt

import numpy as np

from scipy.signal import butter, lfilter, lfilter_zi

from IPython import get_ipython




# Notch filter, but not sure what it is fitering
NOTCH_B, NOTCH_A = butter(4, np.array([55, 65])/(256/2), btype='bandstop')


#==============================================================================
def remove_dc_offset(data_epoch, fs):
       
    hp_cutoff_Hz = 1.0

    b, a = butter(2, hp_cutoff_Hz/(fs / 2.0), 'highpass')
    data_epoch = lfilter(b, a, data_epoch, 0)
    
    return data_epoch
        
#==============================================================================
def mastoid_Reref(ch_names, n_chEEG, data_epoch):
   """Re-reference to average mastoid.
    """ 
   ref_idx = int(ch_names.index('M2'))
   data_epoch_new = data_epoch
        
   for i in range(0, n_chEEG): 
       data_epoch_new[:,i] =  data_epoch[:,i]  -  data_epoch[:,ref_idx] * .5  
    
   return data_epoch_new 

#==============================================================================
def GrattonEmcpRaw(ch_names, n_chEEG, data_epoch):
   """Gratton method to regress out EOG activity from brain data.
       - needs to be applied separately for each EOG channel if want to correct
         for both horizontal and vertical EOGs  
    
    Args:
        n_chEEG: number of EEG channels
        data_epoch (numpy.ndarray): array of dimension [number of samples,
                number of channels]
    
    Returns:
        data_epoch (numpy.ndarray): feature matrix of shape [number of feature points,
            number of different features]
    """
   
   # VEOG (eyeblink) correction
   raw_eeg = data_epoch[:,:n_chEEG] #get EEG data only
   raw_eeg = raw_eeg.T
#   eog_idx = [int(ch_names.index('HEOG')),int(ch_names.index('VEOG'))]
   eog_idx = int(ch_names.index('VEOG')) #for eyeblink correction
   raw_eog = np.zeros((data_epoch.shape[0],1)) #pre-allocate
   raw_eog[:,0] = data_epoch[:,eog_idx]
   raw_eog = raw_eog.T
   # Calculate beta values
   beta = np.linalg.solve(np.dot(raw_eog,raw_eog.T), np.dot(raw_eog,raw_eeg.T))
   eeg_corrected = (raw_eeg.T - np.dot(raw_eog.T,beta)).T
   data_epoch[:,:n_chEEG] = eeg_corrected.T #replace w/corrected data
   
   # HEOG (eye movemrny) correction
   raw_eegH = data_epoch[:,:n_chEEG] #get EEG data only
   raw_eegH = raw_eegH.T
   eog_idxH = int(ch_names.index('HEOG')) #for eyeblink correction
   raw_eogH = np.zeros((data_epoch.shape[0],1)) #pre-allocate
   raw_eogH[:,0] = data_epoch[:,eog_idxH]
   raw_eogH = raw_eogH.T
   # Calculate beta values
   betaH = np.linalg.solve(np.dot(raw_eogH,raw_eogH.T), np.dot(raw_eogH,raw_eegH.T))
   eeg_correctedH = (raw_eegH.T - np.dot(raw_eogH.T,betaH)).T
   data_epoch[:,:n_chEEG] = eeg_correctedH.T #replace w/corrected data
    
   
   return data_epoch


#==============================================================================
def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n

#==============================================================================
def compute_feature_vector(eegdata, fs):
    """Extract the features from the EEG.

    Args:
        eegdata (numpy.ndarray): array of dimension [number of samples,
                number of channels]
        fs (float): sampling frequency of eegdata

    Returns:
        (numpy.ndarray): feature matrix of shape [number of feature points,
            number of different features]
    """
    # 1. Compute the PSD
    winSampleLength, nbCh = eegdata.shape

    # Apply Hamming window
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
    dataWinCenteredHam = (dataWinCentered.T*w).T

    NFFT = nextpow2(winSampleLength) #defined below
    
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0)/winSampleLength
    PSD = 2*np.abs(Y[0:int(NFFT/2), :])
    f = fs/2*np.linspace(0, 1, int(NFFT/2))

    # SPECTRAL FEATURES
    # Average of band powers
    # Delta <3
    ind_delta, = np.where(f < 3)
    meanDelta = np.mean(PSD[ind_delta, :], axis=0)
    # Theta 3-8
    ind_theta, = np.where((f >= 3) & (f <= 8))
    meanTheta = np.mean(PSD[ind_theta, :], axis=0)
    # Alpha 8-14
    ind_alpha, = np.where((f >= 8) & (f <= 14))
    meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
    # Beta 14-30
    ind_beta, = np.where((f >= 14) & (f < 30))
    meanBeta = np.mean(PSD[ind_beta, :], axis=0)

    feature_vector = np.concatenate((meanDelta, meanTheta, meanAlpha,
                                     meanBeta), axis=0)

    feature_vector = np.log10(feature_vector)

    return feature_vector



#==============================================================================
def get_feature_names(ch_names):
    """Generate the name of the features.

    Args:
        ch_names (list): electrode names

    Returns:
        (list): feature names
    """
    bands = ['delta', 'theta', 'alpha', 'beta']

    feat_names = []
    for band in bands:
        for ch in range(len(ch_names)):
            feat_names.append(band + '-' + ch_names[ch])

    return feat_names


#==============================================================================
def update_buffer(data_buffer, new_data, notch=False, filter_state=None):
    """
    Concatenates "new_data" into "data_buffer", and returns an array with
    the same size as "data_buffer"
    """
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    if notch:
        if filter_state is None:
            filter_state = np.tile(lfilter_zi(NOTCH_B, NOTCH_A),
                                   (data_buffer.shape[1], 1)).T
        new_data, filter_state = lfilter(NOTCH_B, NOTCH_A, new_data, axis=0,
                                         zi=filter_state)

    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return new_buffer, filter_state

#==============================================================================
def get_last_data(data_buffer, newest_samples):
    """
    Obtains from "buffer_array" the "newest samples" (N rows from the
    bottom of the buffer)
    """
    new_buffer = data_buffer[int((data_buffer.shape[0] - newest_samples)):, :]

    return new_buffer

#==============================================================================
class DataPlotter():
    """
    Class for creating and updating a line plot.
    """

    def __init__(self, nbPoints, chNames, fs=None, title=None):
        """Initialize the figure."""

        self.nbPoints = nbPoints
        self.chNames = chNames
        self.nbCh = len(self.chNames)

        self.fs = 1 if fs is None else fs
        self.figTitle = '' if title is None else title

        data = np.empty((self.nbPoints, 1))*np.nan
        self.t = np.arange(data.shape[0])/float(self.fs)

        # Create offset parameters for plotting multiple signals
        self.yAxisRange = 100
        self.chRange = self.yAxisRange/float(self.nbCh)
        self.offsets = np.round((np.arange(self.nbCh)+0.5)*(self.chRange))

        # Create the figure and axis
#        %matplotlib qt # plots in their own window
#        get_ipython().run_line_magic('matplotlib', 'qt5') # plots in their own window
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_yticks(self.offsets)
        self.ax.set_yticklabels(self.chNames)

        # Initialize the figure
        self.ax.set_title(self.figTitle)

        self.chLinesDict = {}
        for i, chName in enumerate(self.chNames):
            self.chLinesDict[chName], = self.ax.plot(
                    self.t, data+self.offsets[i], label=chName)

        self.ax.set_xlabel('Time')
        self.ax.set_ylim([0, self.yAxisRange])
        self.ax.set_xlim([np.min(self.t), np.max(self.t)])

        plt.show()

    def update_plot(self, data):
        """ Update the plot """

        data = data - np.mean(data, axis=0)
        std_data = np.std(data, axis=0)
        std_data[np.where(std_data == 0)] = 1
        data = data/std_data*self.chRange/5.0

        for i, chName in enumerate(self.chNames):
            self.chLinesDict[chName].set_ydata(data[:, i] + self.offsets[i])

        self.fig.canvas.draw()

    def clear(self):
        """ Clear the figure """

        blankData = np.empty((self.nbPoints, 1))*np.nan

        for i, chName in enumerate(self.chNames):
            self.chLinesDict[chName].set_ydata(blankData)

        self.fig.canvas.draw()

    def close(self):
        """ Close the figure """

        plt.close(self.fig)

#==============================================================================















