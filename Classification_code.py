import scipy.io 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
 
m = scipy.io.loadmat(r'/content/drive/MyDrive/BCICIV_1_mat/BCICIV_calib_ds1d.mat',struct_as_record = True)
 
sample_rate = m['nfo']['fs'][0][0][0][0]
EEG = m['cnt'].T
nchannels, nsamples = EEG.shape
 
channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
event_onsets = m['mrk'][0][0][0]
event_codes = m['mrk'][0][0][1]
 
labels = np.zeros((1,nsamples),int)
labels[0, event_onsets] = event_codes
 
cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
cl1 = cl_lab[0]
cl2 = cl_lab[1]
 
nclasses = len(cl_lab)
nevents = len(event_onsets)
 
print('shape of EEG: ',EEG.shape)
print('sample rate: ',sample_rate)
print('Number of channels: ',nchannels)
print('Channel names: ',channel_names)
print('Number of Events: ',len(event_onsets))
print('Event codes: ',np.unique(event_codes))
print('Class labels: ',cl_lab)
print('Number of classes: ',nclasses)
 
 
 
trials = {}
 
win = np.arange(int(0.5*sample_rate),int(2.5*sample_rate))
 
nsamples = len(win)
 
for cl, code in zip(cl_lab, np.unique(event_codes)):
 
    cl_onsets = event_onsets[event_codes == code]
 
    trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))
 
    for i,onset in enumerate(cl_onsets):
        trials[cl][:,:,i] = EEG[:, win+onset]
 
print('Shape of trials[cl1]: ',trials[cl1].shape)
print('Shape of trials[cl2]: ',trials[cl2].shape)
 
def psd(trials):
    ntrials = trials.shape[2]
    trials_PSD = np.zeros((nchannels, 101, ntrials))
 
    for trial in range(ntrials):
        for ch in range(nchannels):
 
            (PSD,freqs) = mlab.psd(trials[ch,:,trial],NFFT=int(nsamples),Fs=sample_rate)
            trials_PSD[ch,:,trial] = PSD.ravel()
 
    return trials_PSD,freqs
 
psd_r, freqs = psd(trials[cl1])
psd_f, freqs = psd(trials[cl2])
trials_PSD = {cl1: psd_r, cl2: psd_f}
 
def plot_psd(trials_PSD, freqs, chan_ind, chan_lab = None, maxy = None):
 
    plt.figure(figsize = (12,5))
    nchans = len(chan_ind)
 
    nrows = np.ceil(nchans/3)
    ncols = min(3,nchans)
 
    for i,ch in enumerate(chan_ind):
        plt.subplot(nrows,ncols,i+1)
 
        for cl in trials.keys():
            plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:],axis = 1), label = cl)
 
        plt.xlim(1,30)
 
        if(maxy != None):
            plt.ylim(0,maxy)
        
        plt.grid()
 
        plt.xlabel('Frequency (Hz)')
 
        if(chan_lab == None):
            plt.title(f'Channel {ch+1}')
        else:
            plt.title(chan_lab[i])
 
        plt.legend()
 
    plt.tight_layout()
 
plot_psd(trials_PSD,freqs, [channel_names.index(ch) for ch in ['C3','Cz','C4']], chan_lab = ['left','center','right'],maxy = 500)
