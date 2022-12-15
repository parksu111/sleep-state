import os
import numpy as np
import rempy as rp
from tqdm import tqdm
import scipy.io as so
import matplotlib.pyplot as plt

# Path containing recordings
ppath = '/workspace/Competition/SLEEP/EEG/data/recordings'
# Path to save signal trace png
eeg1dst = '/workspace/Competition/SLEEP/EEG/data/trace/eeg1/'
eeg2dst = '/workspace/Competition/SLEEP/EEG/data/trace/eeg2/'
emgdst = '/workspace/Competition/SLEEP/EEG/data/trace/emg/'
csvdst = '/workspace/Competition/SLEEP/EEG/data/'

# List all recordings
recordings = os.listdir(ppath)
recordings = [x for x in recordings if not x.startswith('.')]

fname = []
fstate = []

# Loop through recordings, load raw signals, save each 2.5 second
for rec in recordings:
    print('Working on ' + rec + ' ...')
    # Load raw signals
    eeg1path = os.path.join(ppath, rec, 'EEG.mat')
    eeg2path = os.path.join(ppath, rec, 'EEG2.mat')
    emgpath = os.path.join(ppath, rec, 'EMG.mat')
    eeg1 = np.squeeze(so.loadmat(eeg1path)['EEG'])
    if os.path.isfile(eeg2path):
        eeg2 = np.squeeze(so.loadmat(eeg2path)['EEG2'])
    emg = np.squeeze(so.loadmat(emgpath)['EMG'])
    # Load annotations
    M,S = rp.load_stateidx(ppath, rec)
    # Loop through each 2.5 s time window
    for i in tqdm(range(len(M)-1)):
        figname = rec + '_' + str(i)
        fig1dst = os.path.join(eeg1dst, figname)
        fig2dst = os.path.join(eeg2dst, figname)
        fig3dst = os.path.join(emgdst, figname)

        cur_state = M[i]
        subeeg1 = eeg1[i*2500:(i+1)*2500]
        subemg = emg[i*2500:(i+1)*2500]

        fname.append(figname)
        fstate.append(cur_state)

        fig1 = plt.figure()
        plt.plot(np.arange(0,2500), subeeg1)
        plt.axis('off')
        fig1.savefig(fig1dst)
        plt.close(fig1)

        fig3 = plt.figure()
        plt.plot(np.arange(0,2500), subemg)
        plt.axis('off')
        fig3.savefig(fig3dst)
        plt.close(fig3)

        if os.path.isfile(eeg2path):
            subeeg2 = eeg2[i*2500:(i+1)*2500]
            fig2 = plt.figure()
            plt.plot(np.arange(0,2500), subeeg2)
            plt.axis('off')
            fig2.savefig(fig2dst)
            plt.close(fig2)

# Make dataframe key
keydf = pd.DataFrame(list(zip(fname, fstate)),columns=['imname','state'])
keydf.to_csv(os.path.join(csvdst,'trace_key.csv'),index=False)
