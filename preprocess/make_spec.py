import os
import numpy as np
import rempy as rp
from tqdm import tqdm
import scipy.io as so
import matplotlib.pyplot as plt
import pandas as pd

# Path containing recordings
ppath = '/workspace/Competition/SLEEP/EEG/data/recordings'
# Path to save spectogram png
eeg1dst = '/workspace/Competition/SLEEP/EEG/data/spectogram/eeg1/'
eeg2dst = '/workspace/Competition/SLEEP/EEG/data/spectogram/eeg2/'
emgdst = '/workspace/Competition/SLEEP/EEG/data/spectogram/emg/'
csvdst = '/workspace/Competition/SLEEP/EEG/data/'

# List all recordings
recordings = os.listdir(ppath)
recordings = [x for x in recordings if not x.startswith('.')]

fname = []
fstate = []
o_index = []

# Loop through recordings, load raw signals, save each 2.5 second
for rec in recordings:
    print('Working on ' + rec + ' ...')
    # Load raw signals
    eeg1path = os.path.join(ppath, rec, 'EEG.mat')
    eeg2path = os.path.join(ppath, rec, 'EEG2.mat')
    emgpath = os.path.join(ppath, rec, 'EMG.mat')
    eeg1 = np.squeeze(so.loadmat(eeg1path)['EEG'])
    eeg2 = np.squeeze(so.loadmat(eeg2path)['EEG2'])
    emg = np.squeeze(so.loadmat(emgpath)['EMG'])
    # Load annotations
    M1,_ = rp.load_stateidx(ppath, rec, 'sp')
    M2,_ = rp.load_stateidx(ppath, rec, 'jh')
    M3,_ = rp.load_stateidx(ppath, rec, 'ha')
    M4,_ = rp.load_stateidx(ppath, rec, 'js')
    # Loop through each 2.5 s time window
    figcnt = 0
    for i in tqdm(range(1,len(M1)-1)):
        if (M1[i]==M2[i])&(M2[i]==M3[i])&(M3[i]==M4[i]):
            figname = rec + '_' + str(figcnt)
            figcnt+=1
            fig1dst = os.path.join(eeg1dst, figname)
            fig2dst = os.path.join(eeg2dst, figname)
            fig3dst = os.path.join(emgdst, figname)

            eeg_start = i*2500
            eeg_end = (i+1)*2500

            subarrays1 = []
            subarrays2 = []
            subarrays3 = []
            for substart in np.arange(eeg_start, eeg_end, 250):
                seqstart = substart-500
                seqend = substart+1000
                sup = list(range(seqstart,seqend+1))
                Pow1,F = rp.power_spectrum(eeg1[sup],1000,1/1000)
                Pow2,_ = rp.power_spectrum(eeg2[sup],1000,1/1000)
                Pow3,_ = rp.power_spectrum(emg[sup],1000,1/1000)
                ifreq = np.where((F>=0)&(F<=20))
                subPow1 = Pow1[ifreq]
                subPow2 = Pow2[ifreq]
                subPow3 = Pow3[ifreq]
                subarrays1.append(subPow1)
                subarrays2.append(subPow2)
                subarrays3.append(subPow3)
            totPow1 = np.stack(subarrays1, axis=1)
            totPow2 = np.stack(subarrays2, axis=1)
            totPow3 = np.stack(subarrays3, axis=1)

            cur_state = M1[i]
            fname.append(figname)
            fstate.append(cur_state)
            o_index.append(i)

            fig1 = plt.figure(figsize=(2.0,4.2),dpi=100)
            plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
            plt.imshow(totPow1,cmap='hot',interpolation='nearest',origin='lower')
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            fig1.savefig(fig1dst)
            plt.close(fig1)

            fig2 = plt.figure(figsize=(2.0,4.2),dpi=100)
            plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
            plt.imshow(totPow2,cmap='hot',interpolation='nearest',origin='lower')
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            fig2.savefig(fig2dst)
            plt.close(fig2)

            fig3 = plt.figure(figsize=(2.0,4.2),dpi=100)
            plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
            plt.imshow(totPow3,cmap='hot',interpolation='nearest',origin='lower')
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            fig3.savefig(fig3dst)
            plt.close(fig3)

# Make dataframe key
keydf = pd.DataFrame(list(zip(fname, fstate, o_index)),columns=['imname','state','original_index'])
keydf.to_csv(os.path.join(csvdst,'spec_key.csv'),index=False)    