import os
import numpy as np
import rempy as rp
from tqdm import tqdm
import scipy.io as so
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import pathlib
from scipy import signal
import json

#
def make_trace_middle(raw_sig, bin_index, bin_length, fname):
    if bin_length==1:
        start = bin_index
        end = bin_index+1
    elif bin_length==3:
        start = bin_index-1
        end = bin_index+2
    eeg_start = start*2500
    eeg_end = end*2500
    subsig = raw_sig[eeg_start:eeg_end]
    fig = plt.figure(figsize=(6.4*bin_length, 4.8))
    plt.plot(np.arange(0,bin_length*2500), subsig)
    plt.axis('off')
    fig.savefig(fname)
    plt.close(fig)

def make_spec_middle(raw_sig, bin_index, bin_length, fname):
    if bin_length==1:
        start = bin_index
        end = bin_index+1
        fsize = (0.5,3.1)
    elif bin_length==3:
        start = bin_index-1
        end = bin_index+2
        fsize = (5.5,3.1)
    eeg_start = start*2500
    eeg_end = end*2500
    subarrays = []
    for substart in np.arange(eeg_start, eeg_start + bin_length*2500-2000, 100):
        seqstart = substart
        seqend = seqstart+2000
        if seqstart<0:
            raise ValueError('Value must be non-negative')
        if seqend>len(raw_sig):
            raise ValueError('Value must be shorter than signal length')
        f,pxx = signal.welch(raw_sig[seqstart:seqend],fs=1000,window='hanning',nperseg=2000,noverlap=1000)
        ifreq = np.where((f>=0)&(f<=15))
        Pow = pxx[ifreq]
        subarrays.append(Pow)
    totPow = np.stack(subarrays, axis=1)
    # Plot spectogram
    fig = plt.figure(figsize=fsize,dpi=100)
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
    plt.imshow(totPow,cmap='hot',interpolation='nearest',origin='lower')
    plt.gca().set_axis_off()
    plt.margins(0,0)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(fname)
    plt.close(fig)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=pathlib.Path, required=True, help='Path to output directory')
    parser.add_argument('--datatype', choices=['trace','spectrogram'],default='trace',help='Type of image data to save')
    parser.add_argument('--num_bins', type=int, choices=[1,3], default=1, required=True, help='Number of bins')

    # read arguments
    args = parser.parse_args()
    outpath = args.output_dir
    data_type = args.datatype
    binlen = args.num_bins

    # path to output images
    eeg1dst = os.path.join(outpath, 'eeg1')
    eeg2dst = os.path.join(outpath, 'eeg2')
    emgdst = os.path.join(outpath, 'emg')
    # make output directories
    os.makedirs(eeg1dst, exist_ok=True)
    os.makedirs(eeg2dst, exist_ok=True)
    os.makedirs(emgdst, exist_ok=True)

    # input recordings
    ppath = '/workspace/Competition/SLEEP/EEG/data/recordings/'
    recordings = os.listdir(ppath)
    recordings = [x for x in recordings if not x.startswith('.')]

    # commonly labelled bins
    clabels = json.load(open('/workspace/Competition/SLEEP/EEG/data/common_labels.json'))

    # valid range
    if binlen == 3:
        valrange = range(1,)

    fnames = []
    fstate = []
    o_index = []

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
        inds = clabels[rec]["ind"]
        states = clabels[rec]["state"]
        M,_ = rp.load_stateidx(ppath, rec, 'sp')
        # valid range
        if binlen == 3:
            min_ind = 1
            max_ind = len(M)-3
        if binlen == 1:
            min_ind = 0
            max_ind = len(M)-3
        # loop
        img_cnt = 0
        for idx,i in tqdm(enumerate(inds)):
            if (i>=min_ind)&(i<=max_ind):
                fname = rec + '_' + str(img_cnt)
                fpath1 = os.path.join(eeg1dst, fname)
                fpath2 = os.path.join(eeg2dst, fname)
                fpath3 = os.path.join(emgdst, fname)
                if data_type == "trace":
                    make_trace_middle(eeg1,i,binlen,fpath1)
                    make_trace_middle(eeg2,i,binlen,fpath2)
                    make_trace_middle(emg,i,binlen,fpath3)
                if data_type == "spectrogram":
                    make_spec_middle(eeg1,i,binlen,fpath1)
                    make_spec_middle(eeg2,i,binlen,fpath2)
                    make_spec_middle(emg,i,binlen,fpath3)
                img_cnt+=1
                fnames.append(fname)
                fstate.append(states[idx])
                o_index.append(i)
    keydf = pd.DataFrame(list(zip(fnames,fstate,o_index)),columns=['fname','state','org_index'])
    keypath = os.path.join('/workspace/Competition/SLEEP/EEG/data/',data_type+str(binlen)+'_key.csv')
    keydf.to_csv(keypath, index=False)



            



'''
def make_trace_prev(raw_sig, bin_index, bin_length, fname):
    start = bin_index - (bin_length-1)
    end = bin_index+1
    eeg_start = start*2500
    eeg_end = end*2500
    subsig = raw_sig[eeg_start:eeg_end]
    fig = plt.figure(figsize=(6.4*bin_length, 4.8))
    plt.plot(np.arange(0,bin_length*2500), subsig)
    plt.axis('off')
    fig.savefig(fname)
    plt.close(fig)

def make_spec_prev(raw_sig, bin_index, bin_length, fname):
    # calculate spectrum
    start = bin_index - (bin_length-1)
    end = bin_index+1
    eeg_start = start*2500
    eeg_end = end*2500
    subarrays = []
    for substart in np.arange(eeg_start, eeg_end, 250):
        seqstart = substart-500
        seqend = substart+1000
        if seqstart<0:
            raise ValueError('Value must be non-negative')
        sup = list(range(seqstart,seqend+1))
        Pow,F = rp.power_spectrum(raw_sig[sup],1000,1/1000)
        ifreq = np.where((F>=0)&(F<=20))
        subPow = Pow[ifreq]
        subarrays.append(subPow)
    totPow = np.stack(subarrays, axis=1)
    # Plot spectogram
    fig = plt.figure(figsize=(2.0*bin_length,4.2),dpi=100)
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
    plt.imshow(totPow,cmap='hot',interpolation='nearest',origin='lower')
    plt.gca().set_axis_off()
    plt.margins(0,0)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(fname)
    plt.close(fig)
'''


