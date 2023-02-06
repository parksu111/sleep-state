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
    parser.add_argument('--input_dir', type=pathlib.Path, required=True, help='Path to input directory')
    parser.add_argument('--output_dir', type=pathlib.Path, required=True, help='Path to output directory')
    parser.add_argument('--datatype', choices=['trace','spectrogram'],default='trace',help='Type of image data to save')
    parser.add_argument('--num_bins', type=int, choices=[1,3], default=1, required=True, help='Number of bins')
    parser.add_argument('--common_labels', type=int, choices=[0,1], default=1, required=True, help='Use common labels')
    parser.add_argument('--title', type=str, default='key.csv')
    parser.add_argument('--channel', type=int, choices=[0,1,2,3], required=True, help='Choose channel')

    # read arguments
    args = parser.parse_args()
    ppath = args.input_dir
    outpath = args.output_dir
    data_type = args.datatype
    binlen = args.num_bins
    use_common = args.common_labels
    title = args.title
    channel = args.channel

    # channels
    all_channels = ['eeg1','eeg2','emg']
    channel_type = ['EEG', 'EEG2', 'EMG']
    if channel==3:
        channel_inds = [0,1,2]
    else:
        channel_inds = [channel]
    # Make output directories
    srcpaths = []
    for ch_ind in channel_inds:
        srcdst = os.path.join(os.path.join(outpath), all_channels[ch_ind])
        os.makedirs(srcdst, exist_ok = True)
        srcpaths.append(srcdst)

    # input recordings
    recordings = os.listdir(ppath)
    recordings = [x for x in recordings if not x.startswith('.')]

    # commonly labelled bins
    if use_common==1:
        clabels = json.load(open('/workspace/Competition/SLEEP/EEG/data/common_labels.json'))

    # valid range
    if binlen == 3:
        valrange = range(1,)

    fnames = []
    fstate = []
    o_index = []

    for rec in recordings:
        print('Working on ' + rec + ' ...')
        # Load annotations
        M,_ = rp.load_stateidx(ppath, rec, 'sp')
        if use_common==1:
            inds = clabels[rec]["ind"]
            states = clabels[rec]["state"]
        else:
            inds = list(range(len(M)))
            states = M
        # valid range
        if binlen == 3:
            min_ind = 1
            max_ind = len(M)-3
        if binlen == 1:
            min_ind = 0
            max_ind = len(M)-3
        # Load raw signals
        for ch_ind in channel_inds:
            sigpath = os.path.join(ppath, rec, channel_type[ch_ind]+'.mat')
            rawsignal = np.squeeze(so.loadmat(sigpath)[channel_type[ch_ind]])
            # loop
            img_cnt = 0
            for idx,i in tqdm(enumerate(inds)):
                if (i>=min_ind)&(i<=max_ind):
                    if use_common==1:
                        fname = rec + '_' + str(img_cnt)
                    else:
                        fname = rec + '_' + str(i)
                    fpath = os.path.join(srcpaths[ch_ind], fname)
                    if data_type == "trace":
                        make_trace_middle(rawsignal,i,binlen,fpath)
                    if data_type == "spectrogram":
                        make_spec_middle(rawsignal,i,binlen,fpath)
                    img_cnt+=1
                    fnames.append(fname)
                    fstate.append(states[idx])
                    o_index.append(i)
    keydf = pd.DataFrame(list(zip(fnames,fstate,o_index)),columns=['fname','state','org_index'])
    keypath = os.path.join(outpath,title)
    keydf.to_csv(keypath, index=False)