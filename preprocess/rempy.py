import os
import numpy as np
import re
import scipy.signal

def load_stateidx(ppath, name, annotator = 'sp'):
    """ 
    Load the sleep state annotation txt file of recording.
    @Params:
        ppath       Path of folder containing all recordings.
        name        Name of the recording.

    @Return:
        M       List of sleep states corresponding to each 2.5 s window.
        K       List of 0's and 1's indicating whether or not a 
    """
    ann_key = {'sp':'3_remidx_', 'ha':'ha_remidx_','jh':'jh_remidx_','js':'js_remidx_'}
    sfile = os.path.join(ppath,name,ann_key[annotator]+name+'.txt')
        
    with open(sfile) as f:
        lines = f.readlines()

    M = []
    K = []
    for line in lines:
        if line.startswith('#') or line.isspace():
            continue
        a = line.strip().split()
        if len(a) != 2:
            continue
        M.append(int(a[0]))
        K.append(int(a[1]))

    return M,K

def get_sr(ppath, name):
    """
    read and return sampling rate (SR) from the info.txt file $ppath/$name/info.txt 
    """
    fid = open(os.path.join(ppath, name, 'info.txt'), newline=None)
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines :
        a = re.search("^" + 'SR' + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))            
    return float(values[0])

def decode(M):
    """
    Convert annotated states from number code to letter code.
    @Params:
        M       List containing sleep states (int) of a recording extracted from remidx file.
    
    @Return:
        res     List containing converted sleep states
                REM = R
                NREM = N
                WAKE = W
    """
    state_dict = {0:'N', 1:'R', 2:'W', 3:'N', 4:'N', 5:'N', 6:'N'}
    res = [state_dict[x] for x in M]
    return res

def encode(M):
    """
    Convert annotated states from letter code to number code.
    @Params:
        M       List containing sleep states (string) of a recording extracted from remidx file.
    
    @Return:
        res     List containing converted sleep states
                0 = N
                1 = R
                2 = W
    """    
    state_dict = {'N':0, "R":1, "W":2}
    res = [state_dict[x] for x in M]
    return res

def power_spectrum(data, length, dt):
    f, pxx = scipy.signal.welch(data, fs=1/dt, window='hanning', nperseg=int(length), noverlap=int(length/2))
    return pxx, f

'''
Functions to deal with remidx sequences
'''
def nts(M):
    """
    Receive list of states coded as number and change to letters representing state
    """
    rvec = [None]*len(M)
    for idx,x in enumerate(M):
        if x==1:
            rvec[idx]='R'
        elif x==2:
            rvec[idx]='W'
        else:
            rvec[idx]='N'
    return rvec

def vecToTup(rvec, start=0):
    """
    Receive list of states and change to a list of tuples (state, duration, starting index)
    """
    result = []
    cnt1 = 0
    i_start = start
    sum1 = 1
    curr = 'a'
    while cnt1 < len(rvec)-1:
        curr = rvec[cnt1]
        nxt = rvec[cnt1+1]
        if curr==nxt:
            sum1+=1
        else:
            result.append((curr, sum1, i_start))
            sum1=1
        cnt1+=1
        i_start+=1
    last = rvec[-1]
    if curr==last:
        result.append((curr, sum1, i_start))
    else:
        result.append((last, sum1, i_start))
    return result

def nrt_locations(tupList):
    """
    Receive list of tuples and return index of NREM-->REM transitions
    """
    nrt_locs = []
    cnt1 = 0
    while cnt1 < len(tupList)-1:
        curr = tupList[cnt1]
        nxt = tupList[cnt1+1]
        if (curr[0]=='N')&(nxt[0]=='R'):
            nrt_locs.append(cnt1+1)
        cnt1+=1
    return nrt_locs

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def stateIndex(sub, stateInds):
    res = []
    for x in stateInds:
        cur = sub[x]
        curIndex = list(np.arange(cur[2]-cur[1]+1,cur[2]+1))
        res.extend(curIndex)
    return res

def stateSeq(sub, stateInds):
    stateIndices = stateIndex(sub, stateInds)
    stateRanges = ranges(stateIndices)
    stateSeqs = []
    for x in stateRanges:
        stateSeqs.append(np.arange(x[0],x[1]+1))
    return stateSeqs

'''
Plot
'''

def sleep_example(ppath, name, tlegend, tstart, tend, fmax=30, fig_file='', vm=[], ma_thr=10,
                  fontsize=12, cb_ticks=[], emg_ticks=[], r_mu = [10, 100], fw_color=True):
    """
    plot sleep example
    :param ppath: base folder
    :param name: recording name
    :param tstart: start (in seconds) of shown example interval
    :param tend: end of example interval
    :param tlegend: length of time legend
    :param fmax: maximum frequency shown for EEG spectrogram
    :param fig_file: file name where figure will be saved
    :param vm: saturation of EEG spectrogram
    :param fontsize: fontsize
    :param cb_ticks: ticks for colorbar
    :param emg_ticks: ticks for EMG amplitude axis (uV)
    :param r_mu: range of frequencies for EMG amplitude
    :param fw_color: if True, use standard color scheme for brainstate (gray - NREM, violet - Wake, cyan - REM);
            otherwise use Shinjae's color scheme
    """
    set_fontarial()
    set_fontsize(fontsize)

    # True, if laser exists, otherwise set to False
    plaser = False

    sr = get_snr(ppath, name)
    nbin = np.round(2.5 * sr)
    dt = nbin * 1 / sr

    istart = int(np.round(tstart/dt))
    iend   = int(np.round(tend/dt))
    dur = (iend-istart+1)*dt

    M,K = load_stateidx(ppath, name)
    #kcut = np.where(K>=0)[0]
    #M = M[kcut]
    if tend==-1:
        iend = len(M)
    M = M[istart:iend]

    seq = get_sequences(np.where(M==2)[0])
    for s in seq:
        if len(s)*dt <= ma_thr:
            M[s] = 0

    t = np.arange(0, len(M))*dt

    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name), squeeze_me=True)
    SPEEG = P['SP']#/1000000.0
    # calculate median for choosing right saturation for heatmap
    med = np.median(SPEEG.max(axis=0))
    if len(vm) == 0:
        vm = [0, med*2.5]
    #t = np.squeeze(P['t'])
    freq = P['freq']
    P = so.loadmat(os.path.join(ppath, name, 'msp_%s.mat' % name), squeeze_me=True)
    SPEMG = P['mSP']#/1000000.0


    # create figure
    plt.ion()
    plt.figure(figsize=(12,6))

    # axis in the background to draw laser patches
    axes_back = plt.axes([0.1, .4, 0.8, 0.52])
    axes_back.get_xaxis().set_visible(False)
    axes_back.get_yaxis().set_visible(False)
    axes_back.spines["top"].set_visible(False)
    axes_back.spines["right"].set_visible(False)
    axes_back.spines["bottom"].set_visible(False)
    axes_back.spines["left"].set_visible(False)

    plt.ylim((0,1))
    plt.xlim([t[0], t[-1]])


    # show brainstate
    axes_brs = plt.axes([0.1, 0.4, 0.8, 0.05])
    cmap = plt.cm.jet
    if fw_color:
        my_map = cmap.from_list('brs', [[1, 118./255, 245./255], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    else:
        my_map = cmap.from_list('brs', [[0, 0, 0], [153 / 255.0, 76 / 255.0, 9 / 255.0],
                                        [120 / 255.0, 120 / 255.0, 120 / 255.0], [1, 0.75, 0]], 4)

    tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
    tmp.set_cmap(my_map)
    axes_brs.axis('tight')
    axes_brs.axes.get_xaxis().set_visible(False)
    axes_brs.axes.get_yaxis().set_visible(False)
    axes_brs.spines["top"].set_visible(False)
    axes_brs.spines["right"].set_visible(False)
    axes_brs.spines["bottom"].set_visible(False)
    axes_brs.spines["left"].set_visible(False)

    axes_legend = plt.axes([0.1, 0.33, 0.8, 0.05])
    plt.ylim((0,1.1))
    plt.xlim([t[0], t[-1]])
    plt.plot([0, tlegend], [1, 1], color='black')
    plt.text(tlegend/4.0, 0.1, str(tlegend) + ' s')
    axes_legend.spines["top"].set_visible(False)
    axes_legend.spines["right"].set_visible(False)
    axes_legend.spines["bottom"].set_visible(False)
    axes_legend.spines["left"].set_visible(False)
    axes_legend.axes.get_xaxis().set_visible(False)
    axes_legend.axes.get_yaxis().set_visible(False)

    # show spectrogram
    ifreq = np.where(freq <= fmax)[0]
    # axes for colorbar
    axes_cbar = plt.axes([0.82, 0.68, 0.1, 0.2])
    # axes for EEG spectrogram
    axes_spec = plt.axes([0.1, 0.68, 0.8, 0.2], sharex=axes_brs)
    im = axes_spec.pcolorfast(t, freq[ifreq], SPEEG[ifreq, istart:iend], cmap='jet', vmin=vm[0], vmax=vm[1])
    axes_spec.axis('tight')
    axes_spec.set_xticklabels([])
    axes_spec.set_xticks([])
    axes_spec.spines["bottom"].set_visible(False)
    plt.ylabel('Freq (Hz)')
    box_off(axes_spec)
    plt.xlim([t[0], t[-1]])

    # colorbar for EEG spectrogram
    cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0)
    cb.set_label('Power ($\mathrm{\mu}$V$^2$s)')
    if len(cb_ticks) > 0:
        cb.set_ticks(cb_ticks)
    axes_cbar.set_alpha(0.0)
    axes_cbar.spines["top"].set_visible(False)
    axes_cbar.spines["right"].set_visible(False)
    axes_cbar.spines["bottom"].set_visible(False)
    axes_cbar.spines["left"].set_visible(False)
    axes_cbar.axes.get_xaxis().set_visible(False)
    axes_cbar.axes.get_yaxis().set_visible(False)

    # show EMG
    i_mu = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    # * 1000: to go from mV to uV
    p_mu = np.sqrt(SPEMG[i_mu, :].sum(axis=0) * (freq[1] - freq[0])) #* 1000.0 # back to muV
    axes_emg = plt.axes([0.1, 0.5, 0.8, 0.1], sharex=axes_spec)
    axes_emg.plot(t, p_mu[istart:iend], color='black')
    axes_emg.patch.set_alpha(0.0)
    axes_emg.spines["bottom"].set_visible(False)
    if len(emg_ticks) > 0:
        axes_emg.set_yticks(emg_ticks)
    plt.ylabel('Ampl. ' + '$\mathrm{(\mu V)}$')
    plt.xlim((t[0], t[-1] + 1))
    box_off(axes_emg)

    #if len(fig_file) > 0:
    #    save_figure(fig_file)

    plt.show()