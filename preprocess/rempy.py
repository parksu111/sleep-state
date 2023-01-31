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

def stateSeq(sub, stateInds):
    stateIndices = stateIndex(sub, stateInds)
    stateRanges = ranges(stateIndices)
    stateSeqs = []
    for x in stateRanges:
        stateSeqs.append(np.arange(x[0],x[1]+1))
    return stateSeqs