import os
import numpy as np
import re

def load_stateidx(ppath, name, my_ann = True):
    """ 
    Load the sleep state annotation txt file of recording.
    @Params:
        ppath       Path of folder containing all recordings.
        name        Name of the recording.

    @Return:
        M       List of sleep states corresponding to each 2.5 s window.
        K       List of 0's and 1's indicating whether or not a 
    """
    if my_ann:
        sfile = os.path.join(ppath, name, '3_remidx_' + name + '.txt')
    else:
        sfile = os.path.join(ppath, name, 'remidx_' + name + '.txt')
        
    f = open(sfile, 'r')
    lines = f.readlines()
    f.close()
    
    n = 0
    for l in lines:
        if re.match('\d', l):
            n += 1
            
    M = np.zeros(n, dtype='int')
    K = np.zeros(n, dtype='int')
    
    i = 0
    for l in lines :
        
        if re.search('^\s+$', l) :
            continue
        if re.search('\s*#', l) :
            continue
        
        if re.match('\d+\s+-?\d+', l) :
            a = re.split('\s+', l)
            M[i] = int(a[0])
            K[i] = int(a[1])
            i += 1
            
    return M,K