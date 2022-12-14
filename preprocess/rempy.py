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