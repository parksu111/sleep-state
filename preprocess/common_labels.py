import os
import rempy as rp
from tqdm import tqdm
import pandas as pd
import json

# Path containing recordings
ppath = '/workspace/Competition/SLEEP/EEG/data/recordings'

recordings = os.listdir(ppath)
recordings = [x for x in recordings if not x.startswith('.')]

common_labels = {}
for rec in recordings:
    indices = []
    states = []
    M1,_ = rp.load_stateidx(ppath, rec, 'sp')
    M2,_ = rp.load_stateidx(ppath, rec, 'ha')
    M3,_ = rp.load_stateidx(ppath, rec, 'jh')
    M4,_ = rp.load_stateidx(ppath, rec, 'js')
    for i in range(len(M1)):
        if (M1[i]==M2[i])&(M2[i]==M3[i])&(M3[i]==M4[i]):
            indices.append(i)
            states.append(M1[i])
    common_labels[rec] = {'ind':indices, 'state':states}

with open('/workspace/Competition/SLEEP/EEG/data/common_labels.json','w') as outfile:
    json.dump(common_labels, outfile)
