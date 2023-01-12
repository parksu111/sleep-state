import os
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

datapath = '/workspace/Competition/SLEEP/EEG/data/'
masterkey = pd.read_csv(os.path.join(datapath, 'masterkey.csv'))
masterdict = masterkey.set_index('ind_name').to_dict(orient='index')

testpath = '/workspace/Competition/SLEEP/EEG/data/test'
trainpath = '/workspace/Competition/SLEEP/EEG/data/train'

# train test split
train,test = train_test_split(masterkey, test_size=0.2, random_state=2021)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
train.to_csv(os.path.join(datapath, 'trainkey.csv'), index=False)
test.to_csv(os.path.join(datapath, 'testkey.csv'), index=False)

print('Moving test files')

### move test files
for imfile in tqdm(list(test['ind_name'])):
    ##### 1 bin
    bin1name = masterdict[imfile]['bin1_name']+'.png'
    ## trace
    eeg1src = os.path.join(datapath,'raw','trace1','eeg1',bin1name)
    eeg2src = os.path.join(datapath,'raw','trace1','eeg2',bin1name)
    emgsrc = os.path.join(datapath,'raw','trace1','emg',bin1name)
    eeg1dst = os.path.join(testpath, 'trace1', 'eeg1')
    eeg2dst = os.path.join(testpath, 'trace1', 'eeg2')
    emgdst = os.path.join(testpath, 'trace1', 'emg')
    shutil.move(eeg1src, eeg1dst)
    shutil.move(eeg2src, eeg2dst)
    shutil.move(emgsrc, emgdst)    
    ## spec
    eeg1src = os.path.join(datapath,'raw','spec1','eeg1',bin1name)
    eeg2src = os.path.join(datapath,'raw','spec1','eeg2',bin1name)
    emgsrc = os.path.join(datapath,'raw','spec1','emg',bin1name)
    eeg1dst = os.path.join(testpath, 'spec1', 'eeg1')
    eeg2dst = os.path.join(testpath, 'spec1', 'eeg2')
    emgdst = os.path.join(testpath, 'spec1', 'emg')
    shutil.move(eeg1src, eeg1dst)
    shutil.move(eeg2src, eeg2dst)
    shutil.move(emgsrc, emgdst)   
    ##### 3 bins
    if masterdict[imfile]['bin3_name'] != 'missing':
        bin3name = masterdict[imfile]['bin3_name']+'.png'
        ## trace
        eeg1src = os.path.join(datapath,'raw','trace3','eeg1',bin3name)
        eeg2src = os.path.join(datapath,'raw','trace3','eeg2',bin3name)
        emgsrc = os.path.join(datapath,'raw','trace3','emg',bin3name)
        eeg1dst = os.path.join(testpath, 'trace3', 'eeg1')
        eeg2dst = os.path.join(testpath, 'trace3', 'eeg2')
        emgdst = os.path.join(testpath, 'trace3', 'emg')
        shutil.move(eeg1src, eeg1dst)
        shutil.move(eeg2src, eeg2dst)
        shutil.move(emgsrc, emgdst)    
        ## spec
        eeg1src = os.path.join(datapath,'raw','spec3','eeg1',bin3name)
        eeg2src = os.path.join(datapath,'raw','spec3','eeg2',bin3name)
        emgsrc = os.path.join(datapath,'raw','spec3','emg',bin3name)
        eeg1dst = os.path.join(testpath, 'spec3', 'eeg1')
        eeg2dst = os.path.join(testpath, 'spec3', 'eeg2')
        emgdst = os.path.join(testpath, 'spec3', 'emg')
        shutil.move(eeg1src, eeg1dst)
        shutil.move(eeg2src, eeg2dst)
        shutil.move(emgsrc, emgdst)  

print('Moving train files')

### move train files
for imfile in tqdm(list(train['ind_name'])):
    ##### 1 bin
    bin1name = masterdict[imfile]['bin1_name']+'.png'
    ## trace
    eeg1src = os.path.join(datapath,'raw','trace1','eeg1',bin1name)
    eeg2src = os.path.join(datapath,'raw','trace1','eeg2',bin1name)
    emgsrc = os.path.join(datapath,'raw','trace1','emg',bin1name)
    eeg1dst = os.path.join(trainpath, 'trace1', 'eeg1')
    eeg2dst = os.path.join(trainpath, 'trace1', 'eeg2')
    emgdst = os.path.join(trainpath, 'trace1', 'emg')
    shutil.move(eeg1src, eeg1dst)
    shutil.move(eeg2src, eeg2dst)
    shutil.move(emgsrc, emgdst)    
    ## spec
    eeg1src = os.path.join(datapath,'raw','spec1','eeg1',bin1name)
    eeg2src = os.path.join(datapath,'raw','spec1','eeg2',bin1name)
    emgsrc = os.path.join(datapath,'raw','spec1','emg',bin1name)
    eeg1dst = os.path.join(trainpath, 'spec1', 'eeg1')
    eeg2dst = os.path.join(trainpath, 'spec1', 'eeg2')
    emgdst = os.path.join(trainpath, 'spec1', 'emg')
    shutil.move(eeg1src, eeg1dst)
    shutil.move(eeg2src, eeg2dst)
    shutil.move(emgsrc, emgdst)   
    ##### 3 bins
    if masterdict[imfile]['bin3_name'] != 'missing':
        bin3name = masterdict[imfile]['bin3_name']+'.png'
        ## trace
        eeg1src = os.path.join(datapath,'raw','trace3','eeg1',bin3name)
        eeg2src = os.path.join(datapath,'raw','trace3','eeg2',bin3name)
        emgsrc = os.path.join(datapath,'raw','trace3','emg',bin3name)
        eeg1dst = os.path.join(trainpath, 'trace3', 'eeg1')
        eeg2dst = os.path.join(trainpath, 'trace3', 'eeg2')
        emgdst = os.path.join(trainpath, 'trace3', 'emg')
        shutil.move(eeg1src, eeg1dst)
        shutil.move(eeg2src, eeg2dst)
        shutil.move(emgsrc, emgdst)    
        ## spec
        eeg1src = os.path.join(datapath,'raw','spec3','eeg1',bin3name)
        eeg2src = os.path.join(datapath,'raw','spec3','eeg2',bin3name)
        emgsrc = os.path.join(datapath,'raw','spec3','emg',bin3name)
        eeg1dst = os.path.join(trainpath, 'spec3', 'eeg1')
        eeg2dst = os.path.join(trainpath, 'spec3', 'eeg2')
        emgdst = os.path.join(trainpath, 'spec3', 'emg')
        shutil.move(eeg1src, eeg1dst)
        shutil.move(eeg2src, eeg2dst)
        shutil.move(emgsrc, emgdst)  