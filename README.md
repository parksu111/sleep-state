# Automatic Classification of Sleep States in Mice

## Introduction
Mammalian sleep consists of 3 distinct brain states: Wakefulness, REM (Rapid Eye Movement) sleep, and NREM (non-REM) sleep. 
These 3 states can be identified through inspection of EEG (electroencephalogram) and EMG (electromyogram) data. 
A major bottleneck in sleep research is that hours of sleep recordings have to be manually annotated by experts, a very time consuming process. 
Here, we will use deep neural networks to automatically classify sleep states based on EEG and EMG data obtained from mice. 
The data used for this project were made available by dedicated researchers at the [Chung Lab](https://chunglab.med.upenn.edu/) and [Weber Lab](https://www.med.upenn.edu/weberlab/) at the 
University of Pennsylvania Perelman School of Medicine.

## Data
The data used for this project is available for download [here](). It is the same data used for the 2021 paper 'A probabilistic model for the ultradian timing of REM sleep in mice', which can be found [here](). The data repository contains 2 folders, 'code' and 'recordings'. The sleep recordings are in the 'recordings' folder.

### Recording Folder
Each recording folder has a title of the following structure: 'XXX_XXXXXXn1'.
* The characters in front of the underscore is a unique ID for a mouse. 
* The numbers following the underscore indicates the date of the recording.
* Example: J16_062618n1 - This is a recording of the mouse named 'J16' taken on June 26th, 2018.

Each recording folder contains 7 different files:
* **EEG.mat** - A MATLAB Data File containing the raw EEG signal data. Parietal EEG sampled at 1000Hz.
* **EEG2.mat** - A MATLAB Data File containing the prefrontal EEG signal data sampled at 1000Hz.
* **EMG.mat** - EMG recorded from the neck sampled at 1000Hz.
* 3_remidx_XXX_XXXXXXn1.txt - A text file containing labels for each 2.5 s window of the sleep recording.
  * 0 - Undefined
  * 1 - REM
  * 2 - Wake
  * 3 - REM
* sp_XXX_XXXXXXn1.mat - A MATLAB Data File containing the power spectrum of the raw EEG signals.
* msp_XXX_XXXXXXn1.mat - A MATLAB Data File containing the power spectrum of the raw EMG signals.
* **info.txt** - A text file containing basic information about the recording such as sampling rate, duration of the recording, etc.
* Each recording is typically 8 hours or 24 hours long.
* If the data files containing the power spectrum are not in the folder, they can be created by running the sleep_annotation_qt program found [here]().

### EDA
A detailed EDA of sleep recordings can be found in the following [notebook]().

### Multiple Annotations
A common issue with sleep recordings, and biomedical signals in general, is that there is a lot of noise in the data and even experts do not agree completely on how to label parts of the signals. 

In this project, we use a subset of the entire dataset that can be downloaded above. These 6 recordings were annotated by 4 different experts. The inter-rater agreement between these 4 experts is 88.099% (More details can be found [here]()). 

When training our classification model, we only use data points for which all 4 experts agree. The different annotations are available for download [here]().

