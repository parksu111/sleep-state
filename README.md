# Automatic Classification of Sleep States in Mice
![alt text](https://github.com/parksu111/sleep-state/blob/master/img/sleep-state.png)
Image was taken from this [paper](https://www.frontiersin.org/articles/10.3389/fpsyg.2020.01662/full).

## Introduction
Mammalian sleep consists of 3 distinct brain states: Wakefulness, REM (Rapid Eye Movement) sleep, and NREM (non-REM) sleep. 
These 3 states can be identified through inspection of EEG (electroencephalogram) and EMG (electromyogram) data. 
A major bottleneck in sleep research is that hours of sleep recordings have to be manually annotated by experts, a very time consuming process. 
Here, we will use deep neural networks to automatically classify sleep states based on EEG and EMG data obtained from mice. 

## Data
The data used for this project were made available by dedicated researchers at the [Chung Lab](https://chunglab.med.upenn.edu/) and [Weber Lab](https://www.med.upenn.edu/weberlab/) at the University of Pennsylvania Perelman School of Medicine. It is the same data used for the 2021 paper 'A probabilistic model for the ultradian timing of REM sleep in mice', which can be found [here](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009316).
* The raw sleep recordings used for this project are available for download [here](https://upenn.app.box.com/s/3zcesr4a7l7hgb9andmq4di4t6zvaoql).
* The processed images used for classification are available for download [here](https://drive.google.com/drive/folders/1v-PzV_-R47bKG65EnUDEsmWCFTlT4zAo?usp=share_link).

### Recording Folder
Each recording folder has a title of the following structure: 'XXX_XXXXXXn1'.
* The characters in front of the underscore is a unique ID for a mouse. 
* The numbers following the underscore indicates the date of the recording.
* Example: J16_062618n1 - This is a recording of the mouse named 'J16' taken on June 26th, 2018.

Each recording folder contains 7 different files:
* **EEG.mat** - A MATLAB Data File containing the raw EEG signal data. Parietal EEG sampled at 1000Hz.
* **EEG2.mat** - A MATLAB Data File containing the prefrontal EEG signal data sampled at 1000Hz.
* **EMG.mat** - EMG recorded from the neck sampled at 1000Hz.
* **3_remidx_XXX_XXXXXXn1.txt** - A text file containing labels for each 2.5 s window of the sleep recording.
  * 0 - Undefined
  * 1 - REM
  * 2 - Wake
  * 3 - REM
* **sp_XXX_XXXXXXn1.mat** - A MATLAB Data File containing the power spectrum of the raw EEG signals.
* **msp_XXX_XXXXXXn1.mat** - A MATLAB Data File containing the power spectrum of the raw EMG signals.
* **info.txt** - A text file containing basic information about the recording such as sampling rate, duration of the recording, etc.
* Each recording is typically 8 hours or 24 hours long.
* If the data files containing the power spectrum are not in the folder, they can be created by running the sleep_annotation_qt program found [here](https://github.com/parksu111/sleep-annotation).

### EDA
A detailed EDA of sleep recordings can be found in the following [notebook](https://github.com/parksu111/sleep-state/blob/master/EDA.ipynb).

### Multiple Annotations
A common issue with sleep recordings, and biomedical signals in general, is that there is a lot of noise in the data and even experts do not agree completely on how to label parts of the signals. 

In this project, we use a subset of the entire dataset that can be downloaded above. These 6 recordings were annotated by 4 different experts. The inter-rater agreement between these 4 experts is 88.099% (More details can be found in this [notebook](https://github.com/parksu111/sleep-state/blob/master/common_labels.ipynb)). 

When training our classification model, we only use data points for which all 4 experts agree. The different annotations are available for download [here](https://drive.google.com/drive/folders/1PLpvB6GAbd3y3Wu1qAsZDGW6x2iPqUnN?usp=share_link).

## Preprocessing
There are several ways to approach the problem of classifying sleep states using EEG and EMG data:
* Use the raw signal data.
* Use images of the waveform.
* Use spectrogram of the raw waveform.

In the cases of using either images of the waveform or the spectrogram, the problem of classifying sleep states becomes an image classification task. We first approach the problem as an image classification task. I will continue to update the project with different approaches.

### Files
* **common_labels.py** - Script to find commonly labeled windows and save them as a json file.
* **make_im_data.py** - Script to make image data from raw EEG and EMG signals.
  * Can choose channel (eeg1, eeg2, emg), number of windows, raw trace vs. spectrogram
  * How to use:
  ```
  python make_im_data --input_dir $/path/to/input/recordings --output_dir $/path/to/output/folder --datatype trace/spectrogram --num_bins 1/3 --common_labels 0/1 --channel 0/1/2/3
  ```
    * --datatype: Trace is the image of the raw waveform and spectrogram is the spectrogram of the raw waveform
    * --num_bins: Number of windows to use. 1 refers to one 2.5 s window and 3 refers to 7.5 seconds with 2.5 seconds before and after the window of interest.
    * --common_labels: Whether or not to use commonly annotated windows. 0 means False and 1 means True. Defaults to 1.
    * --channel: Which of the 3 channels to use. 0 = EEG1, 1 = EEG2, 2 = EMG, 3 = All Three.
* **split_data.py** - Split trace and spectrogram data into train and test set. Save key as csv file and move train and test files to separate folders.

## Model
### Classifier 1
#### Input Data
* Signals used: EEG1
* Data Type: Trace (Image of raw waveform)
* Number of windows: 1

#### Model Architecture
For this first classifier, we fine-tune the pretrained ResNet18 model. We modify only the fully-connected layer at the end. All parameters besides those in the fully-connected layer are frozen.

#### Files
* **train.py** - Script to train the model. Saves model parameters with highest accuracy as 'best.pt'.
* **predict.py** - Script to make predictions on test data. Saves predictions in a csv file.
* **best.pt** - The final model parameters are available for download [here](https://drive.google.com/drive/folders/1tMhWEJwJuFSEvhqMSzvXg4wtxlxR00qm?usp=share_link).

## Results
Model | Input type | # Windows | Accuracy | OOD Accuracy
------|------------|-----------|----------|--------------
Classifier 1| Trace|      1    | 94.96%      | 89.96%

The accuracy is calculated on the test dataset. OOD refers to the accuracy calculated on a separate dataset that was only annotated by me.
The details of the results can be found in this [notebook](https://github.com/parksu111/sleep-state/blob/master/evaluate.ipynb).

## To-do
- [ ] Use spectrogram as input image.
- [ ] Train LSTM using raw signal data.