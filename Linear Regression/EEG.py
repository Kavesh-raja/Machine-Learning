from glob import glob
import os
import numpy as np
import pandas as pd
import mne
from matplotlib import  pyplot as plt
data_set = glob('dataverse_files/*.edf')
raw = mne.io.read_raw_edf(data_set[0])

def read_data(file_path):
    data = mne.io.read_raw_edf(file_path,preload=True)
    epochs = mne.make_fixed_length_epochs(data,duration = 5, overlap = 1)
    array = epochs.get_data()
    return  max(array[0][0]*1000000)
data_array = [read_data(i) for i in data_set]
df = pd.DataFrame()
df['EEG_epochs']  = data_array
df2 =pd.read_csv('Sor.csv')
df['ANX'] = df2['ANX']
df.to_csv('epo2.csv')

