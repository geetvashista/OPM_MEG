# TO DO: get data, cut it up into windows, psd, bring it all together.
import numpy as np
import matplotlib
import os
from scipy import signal
import seaborn as sns

matplotlib.use('TkAgg')

def psd_calculator(signal):
    psd = 10 * np.log10(np.maximum(signal, 1e-10) / 1)
    return psd


def psd_with_scipy(signal):
    return signal.welch(signal, axis=0)


# Get data

def get_data(root_directory):
    all_data = []
    for file in os.listdir(root_directory):
        all_data.append(np.load(os.path.join(root_directory, file)))
    time_len = [len(i[:, 0]) for i in all_data]
    all_data_cropped = [i[0:min(time_len)] for i in all_data]
    return np.array(all_data_cropped)


root_directory = r'C:\Users\em17531\Desktop\OPM_MEG\data\run_1'
run_1_data = get_data(root_directory)
input_data = run_1_data[0, :, :]
