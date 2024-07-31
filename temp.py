import numpy as np
import os
import dyconnmap

input_dir = r'C:\Users\em17531\Desktop\OPM_MEG\data'  # The directory containing files of interest
output_dir = r'C:\Users\em17531\Desktop\OPM_MEG\Graph_output\Beta'  # The output directory
array_type = 'run-001'  # This is the type of array that will be loaded for further calculations, eg. "participant_conversation_alpha"
target_fb = [13, 30]  # The target frequency band     TODO: set this up so variable can be a this can be list of bands and each generates it's own adjacency matrix
band = '_beta_'
fs = 1200

data = []
for folder in os.listdir(input_dir):
    for file in os.listdir(os.path.join(input_dir, folder)):
        if array_type in os.path.basename(file):
            target_array = np.load(os.path.join(input_dir, folder, file))
            target_array = target_array[0:606000, :]
            data.append(target_array)
array = np.stack(data, axis=0)

del data
del target_array

adj_matrix = []
for participant in array:
    adj_matrix.append(dyconnmap.fc.wpli(participant, fs=fs, fb=target_fb))
