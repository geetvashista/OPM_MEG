import numpy as np
import os
import bct
import netplotbrain
import pandas as pd

def load_npy_array(root_dir, filename):
    return np.load(os.path.join(root_dir, filename))


def get_file_names(master_dir, band, task):
    temp = []
    for filename in os.listdir(master_dir):
        if task in filename:
            if band in filename:
                temp.append(filename)
    return temp


def load_data(master_dir, target_task, target_band):
    target_files = get_file_names(master_dir, task=target_task, band=target_band)
    array = [load_npy_array(root_dir=master_dir, filename=i) for i in target_files]
    return np.array(array)


# __main__

master_dir = r'F:\Geet\Output_dir'

band = ['theta',
        'alpha',
        'beta',
        'gamma']

tasks = ['participant_conversation',
         'interviewer_repetition',
         'participant_repetition',
         'da',
         'ba',
         'interviewer_conversation']

convo_data_theta = load_data(master_dir=master_dir, target_task=tasks[0], target_band=band[1])
rep_data_theta = load_data(master_dir=master_dir, target_task=tasks[2], target_band=band[1])

_, adj, _ = bct.nbs_bct(convo_data_theta.T, rep_data_theta.T, 3.2, k=5000, seed=2022, paired=True)


# Plotting

array = xyz_loc_array


# Plotting
nodes = pd.DataFrame(data={'x': (array[:,0] - 128),
                           'y': (array[:,1] - 150),
                           'z': (array[:,2] - 100),
                           })

netplotbrain.plot(template='MNI152NLin2009cAsym',
                  nodes=nodes,
                  edges=gdif
                  template_style='filled',
                  view=['LSR'],
                  arrowaxis=None,
                  node_scale=100,
                  highlight_edges=adj)
