"""
A script for the plotting of AAP atlas graph. These regions ARE NOT set to MNI space as yet.
78 ROI's where used to match with:
https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00020/117508/Test-retest-reliability-of-the-human-connectome-An
A number of changes are possible from here such as, communities and highlight_edges.
"""
import netplotbrain
import pandas as pd
import numpy as np
import scipy
import matplotlib

# Setting some backend perameters
# matplotlib.use('TkAgg')

com = []
for i in range(78):
    com.append(0)

# # Strength
stg = [4, 19, 20, 22, 74]
for i, val in enumerate(com):
    for k in stg:
        if i == k:
            com[i] = 1
#
# Betweeness
stg = [59]
for i, val in enumerate(com):
    for k in stg:
        if i == k:
            com[i] = 2

# Eignvec
stg = [7, 17, 28, 30, 38, 46, 68, 69]
for i, val in enumerate(com):
    for k in stg:
        if i == k:
            com[i] = 3
#
# Clustering
stg = [0, 12, 13, 16, 38, 75]
for i, val in enumerate(com):
    for k in stg:
        if i == k:
            com[i] = 4



# Set depanneurs
mat_voxlox = scipy.io.loadmat(r'C:\Users\em17531\Desktop\Atlas\OPM_atlas_xyz.mat')
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Theta\run_1\Master_adj_matrix__Theda_run-001.npy')
r1 = np.mean(r1, axis=0)
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Theta\run_2\Master_adj_matrix__Theda_run-002.npy')
r2 = np.mean(r2, axis=0)
adj = r1 - r2
hig = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Theta\stats_Theta\nbs\_adj.npy')

# pep loc file
loc = mat_voxlox['voxlox']
del mat_voxlox

# adj = np.mean(adj, axis=0)

# Plotting

nodes = pd.DataFrame(data={'x': loc[0, :],
                           'y': loc[1, :],
                           'z': loc[2, :],
                           'community': com
                           })

netplotbrain.plot(nodes=nodes,
                  template_style='filled',
                  edges=adj,
                  highlight_edges=hig,
                  view=['LSR'],
                  node_color='community',
                  title='Theta',
                  arrowaxis=None,
                  highlight_level=0.8,
                  node_scale=100)

# import seaborn as sns
# r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Full_band\Full_band_run_2\master_adj_matrix_Full_band_run_2.npy')
# df = pd.DataFrame(np.mean(r1, axis=-1))
# sns.displot(df.T, kind='kde')
