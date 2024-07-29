import numpy as np
import bct
import netplotbrain
import matplotlib.pyplot as plt

# Setup back end
plt.matplotlib.use('Qt5Agg')

# load data
run_1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\Graph_output\Alpha_run_1\master_adj_matrix_alpha_run_1.npy')
run_2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\Graph_output\Alpha_run_2\master_adj_matrix_alpha_run_2.npy')

# nbs
P_val, adj, null = bct.nbs_bct(run_1.T, run_2.T, thresh= 3.1, k = 5000, paired=True)

# Group diffs
gdif = np.mean(run_1.T,axis=-1) - np.mean(run_2.T,axis=-1)

# Plotting
fig, ax = netplotbrain.plot(template='MNI152NLin2009cAsym',
                            nodes={'atlas': 'Schaefer2018',
                                     'desc': '100Parcels7Networks',
                                     'resolution': 1},
                            edges=gdif,
                            highlight_edges=adj,
                            template_style='glass',
                            view=['LSR'],
                            title='Between-run edge significance of Alpha',
                            node_type='circles',
                            highlight_level=0.5)
