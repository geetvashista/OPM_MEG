import numpy as np
import bct
from scipy import stats

# T_test and FDC
# Target graph metrics
Metrics = ['Eigenvector', 'Strength', 'Betweenness', 'Clustering']

# Stats loop
for Target in Metrics:
    # Set up target files
    run_1 = '/home/sahib/Documents/OPM_MEG/derivatives/Beta/Beta_run_1/_All_' + Target + '__Beta_run-001.npy'
    run_2 = '/home/sahib/Documents/OPM_MEG/derivatives/Beta/Beta_run_2/_All_' + Target + '__Beta_run-002.npy'
    measure = '_' + Target + '_Beta_'
    out_dir_graphs = '/home/sahib/Documents/OPM_MEG/derivatives/Beta/stats_beta/Graph_stats/'

    # Load data
    run_1 = np.load(run_1)
    run_2 = np.load(run_2)

    # Stats
    T_val, p_val = stats.ttest_rel(run_1, run_2)
    np.nan_to_num(p_val, copy=False, nan=1)
    fdc = stats.false_discovery_control(p_val)

    # Saving
    np.save(out_dir_graphs + measure, fdc)
    np.save(out_dir_graphs + measure + 'P_val', p_val)

# Network based statistics
# Output dir
out_dir_nbs = '/home/sahib/Documents/OPM_MEG/derivatives/Beta/stats_beta/nbs'

# load data
run_1 = np.load('/home/sahib/Documents/OPM_MEG/derivatives/Beta/Beta_run_1/Master_adj_matrix__Beta_run-001.npy')
run_2 = np.load('/home/sahib/Documents/OPM_MEG/derivatives/Beta/Beta_run_2/Master_adj_matrix__Beta_run-002.npy')

np.nan_to_num(run_1, copy=False, nan=0)
np.nan_to_num(run_2, copy=False, nan=0)

# nbs
P_val, adj, null = bct.nbs_bct(run_1.T, run_2.T, thresh= 3.1, k = 5000, paired=True)

np.save(out_dir_nbs + '_P_val', P_val)
np.save(out_dir_nbs + '_adj', adj)
np.save(out_dir_nbs + '_null', null)
