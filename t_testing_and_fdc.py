import numpy as np
from scipy import stats

# Target graph metrics
Metrics = ['Eigenvector', 'Strength', 'Betweenness', 'Clustering']

for Target in Metrics:
    # Set up target files
    run_1 = '/home/sahib/Documents/OPM_MEG/derivatives/Alpha/run_1/Beta_All_' + Target + '__alpha_run-001.npy'
    run_2 = '/home/sahib/Documents/OPM_MEG/derivatives/Alpha/run_2/Beta_All_' + Target + '__alpha_run-002.npy'
    type = '_' + Target + '_Alpha_'
    out_put = '/home/sahib/Documents/OPM_MEG/derivatives/Beta/stats_beta/'


    # Load data
    run_1 = np.load(run_1)
    run_2 = np.load(run_2)

    # Stats
    T_val, p_val = stats.ttest_rel(run_1, run_2)
    np.nan_to_num(p_val, copy=False, nan=1)
    fdc = stats.false_discovery_control(p_val)

    # Saving
    np.save(out_put + type, fdc)
    np.save(out_put + type + 'P_val', p_val)
