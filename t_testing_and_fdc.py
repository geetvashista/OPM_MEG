import numpy as np
from scipy import stats

# Set up target files
run_1 = '/home/sahib/Documents/OPM_MEG/Graph_outputs/Alpha_run_1/Graph_output_All_Strength_1_VE_1_40_Hz_Z.mat_array.npy'
run_2 = '/home/sahib/Documents/OPM_MEG/Graph_outputs/Alpha_run_2/Alpha_run_2_All_Strength_alpha_run_2.npy'
type = '_Clustering_alpha_'
out_put = '/home/sahib/Documents/OPM_MEG/stats_alpha/'


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
