import pandas as pd
import numpy as np

# set paths and load data

Strength = np.load('/home/sahib/Documents/OPM_MEG/derivatives/Theta/stats_Theta/Graph_stats/Graph_stats_Strength_Theta_P_val.npy')
betweenness = np.load('/home/sahib/Documents/OPM_MEG/derivatives/Theta/stats_Theta/Graph_stats/Graph_stats_Betweenness_Theta_P_val.npy')
eigenvec = np.load('/home/sahib/Documents/OPM_MEG/derivatives/Theta/stats_Theta/Graph_stats/Graph_stats_Eigenvector_Theta_P_val.npy')
clustering = np.load('/home/sahib/Documents/OPM_MEG/derivatives/Theta/stats_Theta/Graph_stats/Graph_stats_Clustering_Theta_P_val.npy')
output = '/home/sahib/Documents/OPM_MEG/derivatives/Theta/stats_Theta/sig_node_properties.xlsx'

def extract_indices_vals(array):
    """
    This function extracts the index and P_val of
    all significant nodes.

    :param array: shaped as a 1D numpy object
    :returns: 2D array shaped (ROI_indies, P_val)
    """
    ind_list = []
    val_list = []
    for i, val in enumerate(array):
        if val <= 0.05:
            ind_list.append(i)
            val_list.append(val)
    Sig_info = np.stack([ind_list, val_list])
    return Sig_info.T

# Get indices and P vals
Strength_info = extract_indices_vals(Strength)
Betweenness_info = extract_indices_vals(betweenness)
Eigenvector_info = extract_indices_vals(eigenvec)
Clustering_info = extract_indices_vals(clustering)

# Clean up
del Strength
del betweenness
del eigenvec
del clustering

# Get the max length for sig ROI's
max_len = max(len(Strength_info), len(Betweenness_info), len(Eigenvector_info), len(Clustering_info))

def prep_array(org_array, max_len):
    """
    Function to set all arrays to the same length,
    filling extra entries with nan

    :param org_array: Array to extend, shaped as 2D (ROI_indies, P_vals)
    :param max_len: Int, the length to which org_array will be extended too.
    :return: 2D array set to the max_len length
    """
    current_coll, current_row = org_array.shape
    empty__array = np.full([max_len, 2], np.nan)
    empty__array[:current_coll, :current_row] = org_array
    return empty__array


# Set up data as Pandas DataFrames
df1 = pd.DataFrame(prep_array(Strength_info, max_len), columns=['Strength_ROI_index', 'Strength_Val'])
df2 = pd.DataFrame(prep_array(Betweenness_info, max_len), columns=['Betweenness_ROI_index', 'Betweenness_Val'])
df3 = pd.DataFrame(prep_array(Eigenvector_info, max_len), columns=['Eigenvector_ROI_index', 'Eigenvector_Val'])
df4 = pd.DataFrame(prep_array(Clustering_info, max_len), columns=['Clustering_ROI_index', 'Clustering_Val'])

# Concatenate the DataFrames horizontally (axis=1)
df_combined = pd.concat([df1, df2, df3, df4], axis=1)
df_combined.to_excel(output, index=False)   # For saving as .xlsx file
