import pandas as pd
import numpy as np

# set paths and load data

nodes_with_sig_edges = pd.read_excel('/home/sahib/Documents/OPM_MEG/derivatives/Nodes_with_sig_edges.xlsx')

Strength = np.load('/home/sahib/Documents/OPM_MEG/derivatives/Theta/stats_Theta/Graph_stats/Graph_stats_Strength_Theta_P_val.npy')
betweenness = np.load('/home/sahib/Documents/OPM_MEG/derivatives/Theta/stats_Theta/Graph_stats/Graph_stats_Betweenness_Theta_P_val.npy')
eigenvec = np.load('/home/sahib/Documents/OPM_MEG/derivatives/Theta/stats_Theta/Graph_stats/Graph_stats_Eigenvector_Theta_P_val.npy')
clustering = np.load('/home/sahib/Documents/OPM_MEG/derivatives/Theta/stats_Theta/Graph_stats/Graph_stats_Clustering_Theta_P_val.npy')
output = '/home/sahib/Documents/OPM_MEG/derivatives/Theta/stats_Theta/sig_node_properties.xlsx'

def extract_indices_vals(array):
    ind_list = []
    val_list = []
    for i, val in enumerate(array):
        if val <= 0.05:
            ind_list.append(i)
            val_list.append(val)
    Sig_info = np.stack([ind_list, val_list])
    return Sig_info.T

Strength_info = extract_indices_vals(Strength)
Betweenness_info = extract_indices_vals(betweenness)
Eigenvector_info = extract_indices_vals(eigenvec)
Clustering_info = extract_indices_vals(clustering)
del Strength
del betweenness
del eigenvec
del clustering

max_len = max(len(Strength_info), len(Betweenness_info), len(Eigenvector_info), len(Clustering_info))

def prep_array(org_array, max_len):
    current_coll, current_row = org_array.shape
    empty__array = np.full([max_len, 2], np.nan)
    empty__array[:current_coll, :current_row] = org_array
    return empty__array

df1 = pd.DataFrame(prep_array(Strength_info, max_len), columns=['Strength_ROI_index', 'Strength_Val'])
df2 = pd.DataFrame(prep_array(Betweenness_info, max_len), columns=['Betweenness_ROI_index', 'Betweenness_Val'])
df3 = pd.DataFrame(prep_array(Eigenvector_info, max_len), columns=['Eigenvector_ROI_index', 'Eigenvector_Val'])
df4 = pd.DataFrame(prep_array(Clustering_info, max_len), columns=['Clustering_ROI_index', 'Clustering_Val'])
# Concatenate the DataFrames horizontally (axis=1)
df_combined = pd.concat([df1, df2, df3, df4], axis=1)

df_combined.to_excel(output, index=False)
