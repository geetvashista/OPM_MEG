# ROI_names = ['Frontal_Sup_Orb_L',
# 'Frontal_Med_Orb_L',
# 'Frontal_Mid_Orb_L',
# 'Frontal_Inf_Orb_L',
# 'Frontal_Sup_L' ,
# 'Frontal_Mid_L' ,
# 'Frontal_Inf_Oper_L' ,
# 'Frontal_Inf_Tri_L' ,
# 'Frontal_Sup_Medial_L',
# 'Supp_Motor_Area_L' ,
# 'Paracentral_Lobule_L',
# 'Precentral_L' ,
# 'Rolandic_Oper_L',
# 'Postcentral_L' ,
# 'Parietal_Sup_L' ,
# 'Parietal_Inf_L' ,
# 'SupraMarginal_L',
# 'Angular_L' ,
# 'Precuneus_L' ,
# 'Occipital_Sup_L' ,
# 'Occipital_Mid_L',
# 'Calcarine_L',
# 'Cuneus_L' ,
# 'Lingual_L',
# 'Heschl_L',
# 'Temporal_Sup_L',
# 'Temporal_Mid_L',
# 'Cingulum_Ant_L',
# 'Cingulum_Mid_L',
# 'Cingulum_Post_L',
# 'Insula_L' ,
# 'Frontal_Sup_Orb_R',
# 'Frontal_Med_Orb_R',
# 'Frontal_Mid_Orb_R',
# 'Frontal_Inf_Orb_R',
# 'Frontal_Sup_R',
# 'Frontal_Mid_R',
# 'Frontal_Inf_Oper_R',
# 'Frontal_Inf_Tri_R',
# 'Frontal_Sup_Medial_R',
# 'Supp_Motor_Area_R',
# 'Paracentral_Lobule_R',
# 'Precentral_R',
# 'Rolandic_Oper_R',
# 'Postcentral_R',
# 'Parietal_Sup_R',
# 'Parietal_Inf_R',
# 'SupraMarginal_R',
# 'Angular_R',
# 'Precuneus_R',
# 'Occipital_Sup_R',
# 'Occipital_Mid_R',
# 'Calcarine_R',
# 'Cuneus_R',
# 'Lingual_R',
# 'Heschl_R',
# 'Temporal_Sup_R',
# 'Temporal_Mid_R',
# 'Cingulum_Ant_R',
# 'Cingulum_Mid_R',
# 'Cingulum_Post_R',
# 'Insula_R']


import pandas as pd
import numpy as np
# df = pd.read_excel(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Gamma\stats_Gamma\Gamma_sig_node_properties.xlsx')
#
# def name_ROI_by_index(df, metric, ROI_names):
#     new = []
#     for i, val in enumerate(df[str(metric + '_ROI_index')]):
#         if val < 62:
#             new.append(ROI_names[(int(val))])
#         else:
#             new.append(val)
#     return new
#
# # Rename values
# df['Strength_ROI_index'] = name_ROI_by_index(df, 'Strength', ROI_names)
# df['Betweenness_ROI_index'] = name_ROI_by_index(df, 'Betweenness', ROI_names)
# df['Eigenvector_ROI_index'] = name_ROI_by_index(df, 'Eigenvector', ROI_names)
# df['Clustering_ROI_index'] = name_ROI_by_index(df, "Clustering", ROI_names)
#
# df.to_excel('Gamma_graph_results.xlsx')






df = pd.read_excel(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Theta\stats_Theta\Theta_sig_node_properties.xlsx')

def get_vals(df, metric, run_1, run_2):
    def mean(run):
        run = np.mean(run, axis=0)
        return run
    run_1 = mean(run_1)
    run_2 = mean(run_2)

    r1_val = []
    r2_val = []
    for roi in df[str(metric + '_ROI_index')]:
        if float(roi) > 0:
            r1_val.append(run_1[int(roi)])
            r2_val.append(run_2[int(roi)])
        else:
            r1_val.append(roi)
            r2_val.append(roi)
    return pd.DataFrame({'R1': r1_val, 'R2': r2_val})


# Strength
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Theta\run_1\_All_Strength__Theta_run-001.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Theta\run_2\_All_Strength__Theta_run-002.npy')
S_df = get_vals(df, 'Strength', run_1=r1, run_2=r2)

# Betweenness
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Gamma\run_1\_All_Betweenness__Gamma_run-001.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Gamma\run_2\_All_Betweenness__Gamma_run-002.npy')
B_df = get_vals(df, 'Betweenness', run_1=r1, run_2=r2)

# Eigenvector
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Gamma\run_1\_All_Eigenvector__Gamma_run-001.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Gamma\run_2\_All_Eigenvector__Gamma_run-002.npy')
E_df = get_vals(df, 'Eigenvector', run_1=r1, run_2=r2)

# Clustering
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Theta\run_1\_All_Clustering__Theda_run-001.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Theta\run_2\_All_Clustering__Theda_run-002.npy')
C_df = get_vals(df, "Clustering", run_1=r1, run_2=r2)
del r1
del r2
