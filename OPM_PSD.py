import numpy as np
import mne

raw_data = np.load('/home/sahib/Documents/OPM_MEG/data/run_1/sub-001_run-001_VE_1_40_Hz_Z.mat_array.npy')
raw_data = raw_data.T

ROI_names = ['Frontal_Sup_Orb_L',
'Frontal_Med_Orb_L',
'Frontal_Mid_Orb_L',
'Frontal_Inf_Orb_L',
'Frontal_Sup_L' ,
'Frontal_Mid_L' ,
'Frontal_Inf_Oper_L' ,
'Frontal_Inf_Tri_L' ,
'Frontal_Sup_Medial_L',
'Supp_Motor_Area_L' ,
'Paracentral_Lobule_L',
'Precentral_L' ,
'Rolandic_Oper_L',
'Postcentral_L' ,
'Parietal_Sup_L' ,
'Parietal_Inf_L' ,
'SupraMarginal_L',
'Angular_L' ,
'Precuneus_L' ,
'Occipital_Sup_L' ,
'Occipital_Mid_L',
'Calcarine_L',
'Cuneus_L' ,
'Lingual_L',
'Heschl_L',
'Temporal_Sup_L',
'Temporal_Mid_L',
'Cingulum_Ant_L',
'Cingulum_Mid_L',
'Cingulum_Post_L',
'Insula_L' ,
'Frontal_Sup_Orb_R',
'Frontal_Med_Orb_R',
'Frontal_Mid_Orb_R',
'Frontal_Inf_Orb_R',
'Frontal_Sup_R',
'Frontal_Mid_R',
'Frontal_Inf_Oper_R',
'Frontal_Inf_Tri_R',
'Frontal_Sup_Medial_R',
'Supp_Motor_Area_R',
'Paracentral_Lobule_R',
'Precentral_R',
'Rolandic_Oper_R',
'Postcentral_R',
'Parietal_Sup_R',
'Parietal_Inf_R',
'SupraMarginal_R',
'Angular_R',
'Precuneus_R',
'Occipital_Sup_R',
'Occipital_Mid_R',
'Calcarine_R',
'Cuneus_R',
'Lingual_R',
'Heschl_R',
'Temporal_Sup_R',
'Temporal_Mid_R',
'Cingulum_Ant_R',
'Cingulum_Mid_R',
'Cingulum_Post_R',
'Insula_R']

info = mne.create_info(ch_names=78,
                       sfreq=1200,
                       ch_types='eeg'
                       )

data = mne.io.RawArray(raw_data, info)
data.info['line_freq'] = 50

remove_ch = [1, 2, 24, 28, 32, 33, 34, 45, 40, 41, 63, 67, 71, 72, 73, 74]

# Set up ch_name mapping
temp = [str(i) for i in range(78)]
for k in remove_ch:
    del temp[k]

for i in remove_ch:
    data.drop_channels(str(i))

ch_names_mapping = dict(zip(temp, ROI_names))

# rename ch_names
mne.rename_channels(data.info, ch_names_mapping)
