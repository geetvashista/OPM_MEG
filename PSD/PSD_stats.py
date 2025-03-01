import numpy as np
import mne
import matplotlib
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
start = time.time()

# Setting some backend parameters
matplotlib.use('TkAgg')

# Needed ROI names
ROI_names = ['Frontal_Sup_Orb_L',
             'Frontal_Med_Orb_L',
             'Frontal_Mid_Orb_L',
             'Frontal_Inf_Orb_L',
             'Frontal_Sup_L',
             'Frontal_Mid_L',
             'Frontal_Inf_Oper_L',
             'Frontal_Inf_Tri_L',
             'Frontal_Sup_Medial_L',
             'Supp_Motor_Area_L',
             'Paracentral_Lobule_L',
             'Precentral_L',
             'Rolandic_Oper_L',
             'Postcentral_L',
             'Parietal_Sup_L',
             'Parietal_Inf_L',
             'SupraMarginal_L',
             'Angular_L',
             'Precuneus_L',
             'Occipital_Sup_L',
             'Occipital_Mid_L',
             'Calcarine_L',
             'Cuneus_L',
             'Lingual_L',
             'Heschl_L',
             'Temporal_Sup_L',
             'Temporal_Mid_L',
             'Cingulum_Ant_L',
             'Cingulum_Mid_L',
             'Cingulum_Post_L',
             'Insula_L',
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


def get_data(file_path, ROI_names):
    raw_data = np.load(file_path)
    raw_data = raw_data.T

    info = mne.create_info(ch_names=78,
                           sfreq=1200,
                           ch_types='eeg'
                           )

    data = mne.io.RawArray(raw_data, info)
    data.info['line_freq'] = 50

    remove_ch = [1, 2, 24, 28, 32, 33, 34, 45, 40, 41, 63, 67, 71, 72, 73, 74]

    for i in remove_ch:
        data.drop_channels(str(i))

    temp = data.ch_names
    ch_names_mapping = dict(zip(temp, ROI_names))

    # rename ch_names
    mne.rename_channels(data.info, ch_names_mapping)

    # picks= [i for i in range(4)]
    # psd_spec = data.compute_psd(fmin=7.9, fmax=12., average='mean')
    psd_spec = data.compute_psd(fmin=7.9, fmax=12.)
    return  psd_spec


def get_mean_psd_data(participant, run):
    if participant <= 9.:
        file_path = r'C:\Users\em17531\Desktop\OPM_MEG\data' \
                    r'\run_' + str(run) + '\sub-00' + str(participant) + \
                    '_run-00' + str(run) + '_VE_1_40_Hz_Z.mat_array.npy'

        psd_data = get_data(file_path, ROI_names=ROI_names)
        psd_data = psd_data.get_data()
        return psd_data
    elif participant >= 9.:
        file_path = r'C:\Users\em17531\Desktop\OPM_MEG\data' \
                    r'\run_' + str(run) + '\sub-0' + str(participant) + \
                    '_run-00' + str(run) + '_VE_1_40_Hz_Z.mat_array.npy'

        psd_data = get_data(file_path, ROI_names=ROI_names)
        psd_data = psd_data.get_data()
        return psd_data


# __Main__

run_1 = []
for i in range(1, 11):
    run_1.append(get_mean_psd_data(i, 1))

run_2 = []
for i in range(1, 11):
    run_2.append(get_mean_psd_data(i, 2))

psd_run_1 = np.array(run_1)
psd_run_2 = np.array(run_2)
del run_1
del run_2

stat_output = stats.ttest_rel(psd_run_1, psd_run_2)
p_vals = stat_output.pvalue.T

print('\n' + 'P_values: ' + '\n')
[print(i) for i in p_vals]
print('\n'+'Execution time: ' + str(time.time() - start) + ' sec')

sig_index = []
for key, val in enumerate(p_vals):
    if val <= 0.05:
        sig_index.append(key)

for i in sig_index:
    print(ROI_names[i])

