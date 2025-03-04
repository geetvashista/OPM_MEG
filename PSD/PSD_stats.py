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


# Let make our run averages
run_1_mean_areas = psd_run_1.mean(axis=-1).mean(0)
run_2_mean_areas = psd_run_2.mean(axis=-1).mean(0)

p_vals_for_df = []
for i in sig_index:
    p_vals_for_df.append(p_vals[i])


def val_for_df(list_ob, sig_index):
    roi_for_df = []
    for i in sig_index:
        roi_for_df.append(list_ob[i])
    return np.array(roi_for_df)


df = pd.DataFrame({'Area names': names_for_df,
                   'P_vals': p_vals_for_df,
                   'Run_1': val_for_df(run_1_mean_areas, sig_index),
                   'Run_2': val_for_df(run_2_mean_areas, sig_index)})


# Let's get participant level info

run_1_with_participants = psd_run_1.mean(axis=-1)
run_2_with_participants = psd_run_2.mean(axis=-1)


def PSD_participant_vals(run_with_participants, sig_index):
    hold = []
    for i in sig_index:
        hold.append(run_with_participants[i])
    return hold

all_run_1_participant_vals = []
for i in range(10):
    all_run_1_participant_vals.append(PSD_participant_vals(run_1_with_participants[i], sig_index))
all_run_1_participant_vals = np.array(all_run_1_participant_vals)

all_run_2_participant_vals = []
for i in range(10):
    all_run_2_participant_vals.append(PSD_participant_vals(run_2_with_participants[i], sig_index))
all_run_2_participant_vals = np.array(all_run_2_participant_vals)


def plot_individual_pie(area_index):

    def get_ratio(run_1_array, run_2_array):
        ratio = []
        for i in range(10):
            r1 = run_1_array[i]
            r2 = run_2_array[i]
            if r1 >= r2:
                ratio.append(0)
            else:
                ratio.append(1)
        return ratio


    run_1_array = all_run_1_participant_vals[:, area_index]
    run_2_array = all_run_2_participant_vals[:, area_index]
    area_ratio = get_ratio(run_1_array, run_2_array)

    area_ratio = np.array(area_ratio)
    area_ratio.sort()

    run_1_more_power = 0
    run_2_more_power = 0
    for i in area_ratio:
        if i == 0:
            run_1_more_power += 1

    for i in area_ratio:
        if i > 0:
            run_2_more_power += 1

    pie_data = np.array([run_1_more_power, run_2_more_power])
    labels = ['Higher run 1 power', 'Higher run 2 power']

    plt.pie(pie_data, labels = labels, autopct='%.1f%%')
    plt.title(ROI_names[sig_index[area_index]])
    plt.show()
