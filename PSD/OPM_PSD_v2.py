import numpy as np
import mne
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Setting some backend parameters
matplotlib.use('TkAgg')


def get_mean_psd(file_path):
    raw_data = np.load(file_path)
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

    for i in remove_ch:
        data.drop_channels(str(i))

    temp = data.ch_names
    ch_names_mapping = dict(zip(temp, ROI_names))

    # rename ch_names
    mne.rename_channels(data.info, ch_names_mapping)

    # picks= [i for i in range(4)]
    fig = data.compute_psd(fmin=1, fmax=30, average='mean')
    temp = fig.get_data()

    epsilon = 1e-10
    to_plot = np.mean(temp, axis=0)
    to_plot = 10 * np.log10(np.maximum(to_plot, epsilon) / 1)
    # to_plot = np.array(hold)
    return to_plot


def plot(participant):
    # Get the data we need
    run_1 = get_mean_psd(r'C:\Users\em17531\Desktop\OPM_MEG\data\run_1\sub-00' + str(participant) + '_run-001_VE_1_40_Hz_Z'
                                                                                               '.mat_array.npy')
    run_2 = get_mean_psd(r'C:\Users\em17531\Desktop\OPM_MEG\data\run_2\sub-00' + str(participant) + '_run-002_VE_1_40_Hz_Z'
                                                                                               '.mat_array.npy')

    # Plotting
    df = pd.DataFrame({'Run 1': run_1, 'Run 2': run_2})
    sns.lineplot(df).set_title(f'Participant {participant}')


def get_mean_mne_psd(file_path, ax):
    raw_data = np.load(file_path)
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

    for i in remove_ch:
        data.drop_channels(str(i))

    temp = data.ch_names
    ch_names_mapping = dict(zip(temp, ROI_names))

    # rename ch_names
    mne.rename_channels(data.info, ch_names_mapping)

    # picks= [i for i in range(4)]
    # fig = data.compute_psd(fmin=1, fmax=30, average='mean')

    return data.compute_psd(fmin=1, fmax=30, average='mean', ax=ax)


def plot_mne(participant):    # TO DO: get this working, and add title adding ability
    # Get the data we need
    # run_1 = get_mean_mne_psd(r'C:\Users\em17531\Desktop\OPM_MEG\data\run_1\sub-00' + str(participant) + '_run-001_VE_1_40_Hz_Z'
    #                                                                                            '.mat_array.npy')
    # run_2 = get_mean_mne_psd(r'C:\Users\em17531\Desktop\OPM_MEG\data\run_2\sub-00' + str(participant) + '_run-002_VE_1_40_Hz_Z'
    #                                                                                            '.mat_array.npy')

    # Plotting
    # run_1.plot(average=True).set_title('foo')

    fig, ax = plt.subplots(1)
    run_1 = get_mean_mne_psd(
        r'C:\Users\em17531\Desktop\OPM_MEG\data\run_1\sub-00' + str(participant) + '_run-001_VE_1_40_Hz_Z'
                                                                                   '.mat_array.npy', ax)
    ax[0].set_title('foo')
    fig.tight_layout()
    fig.show()

# __Main__

plot_mne(1)
