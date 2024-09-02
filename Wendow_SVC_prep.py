import numpy as np
import bct
import time
start = time.time()

# Data pathing and loading
r1 = np.load(r'/home/sahib/Documents/OPM_MEG/data/Windows_1/Windowed_run_1.npy')
r2 = np.load(r'/home/sahib/Documents/OPM_MEG/data/Windows_2/Windowed_run_2.npy')

# Graph calculator function
def graph_metrics(adj_matrix):  # TODO: Add in stats, maybe nbs? Could use fdc as well?
    # Strength calculator
    Strength = []
    Strength.append(bct.strengths_und(np.nan_to_num(adj_matrix)))
    Strength = np.array(Strength)

    # Zeroing negative phasing
    Strength[Strength < 0] = 0

    # Betweenness centrality calculator
    Betweenness = []
    Betweenness.append(bct.betweenness_wei(np.nan_to_num(adj_matrix)))
    Betweenness = np.array(Betweenness)

    # Eigenvector centrality calculator
    Eigenvector = []
    Eigenvector.append(bct.eigenvector_centrality_und(np.nan_to_num(adj_matrix)))
    Eigenvector = np.array(Eigenvector)

    # Clustering calculator
    Clustering = []
    Clustering.append(bct.clustering_coef_wu(np.nan_to_num(adj_matrix)))
    Clustering = np.array(Clustering)

    return Strength, Betweenness, Eigenvector, Clustering


# feature extraction function
def feature_extraction(r1, r2):
    r1_array = []
    for sample in r1:
        features = []
        r1_S, r1_B, r1_E, r1_C = graph_metrics(sample)
        features.append(r1_S)
        features.append(r1_B)
        features.append(r1_E)
        features.append(r1_C)
        sample_features = np.array(features)
        r1_array.append(sample_features)

    r2_array = []
    for sample in r2:
        features = []
        r2_S, r2_B, r2_E, r2_C = graph_metrics(sample)
        features.append(r2_S)
        features.append(r2_B)
        features.append(r2_E)
        features.append(r2_C)
        sample_features = np.array(features)
        r2_array.append(sample_features)
        hold = []
        hold.append(sample)
    print('Participant ' + len(hold) + 'feature extraction complete')
    return np.array(r1_array), np.array(r2_array)


# Get band and metric data
def data_extraction(r1, r2, band, participant):
    r1 = r1[band, :, :, :, :]
    r2 = r2[band, :, :, :, :]
    r1 = r1[participant, :, :, :]
    r2 = r2[participant, :, :, :]
    return r1, r2

# Theta
band = 0
all_participants_r1 = []
all_participants_r2 = []
for participant in range(10):
    input_1, input_2 = data_extraction(r1, r2, band=band, participant=participant)
    all_participants_r1.append(input_1)
    all_participants_r2.append(input_2)

r1_array = np.array(all_participants_r1)
r2_array = np.array(all_participants_r1)

participant_features_r1 = []
participant_features_r2 = []

for i in range(10):
    pf1, pf2 = feature_extraction(r1_array, r2_array)

thata_r1_feature_array = np.array(pf1)
thata_r2_feature_array = np.array(pf2)

np.save('thata_r1_feature_array', thata_r1_feature_array)
np.save('thata_r2_feature_array', thata_r2_feature_array)

# Alpha
band = 1
all_participants_r1 = []
all_participants_r2 = []
for participant in range(10):
    input_1, input_2 = data_extraction(r1, r2, band=band, participant=participant)
    all_participants_r1.append(input_1)
    all_participants_r2.append(input_2)

r1_array = np.array(all_participants_r1)
r2_array = np.array(all_participants_r1)

participant_features_r1 = []
participant_features_r2 = []

for i in range(10):
    pf1, pf2 = feature_extraction(r1_array, r2_array)

Alpha_r1_feature_array = np.array(pf1)
Alpha_r2_feature_array = np.array(pf2)

np.save('Alpha_r1_feature_array', Alpha_r1_feature_array)
np.save('Alpha_r2_feature_array', Alpha_r2_feature_array)

# Beta
band = 2
all_participants_r1 = []
all_participants_r2 = []
for participant in range(10):
    input_1, input_2 = data_extraction(r1, r2, band=band, participant=participant)
    all_participants_r1.append(input_1)
    all_participants_r2.append(input_2)

r1_array = np.array(all_participants_r1)
r2_array = np.array(all_participants_r1)

participant_features_r1 = []
participant_features_r2 = []

for i in range(10):
    pf1, pf2 = feature_extraction(r1_array, r2_array)

Beta_r1_feature_array = np.array(pf1)
Beta_r2_feature_array = np.array(pf2)

np.save('Beta_r1_feature_array', Beta_r1_feature_array)
np.save('Beta_r2_feature_array', Beta_r2_feature_array)


# Gamma
band = 3
all_participants_r1 = []
all_participants_r2 = []
for participant in range(10):
    input_1, input_2 = data_extraction(r1, r2, band=band, participant=participant)
    all_participants_r1.append(input_1)
    all_participants_r2.append(input_2)

r1_array = np.array(all_participants_r1)
r2_array = np.array(all_participants_r1)

participant_features_r1 = []
participant_features_r2 = []

for i in range(10):
    pf1, pf2 = feature_extraction(r1_array, r2_array)

thata_r1_feature_array = np.array(pf1)
thata_r2_feature_array = np.array(pf2)

np.save('Beta_r1_feature_array', thata_r1_feature_array)
np.save('Beta_r2_feature_array', thata_r2_feature_array)

print('\n' + "FINAL EXECUTION TIME: " + str(time.time() - start) + " sec")
