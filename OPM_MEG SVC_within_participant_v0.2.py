import numpy as np
import bct
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time

start = time.time()

# Data pathing and loading
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windows_1\Windowed_1.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windows_2\Windowed_2.npy')

# Pep parameters
model = make_pipeline(SVC(C=0.1))
output = '/home/sahib/Documents/OPM_MEG/Results/'
band_names = ['Theta', 'Alpha', 'Beta', 'Gamma']
band = [0, 1, 2, 3]
sig_val = 0.53

# Output files
strength_output = ''
betweenness_output = ''
eigenvec_output = ''
clustering_output = ''


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
    # start = time.time()
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
    # print('\n' + "FEATURE EXECUTION TIME: " + str(time.time() - start) + " sec")
    print('\n')
    return np.array(r1_array), np.array(r2_array)


# Get band and metric data
def data_extraction(r1, r2, band, participant):
    r1 = r1[band, :, :, :, :]
    r2 = r2[band, :, :, :, :]
    r1 = r1[participant, :, :, :]
    r2 = r2[participant, :, :, :]
    return r1, r2


# Testing SCV
def test_SVC(input_1, input_2, metric, sig_val):
    # Get features
    run_1_features, run_2_features = feature_extraction(input_1, input_2)

    # Join the arrays
    x = np.concatenate((run_1_features, run_2_features), axis=0)
    half_len = len(x) // 2
    y = [0] * half_len + [1] * half_len
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    results = []
    for k in range(78):
        model.fit(x_train[:, metric, :, k], y_train)
        results.append(model.score(x_test[:, metric, :, k], y_test))
    results = np.array(results)

    areas = []
    for indx, val in enumerate(results):
        if val >= sig_val:
            areas.append(indx)
    if len(areas) == 0:
        areas = areas.append(0)
    model.fit(x_train[:, metric, 0, np.array(areas)], y_train)
    score = model.score(x_test[:, metric, 0, np.array(areas)], y_test)

    return score, areas


def pad_list(area_lst):
    max_lg = [len(i) for i in area_lst]
    pad = [lst + [np.nan] * (max(max_lg) - len(lst)) for lst in par_areas]
    return np.array(pad)


# __main__
# Strength
print('Calculating Strength' + '\n')
for band in band:
    total_scores = []
    all_areas = []
    for participant in range(10):
        input_1, input_2 = data_extraction(r1, r2, band=band, participant=participant)
        par_score, par_areas = test_SVC(input_1, input_2, metric=0, sig_val=sig_val)
        total_scores.append(par_score)
        all_areas.append(par_areas)
    array_total_scores = np.array(total_scores)
    array_all_areas = pad_list(all_areas)

    # Saving files
    np.save(strength_output + 'Strength_scours_' + str(band_names[band]), array_total_scores)
    np.save(strength_output + 'Strength_areas_' + str(band_names[band]), array_all_areas)

del total_scores
del all_areas
del array_total_scores
del array_all_areas

# Betweenness
print('Calculating Betweenness' + '\n')
for band in band:
    total_scores = []
    all_areas = []
    for participant in range(10):
        input_1, input_2 = data_extraction(r1, r2, band=band, participant=participant)
        test_SVC(input_1, input_2, metric=1, sig_val=sig_val)
        total_scores.append(par_score)
        all_areas.append(par_areas)
    array_total_scores = np.array(total_scores)
    array_all_areas = pad_list(all_areas)

    # Saving files
    np.save(betweenness_output + 'Betweenness_scours_' + str(band_names[band]), array_total_scores)
    np.save(betweenness_output + 'Betweenness_areas_' + str(band_names[band]), array_all_areas)

del total_scores
del all_areas
del array_total_scores
del array_all_areas

# Eigenvec
print('Calculating Eigenvec' + '\n')
for band in band:
    for participant in range(10):
        input_1, input_2 = data_extraction(r1, r2, band=band, participant=participant)
        test_SVC(input_1, input_2, metric=2, sig_val=sig_val)
        total_scores.append(par_score)
        all_areas.append(par_areas)
    array_total_scores = np.array(total_scores)
    array_all_areas = pad_list(all_areas)

    # Saving files
    np.save(eigenvec_output + 'Eigenvec_scours_' + str(band_names[band]), array_total_scores)
    np.save(eigenvec_output + 'Eigenvec_areas_' + str(band_names[band]), array_all_areas)

del total_scores
del all_areas
del array_total_scores
del array_all_areas

# Clustering
print('Calculating Clustering' + '\n')
for band in band:
    for participant in range(10):
        input_1, input_2 = data_extraction(r1, r2, band=band, participant=participant)
        test_SVC(input_1, input_2, metric=3, sig_val=sig_val)
        total_scores.append(par_score)
        all_areas.append(par_areas)
    array_total_scores = np.array(total_scores)
    array_all_areas = pad_list(all_areas)

    # Saving files
    np.save(clustering_output + 'Clustering_scours_' + str(band_names[band]), array_total_scores)
    np.save(clustering_output + 'Clustering_areas_' + str(band_names[band]), array_all_areas)

print('\n' + "FINAL EXECUTION TIME: " + str(time.time() - start) + " sec")
