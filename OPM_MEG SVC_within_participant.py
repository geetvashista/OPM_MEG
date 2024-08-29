import numpy as np
import bct
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time

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

par_result = []
par_areas = []
start = time.time()
model = make_pipeline(SVC(C=0.1))
band = 1    # Pick a target band ([theta, alpha, beta, gamma], across axis one)
metric = 1  # 0: strength, 1: betweenness, 2:eignvec, 3:clustering
for i in range(10):
    # Load data
    print('Now running participant: ' + str(i))
    r1 = np.load(r'/home/sahib/Documents/OPM_MEG/data/Windows_1/Windowed_run_1.npy')
    r2 = np.load(r'/home/sahib/Documents/OPM_MEG/data/Windows_2/Windowed_run_2.npy')

    # Pick a target band ([theta, alpha, beta, gamma], across axis one)
    r1 = r1[band, :, :, :, :]
    r2 = r2[band, :, :, :, :]
    r1 = r1[i, :, :, :]
    r2 = r2[i, :, :, :]

    # Extract features
    run_1_featurs, run_2_featurs = feature_extraction(r1, r2)

    # Join the arrays
    x = np.concatenate((run_1_featurs, run_2_featurs), axis=0)

    half_len = len(x)//2
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
        if val >= 0.53:
            areas.append(indx)
    model.fit(x_train[:, metric, 0, np.array(areas)], y_train)
    par_result.append(model.score(x_test[:, metric, 0, np.array(areas)], y_test))
    par_areas.append(areas)

array_results = np.array(par_result)
print(array_results)
max_lg = [len(i) for i in par_areas]
pad = [lst + [np.nan]*(max(max_lg) - len(lst)) for lst in par_areas]
array_areas = np.array(pad)
print('\n' + "FINAL EXECUTION TIME: " + str(time.time() - start) + " sec")
