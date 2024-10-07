import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
from itertools import zip_longest
import pandas as pd
from sklearn.decomposition import PCA
start = time.time()

# Load data
r1 = np.load('/home/students/Ddrive_2TB/Geet/OPM_MEG/Data/features/Alpha_r1_feature_array.npy')
r2 = np.load('/home/students/Ddrive_2TB/Geet/OPM_MEG/Data/features/Alpha_r2_feature_array.npy')

# Prep PCA
def pca(data):
    pca = PCA(n_components=5)
    temp = pca.fit_transform(data)
    return temp[:, 0:2]

# Function to get equalize the length of arrays with nan
def prep_array(org_array, max_len):
    current_coll, current_row = org_array.shape
    empty__array = np.full([max_len, 2], np.nan)
    empty__array[:current_coll, :current_row] = org_array
    return empty__array


def casade_SVC(r1, r2, model, graph_metric):
    metric_names = ['strength', 'betweenness', 'eignvec', 'clustering']
    scores = []
    participant_areas = []
    for i in range(10):
        x = np.concatenate((r1[i, :, :, :, :], r2[i, :, :, :, :]), axis=0)
        half_len = len(x)//2
        y = [0] * half_len + [1] * half_len

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        results = []
        for i in range(78):
            model.fit(x_train[:, graph_metric, :, i], y_train)
            results.append(model.score(x_test[:, graph_metric, :, i], y_test))
        results = np.array(results)

        areas = []
        for indx, val in enumerate(results):
            if val >= 0.52:
                areas.append(indx)
        # These next to line area will append just 1 area if no areas at all can found
        if len(areas) == 0:
            areas.append(0)
        # Performing PCA and taking only the first 2 components
        processed_x_train = pca(x_train[:, graph_metric, 0, np.array(areas)])
        processed_x_test = pca(x_test[:, graph_metric, 0, np.array(areas)])
        model.fit(processed_x_train, y_train)
        participant_areas.append(areas)
        scores.append(model.score(processed_x_test, y_test))
    temp_scores = np.array(scores)
    arr_filled = [list(tpl) for tpl in zip(*zip_longest(*participant_areas))]
    return pd.DataFrame({'Scores': temp_scores, 'Areas': arr_filled}), processed_x_test, processed_x_train

# __main__
model = make_pipeline(SVC(C=0.1, kernel='rbf'))
df_strength, strength_pca_test_data, strength_pca_train_data = casade_SVC(r1, r2, model=model, graph_metric=0)
df_betweenness, betweenness_pca_test_data, betweenness_pca_train_data = casade_SVC(r1, r2, model=model, graph_metric=1)
# df_eignvec = casade_SVC(r1, r2, model=model, graph_metric=2)
# df_clustering = casade_SVC(r1, r2, model=model, graph_metric=3)
# result = pd.concat([df_strength, df_betweenness, df_eignvec,df_clustering], axis=1)

# If saving is desired
# result.to_csv('Alpha_SVC_it3.csv')

print('\n' + "FEATURE EXECUTION TIME: " + str(time.time() - start) + " sec")



import numpy as np
betweenness_pca_train_data = np.load('/home/students/PycharmProjects/EEG/b_pca_train.npy')
h = .02
# create a mesh to plot in
x_min, x_max = betweenness_pca_train_data[:, 0].min() - 1, betweenness_pca_train_data[:, 0].max() + 1
y_min, y_max = betweenness_pca_train_data[:, 1].min() - 1, betweenness_pca_train_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
np.save('Z_var_for_ploting', Z)

