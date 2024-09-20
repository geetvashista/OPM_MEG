import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
from itertools import zip_longest
import pandas as pd
start = time.time()

# Load data
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Theta_r1_feature_array.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Theta_r2_feature_array.npy')

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
        model.fit(x_train[:, graph_metric, 0, np.array(areas)], y_train)
        participant_areas.append(areas)
        scores.append(model.score(x_test[:, graph_metric, 0, np.array(areas)], y_test))
    temp_scores = np.array(scores)
    arr_filled = [list(tpl) for tpl in zip(*zip_longest(*participant_areas))]

    return pd.DataFrame({'Scores': temp_scores, 'Areas': arr_filled})

# __main__
model = make_pipeline(SVC(C=0.1, kernel='rbf'))
# df_strength = casade_SVC(r1, r2, model=model, graph_metric=0)
df_betweenness = casade_SVC(r1, r2, model=model, graph_metric=1)
# df_eignvec = casade_SVC(r1, r2, model=model, graph_metric=2)
# df_clustering = casade_SVC(r1, r2, model=model, graph_metric=3)
# result = pd.concat([df_strength, df_betweenness, df_eignvec,df_clustering], axis=1)

# If saving is desired
# result.to_csv('Alpha_SVC_it3.csv')

print('\n' + "FEATURE EXECUTION TIME: " + str(time.time() - start) + " sec")
