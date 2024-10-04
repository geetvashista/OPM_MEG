import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
from scipy import stats
start = time.time()

# del possible bad data participant
# r1 = np.delete(r1, 0, 0)
# r2 = np.delete(r2, 0, 0)

permutations = 10    # This is the only think that really needs to be changed. default is (10)

model = make_pipeline(SVC(C=0.1, kernel='rbf'))
hold = []

def fun(r1, r2, model, permutations):
    # flatten out participants
    r1 = r1.reshape((2727 + 303), 4, 1, 78)
    r2 = r2.reshape((2727 + 303), 4, 1, 78)
    hold = []
    for run in range(permutations):
        r1 = stats.zscore(r1, axis=-1)
        r2 = stats.zscore(r2, axis=-1)
        # x = np.concatenate((r1, r2), axis=0)
        x = r1
        nineth = len(x) // 10
        y = [0] * nineth + [1] \
            * nineth + [2] \
            * nineth + [3] \
            * nineth + [4] \
            * nineth + [5] \
            * nineth + [6] \
            * nineth + [7] \
            * nineth + [8] * nineth + [9] * nineth

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        results = []
        for i in range(78):
            model.fit(x_train[:, 1, :, i], y_train)
            results.append(model.score(x_test[:, 1, :, i], y_test))
        results = np.array(results)

        areas = []
        for indx, val in enumerate(results):
            if val >= 0.11:
                areas.append(indx)
        if len(areas) == 0:
            areas.append(0)
        model.fit(x_train[:, 1, 0, np.array(areas)], y_train)
        score = model.score(x_test[:, 1, 0, np.array(areas)], y_test)
        print(str(score))
        hold.append(score)
    arr_1 = np.array(hold)

    print('Run complete')

    hold = []
    for run in range(permutations):
        r1 = stats.zscore(r1, axis=-1)
        r2 = stats.zscore(r2, axis=-1)
        # x = np.concatenate((r1, r2), axis=0)
        x = r2
        nineth = len(x) // 10
        y = [0] * nineth + [1] \
            * nineth + [2] \
            * nineth + [3] \
            * nineth + [4] \
            * nineth + [5] \
            * nineth + [6] \
            * nineth + [7] \
            * nineth + [8] * nineth + [9] * nineth

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        results = []
        for i in range(78):
            model.fit(x_train[:, 1, :, i], y_train)
            results.append(model.score(x_test[:, 1, :, i], y_test))
        results = np.array(results)

        areas = []
        for indx, val in enumerate(results):
            if val >= 0.11:
                areas.append(indx)
        if len(areas) == 0:
            areas.append(0)
        model.fit(x_train[:, 1, 0, np.array(areas)], y_train)
        score = model.score(x_test[:, 1, 0, np.array(areas)], y_test)
        print(str(score))
        hold.append(score)
    arr_2 = np.array(hold)

    # print('\n' + 'Mean result: ' + str(arr_2.mean()))
    print('Run complete')
    print('\n' + ' -- Band complete -- ' + '\n')
    return arr_1, arr_2

# Call

    # Theta
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Theta_r1_feature_array.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Theta_r2_feature_array.npy')
t_arr_1, t_arr_2 = fun(r1=r1, r2=r2, model=model, permutations=permutations)

    # Alpha
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Alpha_r1_feature_array.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Alpha_r2_feature_array.npy')
a_arr_1, a_arr_2 = fun(r1=r1, r2=r2, model=model, permutations=permutations)

    # Beta
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Beta_r1_feature_array.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Beta_r2_feature_array.npy')
b_arr_1, b_arr_2 = fun(r1=r1, r2=r2, model=model, permutations=permutations)

    # Gamma
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Gamma_r1_feature_array.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Gamma_r2_feature_array.npy')
g_arr_1, g_arr_2 = fun(r1=r1, r2=r2, model=model, permutations=permutations)

    ### Plotting ###

import pandas as pd
import matplotlib
import seaborn as sns
matplotlib.use('Qt5Agg')


runs = [1] * permutations + [2] * permutations


bands = ['Theta'] * (permutations * 2) + ['Alpha'] * (permutations * 2)
accuracy = (list(t_arr_1) + list(t_arr_2)) + (list(a_arr_1) + list(a_arr_2) + (list(b_arr_1) + list(b_arr_2) + (list(g_arr_1) + list(g_arr_2))

df = pd.DataFrame({'accuracy': accuracy, 'bands': bands, 'runs': (runs + runs + runs + runs)})

print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")

sns.boxplot(data=df, x='bands', y='accuracy', hue="runs", width=.5)
