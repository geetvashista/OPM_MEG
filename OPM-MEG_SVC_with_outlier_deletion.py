import numpy as np
import bct
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
start = time.time()

# Load data
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Theta_r1_feature_array.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Theta_r2_feature_array.npy')

# del possible bad data participant
r1 = np.delete(r1, 0, 0)
r2 = np.delete(r2, 0, 0)

# flatten out participants
r1 = r1.reshape(2727, 4, 1, 78)
r2 = r2.reshape(2727, 4, 1, 78)

model = make_pipeline(SVC(C=0.1, kernel='rbf'))
hold = []
for run in range(10):
    x = np.concatenate((r1, r2), axis=0)
    half_len = len(x)//2
    y = [0] * half_len + [1] * half_len

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    results = []
    for i in range(78):
        model.fit(x_train[:, 1, :, i], y_train)
        results.append(model.score(x_test[:, 1, :, i], y_test))
    results = np.array(results)

    areas = []
    for indx, val in enumerate(results):
        if val >= 0.52:
            areas.append(indx)
    model.fit(x_train[:, 1, 0, np.array(areas)], y_train)
    score = model.score(x_test[:, 1, 0, np.array(areas)], y_test)
    print(str(score))
    hold.append(score)
arr = np.array(hold)

print('\n' + 'Mean result: ' + str(arr.mean()))
print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")
