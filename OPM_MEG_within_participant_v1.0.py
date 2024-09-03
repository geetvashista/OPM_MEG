import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
start = time.time()


# Load data
r1 = np.load(r'C:\Users\em17531\Downloads\Theta_r1_feature_array.npy')
r2 = np.load(r'C:\Users\em17531\Downloads\Theta_r2_feature_array.npy')


model = make_pipeline(SVC(C=0.1, kernel='rbf'))
scores = []
for i in range(10):
    x = np.concatenate((r1[i, :, :, :, :], r2[i, :, :, :, :]), axis=0)
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
    scores.append(model.score(x_test[:, 1, 0, np.array(areas)], y_test))
[print(k) for k in scores]
print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")

# temp_array = np.array(scores[1:9])
# print('\n' + 'Mean without participant 1: ' + str(temp_array.mean()))
