import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
start = time.time()


# Load data
r1 = np.load(r'C:\Users\em17531\Downloads\Theta_r1_feature_array.npy')
r2 = np.load(r'C:\Users\em17531\Downloads\Theta_r2_feature_array.npy')



from matplotlib import pyplot as plt
import matplotlib
start = time.time()
# Set back ends of viz
matplotlib.use('Qt5Agg')

r1 = np.load(r'C:\Users\em17531\Downloads\Theta_r1_feature_array.npy')
r2 = np.load(r'C:\Users\em17531\Downloads\Theta_r2_feature_array.npy')


model = make_pipeline(SVC(C=0.1, kernel='rbf'))

x = np.concatenate((r1[2, :, :, :, :], r2[2, :, :, :, :]), axis=0)
half_len = len(x) // 2
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



temp_train = x_train[:, 1, 0, np.array(areas)]
# temp_train = temp_train[:, 5:7]
#
temp_test = x_test[:, 1, 0, np.array(areas)]
# temp_test = temp_test[:, 5:7]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
temp_train = pca.fit_transform(temp_train)
temp_test = pca.fit_transform(temp_test)


model.fit(temp_train, y_train)
print('After PCA model score is:' + '\n')
print(model.score(temp_test, y_test))

h = .02

# For mamory perpaces, del the first windows
# ls = []
# [ls.append(i) for i in range(400)]
# temp_train_1 = np.delete(temp_train, axis=0, obj=ls)

_, ax = plt.subplots(figsize=(4, 3))
x_min, x_max, y_min, y_max = temp_train[:, 0].min(), temp_train[:, 0].max(), temp_train[:, 1].min(), temp_train[:, 1].max()
ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

# Plot decision boundary and margins
common_params = {"estimator": model, "X": temp_train, "ax": ax}
DecisionBoundaryDisplay.from_estimator(
    **common_params,
    response_method="predict",
    plot_method="pcolormesh",
    alpha=0.3,
)
DecisionBoundaryDisplay.from_estimator(
    **common_params,
    response_method="decision_function",
    plot_method="contour",
    levels=[-1, 0, 1],
    colors=["k", "k", "k"],
    linestyles=["--", "-", "--"],
)

ax.scatter(temp_train[:, 0], temp_train[:, 1], c=y_train)
plt.show()


x_min, x_max = temp_train_1[:, 0].min() - 1, temp_train_1[:, 0].max() + 1
y_min, y_max = temp_train_1[:, 1].min() - 1, temp_train_1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

plt.scatter(temp_train_1[:, 0], temp_train_1[:, 1], c=y_train, cmap=plt.cm.coolwarm)
plt.xlabel('feature_x')
plt.ylabel('feature_y')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
print('\n' + "FEATURE EXECUTION TIME: " + str(time.time() - start) + " sec")
plt.show()
