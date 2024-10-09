import numpy as np
import matplotlib
import seaborn as sns
matplotlib.use('Qt5Agg')

# Plot matrix
data = np.load('') 

mask = np.triu(np.ones_like(data, dtype=bool))
fig = sns.heatmap(array, mask = mask)


