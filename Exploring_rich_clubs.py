import numpy as np
import networkx as nx
import bct
import matplotlib
import seaborn as sns
import pandas as pd
import time

# Setting some backend perameters
matplotlib.use('TkAgg')

r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Theta\run_1\Master_adj_matrix__Theta_run-001.npy')
r1 = np.mean(r1, axis=0)
remove_chans = [1, 2, 24, 28, 32, 33, 34, 45, 40, 41, 63, 67, 71, 72, 73, 74]
r1 = np.delete(r1, remove_chans, axis=-1)
r1 = np.delete(r1, remove_chans, axis=0)

G = nx.from_numpy_array(r1)
G.remove_edges_from(list(nx.selfloop_edges(G)))

array = nx.to_numpy_array(G)
org_array = array
array[array <0.25] = 0

# ran_net, _ = bct.null_model_und_sign(array)

# start = time.time()
# hold = []
# for i in range(10):
#     ran_net, _ = bct.null_model_und_sign(array, 10)
#     hold.append(ran_net)
# hold = np.array(hold)
# print('Time: ' + str((time.time() - start)) + ' secs')

ran_net, _ = bct.null_model_und_sign(array, 100)
ran_rich = bct.rich_club_wu(ran_net, klevel=100)
result = bct.rich_club_wu(array, klevel=100)
# ran_rich = np.nan_to_num(ran_rich, nan=1)
# result = np.nan_to_num(result, nan=1)
norm = result/ran_rich

sns.lineplot(ran_rich)
sns.lineplot(result)

# Plotting
df = pd.DataFrame({'Result' : result, 'Random' : ran_rich})
fig = sns.lineplot(df)
fig.set_title('Raw Rich Club Curve')

df_norm = pd.DataFrame({'Normalized RC': norm})
fig_norm = sns.lineplot(df_norm)
fig_norm.set_title('Normalized Rich Club Curve')
fig_norm.axhline(y=1, color='black', ls=':')


# Exploring core
core, core_size = bct.score_wu(org_array, s=3.5)  # range seems to run from 0 to about 4
sns.heatmap(core)

# TODO: Participation in the rich club 

# Extra

temp = []
for row in array:
    for i in row:
        if i > 0:
            temp.append(1)
val = (len(temp)/(62*62))*100
print(val, '% of edges retained')
