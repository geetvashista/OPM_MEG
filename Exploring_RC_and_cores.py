import numpy as np
import pandas as pd
import bct
import networkx as nx
import matplotlib
import seaborn as sns

matplotlib.use('Qt5Agg')

# result_path = r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\RC_results\Theta\Participant\run_1\results\result_0.npy'
# ran_net_path = r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\RC_results\Theta\Participant\run_1\ran_nets\ran_net_0.npy'
#
# result = np.load(result_path)
# ran_net = np.load(ran_net_path)
# ran_net = np.mean(ran_net, axis=0)
#
# # Plotting
# df = pd.DataFrame({'Result' : result, 'Random' : ran_net})
# fig = sns.lineplot(df)
# fig.set_title('Raw Rich Club Curve')

# df_norm = pd.DataFrame({'Normalized RC': norm})
# fig_norm = sns.lineplot(df_norm)
# fig_norm.set_title('Normalized Rich Club Curve')
# fig_norm.axhline(y=1, color='black', ls=':')


data = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\Theta\run_1\Master_adj_matrix__Theta_run-001.npy')
remove_chans = [1, 2, 24, 28, 32, 33, 34, 45, 40, 41, 63, 67, 71, 72, 73, 74]
data = np.delete(data, remove_chans, axis=1)
data = np.delete(data, remove_chans, axis=2)
data = data[7, :, :]

G = nx.from_numpy_array(data)
G.remove_edges_from(list(nx.selfloop_edges(G)))
data = nx.to_numpy_array(G)

core, _ = bct.score_wu(data, 36)

core_nodes = []
for index, val in enumerate(core):
    [core_nodes.append(index) for element in val if abs(element) > 0]
core_nodes = set(core_nodes)


# TO be run with variables from fingerprinting pipline

common_nodes = set.intersection(*map(set, t_areas_1))
coreANDcom = set.intersection(common_nodes, core_nodes)

p1 = coreANDcom
p2 = coreANDcom
p5 = coreANDcom
p6 = coreANDcom
p7 = coreANDcom
