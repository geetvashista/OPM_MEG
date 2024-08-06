"""
A script for the plotting of AAP atlas graph. These regions ARE NOT set to MNI space as yet.
78 ROI's where used to match with:
https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00020/117508/Test-retest-reliability-of-the-human-connectome-An
A number of changes are possible from here such as, communities and highlight_edges.
"""
import netplotbrain
import pandas as pd
import numpy as np
import scipy

# Set depanneurs
mat_voxlox = scipy.io.loadmat(r'')
adj = np.load(r'')
hig = np.load(r'')

# pep files
loc = mat_voxlox['voxlox']
del mat_voxlox

adj = np.mean(adj, axis=0)

# Plotting

nodes = pd.DataFrame(data={'x': loc[0, :],
                           'y': loc[1, :],
                           'z': loc[2, :]
                           })

netplotbrain.plot(nodes=nodes,
                  template_style='filled',
                  edges=adj,
                  highlight_edges=hig,
                  view=['LSR'],
                  title='All edges, sig highlighted',
                  arrowaxis=None,
                  highlight_level=0.92,
                  node_scale=100)
