import numpy as np
import os
import xml.etree.ElementTree as ET
import bct
import netplotbrain
import pandas as pd

def load_npy_array(root_dir, filename):
    return np.load(os.path.join(root_dir, filename))


def get_file_names(master_dir, band, task):
    temp = []
    for filename in os.listdir(master_dir):
        if task in filename:
            if band in filename:
                temp.append(filename)
    return temp

def parse_xml(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    coordinates = []

    # Iterate over 'person' elements
    data_element = root.find('data')
    if data_element is not None:
        for label in data_element.findall('label'):
            # Extract attributes
            x = label.get('x')
            y = label.get('y')
            z = label.get('z')
            coordinates.append((float(x), float(y), float(z)))
    return coordinates

def load_data(master_dir, target_task, target_band):
    target_files = get_file_names(master_dir, task=target_task, band=target_band)
    array = [load_npy_array(root_dir=master_dir, filename=i) for i in target_files]
    return np.array(array)


# __main__

master_dir = r'C:\Users\em17531\Desktop\Google_data\Output_dir'

band = ['theta',
        'alpha',
        'beta',
        'gamma']

tasks = ['participant_conversation',
         'interviewer_repetition',
         'participant_repetition',
         'da',
         'ba',
         'interviewer_conversation']

convo_data_alpha = load_data(master_dir=master_dir, target_task=tasks[0], target_band=band[1])
rep_data_alpha = load_data(master_dir=master_dir, target_task=tasks[2], target_band=band[1])

# Stats

# _, adj, _ = bct.nbs_bct(convo_data_alpha.T, rep_data_alpha.T, 3.2, k=500, seed=2022, paired=True)
adj = np.load(r'F:\Geet\nbs_stats\nbs_participant_alpha.npy')

    ### Plotting ###

# Getting ROI loc's
file_path = r'C:\Users\em17531\Desktop\Atlas\HCP-Multi-Modal-Parcellation-1.0.xml'  # Change this as needed
coordinates = parse_xml(file_path)
del coordinates[0]
xyz_loc_array = np.stack(coordinates)

array = xyz_loc_array
del xyz_loc_array


nodes = pd.DataFrame(data={'x': (array[:,0] - 128),
                           'y': (array[:,1] - 150),
                           'z': (array[:,2] - 100),
                           })

edge_data = np.mean(convo_data_alpha, axis= 0) - np.mean(rep_data_alpha, axis=0)

import matplotlib
# Set back ends of viz
matplotlib.use('Qt5Agg')

netplotbrain.plot(template='MNI152NLin2009cAsym',
                  nodes=nodes,
                  node_type='circles',
                  edges=edge_data,
                  template_style='glass',
                  highlight_edges=adj,
                  view=['LSR'],
                  highlight_level=1)
