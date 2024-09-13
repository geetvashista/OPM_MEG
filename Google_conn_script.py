import mne
import dyconnmap
import os
import time
import numpy as np
start = time.time()

def load_data(full_file_path):
     return mne.read_source_estimate(
         full_file_path
     )


def remove_duplicates(strings):
    # Use a set to keep track of seen strings
    seen = set()
    # Use a list comprehension to filter out duplicates
    return [x for x in strings if not (x in seen or seen.add(x))]


def Parcellator(stc):
    labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", "both")
    # DON'T FORGET TO GET RID OF THE FIRST TWO LABELS
    # Use del labels[0]
    del labels[0]
    del labels[0]

    dec = {}
    for k in labels:
        index = labels.index(k)
        temp = str(labels[index])
        dec[temp] = stc.in_label(k).data  # change here from buds to sham as needed

    j = 0
    while True:
        temp = dec[str(labels[j])]
        dec[str(labels[j])] = np.mean(temp, axis=0)
        j += 1
        if j >= 360:
            break

    list = [i for i in dec.values()]
    list_2 = [np.expand_dims(i, axis=0) for i in list]
    temp = np.concatenate(list_2)
    return temp


def wpli_conn(array, target_fb, fs):   # Assumed shape (roi's, time_points)
    print('calculating connectivity')
    adj_matrix = dyconnmap.fc.wpli(array, fs=fs, fb=target_fb)
    adj_matrix =np.array(adj_matrix)
    np.save(output_dir + filename + task + 'adj_matrix', adj_matrix)
    print('calculating complete' + '\n')


def get_file(master_dir, band, task):
    temp = []
    for filename in os.listdir(master_dir):
        if task in filename:
            if 'averaged' in filename:
                if band in filename:
                    if 'time' in filename:
                        _
                    elif 'difference' in filename:
                        _
                    elif 'difference' in filename:
                        _
                    else:
                        temp.append(filename)
    temp = [i.strip('-rh.stc') for i in temp]
    temp = [i.strip('-lh.stc') for i in temp]
    temp = ['s'+i for i in temp]
    return remove_duplicates(temp)

# data = get_file(master_dir, 'alpha', 'articipant_conversation')

# __main__


master_dir = '/media/sahib/Mangor_2TB/Geet/LCMV_output_01082024'
output_dir = '/media/sahib/Mangor_2TB/Geet/Output_dir/'

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

for band in band:
    for task in tasks:
        print('\n')
        data_path = get_file(master_dir, band, task)
        for filename in data_path:
            stc = load_data(os.path.join(master_dir, filename + '-stc'))
            stc_array = Parcellator(stc)
            wpli_conn(stc_array, target_fb=[7, 12], fs=1000)
        print('Task complete')

print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")
