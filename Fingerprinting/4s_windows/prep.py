import numpy as np
import time
import os
import dyconnmap
start = time.time()

def setup_dependencies():    # Change as desired
    input_dir = '/home/sahib/Documents/OPM_MEG/data/run_1'     # The directory containing files of interest
    output_dir = '/home/sahib/Documents/OPM_MEG/data/Windows_r1_4s'    # The output directory
    target_fb = [[4, 7], [8,12], [13, 30], [31, 40]]  # The target frequency band
    # TODO: set this up so variable can be a this can be list of bands and each generates its own adjacency matrix
    fs = 1200    # The sampling frequency
    new_shape = (78, 303, 2000)
    saving = True
    os.makedirs(output_dir, exist_ok=True)
    return input_dir, output_dir, fs, target_fb, new_shape, saving


def prep_data(input_dir, target_fb, fs, new_shape):
    print('Loading data' + '\n')
    data = []
    for file in os.listdir(input_dir):
        array = np.load(os.path.join(input_dir, file))
        array = array[0:606000, :]
        _, _, filtered = dyconnmap.analytic_signal(array.T, fb=target_fb, fs=fs)
        filtered = filtered.reshape(new_shape)
        data.append(np.moveaxis(filtered, 1, 0))
    print('Data loaded' + '\n' )
    return np.stack(data, axis=0)


def wpli_conn(array, target_fb, fs):   # (participants, roi's, time_points)
    print('calculating adj matrices' + '\n')
    master_matrix = []
    for participant in array:
        window_array = []
        for window in participant:
            window_array.append(dyconnmap.fc.wpli(window, fs=fs, fb=target_fb))
        master_matrix.append(window_array)
    return np.array(master_matrix)


def main():
    input_dir, output_dir, fs, target_fb, new_shape, saving = setup_dependencies()

    output_data = []
    for i in target_fb:
        # Load and prep data
        input_data = prep_data(input_dir, i, fs, new_shape)

        # Cal matrices
        output_array = wpli_conn(input_data, i, fs)
        output_data.append(output_array)
    print('Run set complete' + '\n')

    if saving:
        np.save(output_dir + '/Windowed_4s_r1', np.stack(output_data, axis=0))


    # RUN 2
    print('Beginning run 2' + '\n')
    input_dir = '/home/sahib/Documents/OPM_MEG/data/run_2'
    output_dir = '/home/sahib/Documents/OPM_MEG/data/Windows_r1_4s'


    output_data = []
    for i in target_fb:
        # Load and prep data
        input_data = prep_data(input_dir, i, fs, new_shape)

        # Cal matrices
        output_array = wpli_conn(input_data, i, fs)
        output_data.append(output_array)

    if saving:
        np.save(output_dir + r'/Windowed_4s_r2', np.stack(output_data, axis=0))


if __name__ == "__main__":
    main()

print('\n' + "EXECUTION TIME: " + str(time.time()-start))
