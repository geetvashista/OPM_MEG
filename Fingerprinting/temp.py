import numpy as np
import time
import os
import dyconnmap
start = time.time()

def setup_dependencies():    # Change as desired
    input_dir = '/home/sahib/Documents/OPM_MEG/data/run_1'     # The directory containing files of interest
    output_dir = '/home/sahib/Documents/OPM_MEG/data/temp_1'    # The output directory
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
        filtered = array.reshape(new_shape)
        data.append(np.moveaxis(filtered, 1, 0))
    print('Data loaded' + '\n' )
    return np.stack(data, axis=0)


def main():
    input_dir, output_dir, fs, target_fb, new_shape, saving = setup_dependencies()

    output_data = []
    for i in target_fb:
        # Load and prep data
        input_data = prep_data(input_dir, i, fs, new_shape)
        output_data.append(input_data)

    print('Run set complete' + '\n')

    if saving:
        np.save(output_dir + '/Windowed_2s_r1', np.stack(output_data, axis=0))


    # RUN 2
    print('Beginning run 2' + '\n')
    input_dir = '/home/sahib/Documents/OPM_MEG/data/run_2'
    output_dir = '/home/sahib/Documents/OPM_MEG/data/temp_2'


    output_data = []
    for i in target_fb:
        # Load and prep data
        input_data = prep_data(input_dir, i, fs, new_shape)
        output_data.append(input_data)

    if saving:
        np.save(output_dir + r'/Windowed_2s_r2', np.stack(output_data, axis=0))


if __name__ == "__main__":
    main()

print('\n' + "EXECUTION TIME: " + str(time.time()-start))
