import numpy as np
import os
import scipy

# base path
directory = r'C:\Users\em17531\Desktop\New_project\data\dog_day_afternoon_OPM\derivatives\VEs'  # Change as needed.

def convert_files(directory):
    for folder in os.listdir(directory):
        for file in os.listdir(os.path.join(directory, folder)):
            mat = scipy.io.loadmat(os.path.join(directory, folder, file))
            np.save(os.path.join(directory, folder, file) + '_array', mat['VE'])

def main():
    convert_files(directory)

if __name__ == "__main__":
    main()