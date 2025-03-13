import numpy as np
import time
import os
import dyconnmap
import bct
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy import stats

# --- Part 1 ---

def part_1():
    start = time.time()

    def setup_dependencies():  # Change as desired
        input_dir = '/home/sahib/Documents/OPM_MEG/data/run_1'  # The directory containing files of interest
        output_dir = '/home/sahib/Documents/OPM_MEG/data/Windows_r1_4s'  # The output directory
        target_fb = [[4, 7], [8, 12], [13, 30], [31, 40]]  # The target frequency band
        # TODO: set this up so variable can be a this can be list of bands and each generates its own adjacency matrix
        fs = 1200  # The sampling frequency
        new_shape = (78, 303, 2000)    # TO:DO 
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
        print('Data loaded' + '\n')
        return np.stack(data, axis=0)

    def wpli_conn(array, target_fb, fs):  # (participants, roi's, time_points)
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

    print('\n' + "EXECUTION TIME: " + str(time.time() - start))


# ---- Part 2 ----


def part_2():
    start = time.time()

    # Data pathing and loading
    r1 = np.load(r'/home/sahib/Documents/OPM_MEG/data/Windows_1/Windowed_run_1.npy')
    r2 = np.load(r'/home/sahib/Documents/OPM_MEG/data/Windows_2/Windowed_run_2.npy')

    output_dir = '/home/sahib/Documents/OPM_MEG/data/features_4s_windows/'

    # Graph calculator function
    def graph_metrics(adj_matrix):  # TODO: Add in stats, maybe nbs? Could use fdc as well?
        # Strength calculator
        strength = []
        strength.append(bct.strengths_und(np.nan_to_num(adj_matrix)))
        strength = np.array(strength)

        # Zeroing negative phasing
        strength[strength < 0] = 0

        # Betweenness centrality calculator
        betweenness = []
        betweenness.append(bct.betweenness_wei(np.nan_to_num(adj_matrix)))
        betweenness = np.array(betweenness)

        # Eigenvector centrality calculator
        eigenvector = []
        eigenvector.append(bct.eigenvector_centrality_und(np.nan_to_num(adj_matrix)))
        eigenvector = np.array(eigenvector)

        # Clustering calculator
        clustering = []
        clustering.append(bct.clustering_coef_wu(np.nan_to_num(adj_matrix)))
        clustering = np.array(clustering)

        return strength, betweenness, eigenvector, clustering

    # feature extraction function
    def apply_graph_metrics(r1, r2):
        r1_array = []
        for sample_r1 in r1:
            features = []
            r1_S, r1_B, r1_E, r1_C = graph_metrics(sample_r1)
            features.append(r1_S)
            features.append(r1_B)
            features.append(r1_E)
            features.append(r1_C)
            sample_features = np.array(features)
            r1_array.append(sample_features)

        r2_array = []
        for sample_r2 in r2:
            features = []
            r2_S, r2_B, r2_E, r2_C = graph_metrics(sample_r2)
            features.append(r2_S)
            features.append(r2_B)
            features.append(r2_E)
            features.append(r2_C)
            sample_features = np.array(features)
            r2_array.append(sample_features)
        print('Participant feature extraction complete')
        return np.array(r1_array), np.array(r2_array)

    # Get band and metric data
    def data_extraction(r1, r2, band, participant):
        r1 = r1[band, :, :, :, :]
        r2 = r2[band, :, :, :, :]
        r1 = r1[participant, :, :, :]
        r2 = r2[participant, :, :, :]
        return r1, r2

    def feature_extraction(band, r1, r2):
        band = band
        bands = ['Theta', 'Alpha', 'Beta', 'Gamma']
        all_participants_r1 = []
        all_participants_r2 = []
        for participant in range(10):
            input_1, input_2 = data_extraction(r1, r2, band=band, participant=participant)
            all_participants_r1.append(input_1)
            all_participants_r2.append(input_2)
            del input_1
            del input_2

        r1_array = np.array(all_participants_r1)
        r2_array = np.array(all_participants_r2)

        participant_features_r1 = []
        participant_features_r2 = []
        for i in range(10):
            pf1, pf2 = apply_graph_metrics(r1_array[i, :, :, :], r2_array[i, :, :, :])
            participant_features_r1.append(pf1)
            participant_features_r2.append(pf2)

        r1_feature_array = np.array(participant_features_r1)
        r2_feature_array = np.array(participant_features_r2)

        np.save(output_dir + bands[band] + '_r1_feature_array', r1_feature_array)
        np.save(output_dir + bands[band] + '_r2_feature_array', r2_feature_array)

    # Final feature extraction
    for i in range(4):
        feature_extraction(i, r1, r2)

    print('\n' + "FINAL EXECUTION TIME: " + str(time.time() - start) + " sec")

# ---- Part 3 ----


def part_3():
    start = time.time()

    permutations = 10  # This is the only think that really needs to be changed. default is (10)

    model = make_pipeline(SVC(C=0.1, kernel='rbf'))
    hold = []

    def fun(r1, r2, model, permutations):
        # flatten out participants
        r1 = r1.reshape((2727 + 303), 4, 1, 78)
        r2 = r2.reshape((2727 + 303), 4, 1, 78)
        hold = []
        for run in range(permutations):
            r1 = stats.zscore(r1, axis=-1)
            r2 = stats.zscore(r2, axis=-1)
            # x = np.concatenate((r1, r2), axis=0)
            x = r1
            nineth = len(x) // 10
            y = [0] * nineth + [1] \
                * nineth + [2] \
                * nineth + [3] \
                * nineth + [4] \
                * nineth + [5] \
                * nineth + [6] \
                * nineth + [7] \
                * nineth + [8] * nineth + [9] * nineth

            # Split data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            results = []
            for i in range(78):
                model.fit(x_train[:, 1, :, i], y_train)
                results.append(model.score(x_test[:, 1, :, i], y_test))
            results = np.array(results)

            areas = []
            for indx, val in enumerate(results):
                if val >= 0.11:
                    areas.append(indx)
            if len(areas) == 0:
                areas.append(0)
            model.fit(x_train[:, 1, 0, np.array(areas)], y_train)
            score = model.score(x_test[:, 1, 0, np.array(areas)], y_test)
            print(str(score))
            hold.append(score)
        arr_1 = np.array(hold)

        print('Run complete')

        hold = []
        for run in range(permutations):
            r1 = stats.zscore(r1, axis=-1)
            r2 = stats.zscore(r2, axis=-1)
            # x = np.concatenate((r1, r2), axis=0)
            x = r2
            nineth = len(x) // 10
            y = [0] * nineth + [1] \
                * nineth + [2] \
                * nineth + [3] \
                * nineth + [4] \
                * nineth + [5] \
                * nineth + [6] \
                * nineth + [7] \
                * nineth + [8] * nineth + [9] * nineth

            # Split data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            results = []
            for i in range(78):
                model.fit(x_train[:, 1, :, i], y_train)
                results.append(model.score(x_test[:, 1, :, i], y_test))
            results = np.array(results)

            areas = []
            for indx, val in enumerate(results):
                if val >= 0.11:
                    areas.append(indx)
            if len(areas) == 0:
                areas.append(0)
            model.fit(x_train[:, 1, 0, np.array(areas)], y_train)
            score = model.score(x_test[:, 1, 0, np.array(areas)], y_test)
            print(str(score))
            hold.append(score)
        arr_2 = np.array(hold)

        # print('\n' + 'Mean result: ' + str(arr_2.mean()))
        print('Run complete')
        print('\n' + ' -- Band complete -- ' + '\n')
        return arr_1, arr_2

    # Call

    # Theta
    r1 = np.load(r'/home/sahib/Documents/OPM_MEG/data/features_4s_windows/Theta_r1_feature_array.npy')
    r2 = np.load(r'/home/sahib/Documents/OPM_MEG/data/features_4s_windows/Theta_r2_feature_array.npy')
    t_arr_1, t_arr_2 = fun(r1=r1, r2=r2, model=model, permutations=permutations)

    # Alpha
    r1 = np.load(r'/home/sahib/Documents/OPM_MEG/data/features_4s_windows/Alpha_r1_feature_array.npy')
    r2 = np.load(r'/home/sahib/Documents/OPM_MEG/data/features_4s_windows/Alpha_r2_feature_array.npy')
    a_arr_1, a_arr_2 = fun(r1=r1, r2=r2, model=model, permutations=permutations)

    # Beta
    r1 = np.load(r'/home/sahib/Documents/OPM_MEG/data/features_4s_windows/Beta_r1_feature_array.npy')
    r2 = np.load(r'/home/sahib/Documents/OPM_MEG/data/features_4s_windows/Beta_r2_feature_array.npy')
    b_arr_1, b_arr_2 = fun(r1=r1, r2=r2, model=model, permutations=permutations)

    # Gamma
    r1 = np.load(r'/home/sahib/Documents/OPM_MEG/data/features_4s_windows/Gamma_r1_feature_array.npy')
    r2 = np.load(r'/home/sahib/Documents/OPM_MEG/data/features_4s_windows/Gamma_r2_feature_array.npy')
    g_arr_1, g_arr_2 = fun(r1=r1, r2=r2, model=model, permutations=permutations)

    ### Plotting ###
    import matplotlib
    import seaborn as sns

    matplotlib.use('Qt5Agg')

    runs = [1] * permutations + [2] * permutations

    bands = (['Theta'] * (permutations * 2) 
             + ['Alpha'] * (permutations * 2) 
             + ['Beta'] * (permutations * 2) 
             + ['Gamma'] * (permutations * 2))
    
    accuracy = (list(t_arr_1) + list(t_arr_2)) + (list(a_arr_1) + list(a_arr_2)) + (list(b_arr_1) + list(b_arr_2)) + (
            list(g_arr_1) + list(g_arr_2))

    df = pd.DataFrame({'accuracy': accuracy, 'bands': bands, 'runs': (runs + runs + runs + runs)})
    df.to_csv('result_temp')

    print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")

    fig_box = sns.boxplot(data=df, x='bands', y='accuracy', hue="runs", width=.5, palette="light:#5A9")
    fig_viol = sns.violinplot(data=df, x='bands', y='accuracy', hue="runs", palette="light:#5A9", split=True)
    fig_box.set_title('Fingerprinting accuracy boxplot')
    fig_viol.set_title('Fingerprinting accuracy violin plot')
