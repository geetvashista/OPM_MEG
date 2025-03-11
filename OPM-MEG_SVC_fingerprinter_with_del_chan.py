import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
import pandas as pd
from scipy import stats
from itertools import zip_longest
start = time.time()

permutations = 1000    # This is the only think that really needs to be changed. default is (10)
remove_chans = [1, 2, 24, 28, 32, 33, 34, 45, 40, 41, 63, 67, 71, 72, 73, 74]

model = make_pipeline(SVC(C=0.1, kernel='rbf'))
hold = []

def fun(r1, r2, model, permutations):
    # flatten out participants
    r1 = r1.reshape((2727 + 303), 4, 1, 78)
    r2 = r2.reshape((2727 + 303), 4, 1, 78)
    r1 = np.delete(r1, remove_chans, axis=-1)
    r2 = np.delete(r2, remove_chans, axis=-1)

    hold = []
    participant_areas_1 = []
    for run in range(permutations):
        # r1 = stats.zscore(r1, axis=-1)
        # r2 = stats.zscore(r2, axis=-1)
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
        for i in range(62):
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
        participant_areas_1.append(areas)
        print(areas)
        score = model.score(x_test[:, 1, 0, np.array(areas)], y_test)
        print(str(score))
        hold.append(score)
    areas_1 = [list(tpl) for tpl in zip(*zip_longest(*participant_areas_1))]
    arr_1 = np.array(hold)

    print('Run complete')

    hold = []
    participant_areas_2 = []
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
        for i in range(62):
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
        participant_areas_2.append(areas)
        print(areas)
        score = model.score(x_test[:, 1, 0, np.array(areas)], y_test)
        print(str(score))
        hold.append(score)
    areas_2 = [list(tpl) for tpl in zip(*zip_longest(*participant_areas_2))]
    arr_2 = np.array(hold)

    # print('\n' + 'Mean result: ' + str(arr_2.mean()))
    print('Run complete')
    print('\n' + ' -- Band complete -- ' + '\n')
    return arr_1, areas_1, arr_2, areas_2

# Call

    # Theta
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Theta_r1_feature_array.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Theta_r2_feature_array.npy')
t_arr_1, t_areas_1, t_arr_2, t_areas_2 = fun(r1=r1, r2=r2, model=model, permutations=permutations)

    # Alpha
r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Alpha_r1_feature_array.npy')
r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Alpha_r2_feature_array.npy')
a_arr_1, a_areas_1, a_arr_2, a_areas_2 = fun(r1=r1, r2=r2, model=model, permutations=permutations)

#     # Beta
# r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Beta_r1_feature_array.npy')
# r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Beta_r2_feature_array.npy')
# b_arr_1, b_arr_2 = fun(r1=r1, r2=r2, model=model, permutations=permutations)
#
#     # Gamma
# r1 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Gamma_r1_feature_array.npy')
# r2 = np.load(r'C:\Users\em17531\Desktop\OPM_MEG\data\Windowed_features\Gamma_r2_feature_array.npy')
# g_arr_1, g_arr_2 = fun(r1=r1, r2=r2, model=model, permutations=permutations)

    ### Plotting ###
import matplotlib
import seaborn as sns
matplotlib.use('Qt5Agg')


runs = [1] * permutations + [2] * permutations


# bands = ['Theta'] * (permutations * 2) + ['Alpha'] * (permutations * 2) + ['Beta'] * (permutations * 2) + ['Gamma'] * (permutations * 2)
bands = ['Theta'] * (permutations * 2) + ['Alpha'] * (permutations * 2)
# accuracy = (list(t_arr_1) + list(t_arr_2)) + (list(a_arr_1) + list(a_arr_2)) + (list(b_arr_1) + list(b_arr_2)) + (list(g_arr_1) + list(g_arr_2))
accuracy = (list(t_arr_1) + list(t_arr_2)) + (list(a_arr_1) + list(a_arr_2))
df = pd.DataFrame({'accuracy': accuracy, 'bands': bands, 'runs': (runs + runs), 'Areas': (t_areas_1 + t_areas_2 + a_areas_1 + a_areas_2)})

df.to_csv(r'C:\Users\em17531\Documents\fingerprinting_PSD\results_with_1000_permutations')
print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")


def find_common_integers(list_of_lists):
    # Start with the set of the first list
    common_integers = set(list_of_lists[0])

    # Intersect with sets of the remaining lists
    for sublist in list_of_lists[1:]:
        common_integers.intersection_update(sublist)

    # Convert the result to a list and return
    return list(common_integers)


ROI_names = ['Frontal_Sup_Orb_L',
'Frontal_Med_Orb_L',
'Frontal_Mid_Orb_L',
'Frontal_Inf_Orb_L',
'Frontal_Sup_L' ,
'Frontal_Mid_L' ,
'Frontal_Inf_Oper_L' ,
'Frontal_Inf_Tri_L' ,
'Frontal_Sup_Medial_L',
'Supp_Motor_Area_L' ,
'Paracentral_Lobule_L',
'Precentral_L' ,
'Rolandic_Oper_L',
'Postcentral_L' ,
'Parietal_Sup_L' ,
'Parietal_Inf_L' ,
'SupraMarginal_L',
'Angular_L' ,
'Precuneus_L' ,
'Occipital_Sup_L' ,
'Occipital_Mid_L',
'Calcarine_L',
'Cuneus_L' ,
'Lingual_L',
'Heschl_L',
'Temporal_Sup_L',
'Temporal_Mid_L',
'Cingulum_Ant_L',
'Cingulum_Mid_L',
'Cingulum_Post_L',
'Insula_L' ,
'Frontal_Sup_Orb_R',
'Frontal_Med_Orb_R',
'Frontal_Mid_Orb_R',
'Frontal_Inf_Orb_R',
'Frontal_Sup_R',
'Frontal_Mid_R',
'Frontal_Inf_Oper_R',
'Frontal_Inf_Tri_R',
'Frontal_Sup_Medial_R',
'Supp_Motor_Area_R',
'Paracentral_Lobule_R',
'Precentral_R',
'Rolandic_Oper_R',
'Postcentral_R',
'Parietal_Sup_R',
'Parietal_Inf_R',
'SupraMarginal_R',
'Angular_R',
'Precuneus_R',
'Occipital_Sup_R',
'Occipital_Mid_R',
'Calcarine_R',
'Cuneus_R',
'Lingual_R',
'Heschl_R',
'Temporal_Sup_R',
'Temporal_Mid_R',
'Cingulum_Ant_R',
'Cingulum_Mid_R',
'Cingulum_Post_R',
'Insula_R']


def names_of_common_ROI(areas):
    common = find_common_integers(areas)
    names_of_ROI = []
    for i in common:
        names_of_ROI.append(ROI_names[i])
    return names_of_ROI


# run_01 = accuracy[:]
# run_02 = accuracy[:]
#
# #
# # # df = pd.read_excel(r'C:\Users\em17531\Desktop\OPM_MEG\derivatives\df_per_25_accuracy_exc.xlsx')
# # fig = sns.boxplot(data=df, x='bands', y='accuracy', hue="runs", width=.5, palette="light:#5A9")
fig = sns.violinplot(data=df, x='bands', y='accuracy', hue="runs", palette="light:#5A9", split=True)
