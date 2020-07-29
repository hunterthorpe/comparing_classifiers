# ******************* Hunter Thorpe 1079893 EODP Project 2 Task 2 a) **********
import re
import pandas as pd
import statistics as stats 
import numpy as np
import csv

# Magic numbers 
LIFE = 3
LIFE_CODE = 1
WORLD_CODE = 2
R_STATE = 100
T_SIZE = 1/3
MEAN_ = 1
STDEV = 2

# initiliasing data structures 
life_df = pd.read_csv("life.csv")
world_df = pd.read_csv("world.csv")
output_list = [['feature', 'median', 'mean', 'variance']]

# data linkage on countries with life expectancy data
base_set = []
base_set_outcomes = []
for index_w, row_w in world_df.iterrows():
    for index_l, row_l in life_df.iterrows():
        if row_w[WORLD_CODE] == row_l[LIFE_CODE]:
            column_count = 0
            temp_list = []            
            while column_count < len(list(world_df.columns.values)):
                temp_list.append(row_w[column_count])
                column_count += 1
            base_set_outcomes.append(row_l[LIFE])
            base_set.append(temp_list[LIFE:])
            break

# splitting set into test set and training set
import sklearn.model_selection as skm
base_train, base_test, outcomes_train, outcomes_test = \
  skm.train_test_split(base_set, base_set_outcomes, \
    test_size=(T_SIZE), random_state = R_STATE)

# tracking incomplete cells and adding data to feat_dict to calculate median
feat_dict = {}
for column in range(0, len(base_set[0])):
    feat_dict[column] = []  
missing_vals = []
for num in range(0, len(base_train)):
    for column in range(len(base_train[num])):
        if bool(re.search(r'\d', base_train[num][column])) == False or \
          base_train[num][column] == '0' or len(base_train[num][column]) < 1:
            missing_vals.append([num, column])
        else: 
            base_train[num][column] = float(base_train[num][column])
            feat_dict[column].append(base_train[num][column])

# calculating key stats
for key in feat_dict.keys():
    median = stats.median(feat_dict[key])
    mean = stats.mean(feat_dict[key])
    std = np.std(feat_dict[key])
    feat_dict[key] = [median, mean, std]

# imputing training set values with median
for coord in missing_vals:
    base_train[coord[0]][coord[1]] = feat_dict[coord[1]][0]

# imputating and scaling test set 
for num in range(len(base_test)):
    for column in range(len(base_test[num])):
        if bool(re.search(r'\d', base_test[num][column])) == False or \
          base_test[num][column] == '0' or len(base_test[num][column]) < 1:
            base_test[num][column] = feat_dict[column][0]
        base_test[num][column] = (float(base_test[num][column]) - \
                          feat_dict[column][MEAN_]) / feat_dict[column][STDEV]
    
# scaling training set 
for num in range(len(base_train)):
    for column in range(len(base_train[num])):
        base_train[num][column] = (float(base_train[num][column]) - \
                          feat_dict[column][MEAN_]) / feat_dict[column][STDEV]

# training k5 and k10 and calculating their accuracy
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(base_train, outcomes_train)
k5_acc = knn.score(base_test, outcomes_test)

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(base_train, outcomes_train)
k10_acc = knn.score(base_test, outcomes_test)

# training decistiontreeclassifier and calcing accuracy
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 4)
clf = clf.fit(base_train, outcomes_train)
dt_acc = clf.score(base_test, outcomes_test)

# adding median mean and var to task2a.csv
headers = list(world_df)
for feature in range(len(base_train[0])):
    output_list.append([headers[feature + 3]] + \
      [round(feat_dict[feature][0], 3), \
      round(feat_dict[feature][1], 3), \
      round(feat_dict[feature][2]**2, 3)]) 

with open("task2a.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(output_list)
    
print('Accuracy of decision tree: ' + "%.3f" % dt_acc)
print('Accuracy of k-nn (k=5): ' + "%.3f" % k5_acc)
print('Accuracy of k-nn (k=10): ' + "%.3f" % k10_acc)

# ********************** End of Script **************************

