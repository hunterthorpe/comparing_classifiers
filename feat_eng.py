# ******************* Hunter Thorpe 1079893 ELODP Project 2 Task 2 b) **********
import re
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
import pandas as pd
import statistics as stats 

# Magic numbers 
LIFE = 3
SCORE_ = 0
FEAT_ = 1
LIFE_CODE = 1
WORLD_CODE = 2
R_STATE = 100
T_SIZE = 1/3
PH = -1 # placeholder values

# initiliasing data structures 
life_df = pd.read_csv("life.csv")
world_df = pd.read_csv("world.csv")

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
    feat_dict[key] = [median]

# imputing training set values with median
for coord in missing_vals:
    base_train[coord[0]][coord[1]] = feat_dict[coord[1]][0]

# imputating test set with training median
for num in range(len(base_test)):
    for column in range(len(base_test[num])):
        if bool(re.search(r'\d', base_test[num][column])) == False or \
          base_test[num][column] == '0' or len(base_test[num][column]) < 1:
            base_test[num][column] = feat_dict[column][0]
        else:
            base_test[num][column] = float(base_test[num][column])
            
# creating new feats using interaction pairs
feature_tracker = []
for i in range(len(base_train[0])):
    feature_tracker.append('f' + str(i))
def new_inter_feats(original):
    new_feats= []
    for country in range(len(original)):
        original_len = len(original[country])
        new_feats.append(original[country].copy())
        for num1 in range(original_len):
            feat1 = original[country][num1]
            for num2 in range(num1+1, original_len):
                feat2 = original[country][num2]
                new_feats[country].append(feat1 * feat2)
                if len(feature_tracker) < 210:
                    feature_tracker.append(feature_tracker[num1] + ' * ' \
                           + feature_tracker[num2])
    return new_feats
new_feats_train = new_inter_feats(base_train)
new_feats_test = new_inter_feats(base_test)

# scaling base sets prior to clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(base_train)
base_train = list(scaler.transform(base_train))
base_test = list(scaler.transform(base_test))

# creating graph used for selecting number of clusters
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# func calculates within-cluster-sum of squared errors for vals of k
def calc_wss(points, range_):
    sums = []
    for k in range(1, range_+1):
        kmeans = KMeans(n_clusters = k).fit(points)
        centroids = kmeans.cluster_centers_
        predict_clusters = kmeans.predict(points)
        cur_sum = 0
    #getting euc distance from point to center and adding to sum
        for point in range(len(points)):
            cur_center = centroids[predict_clusters[point]]
            cur_sum += (points[point][0] - cur_center[0]) ** 2 + \
              (points[point][1] - cur_center[1]) ** 2
        sums.append(cur_sum)
    return sums
# creating graph to look for elbow 
wss = calc_wss(base_train, 20)
k = []
for i in range(20):
    k.append(i)
plt.plot(k, wss)
plt.ylabel('wss')
plt.xlabel('k')
plt.savefig('task2graph1.png')

# using cluster labels to create another feature
# num of clusters set to 5 from elbow in figure 
Kmean = KMeans(n_clusters = 5)
Kmean.fit(base_train)
labels = Kmean.labels_
for num in range(len(labels)):
    new_feats_train[num].append(labels[num])
# creating a feature for test set 
  # based of closest clusters
close_cluster = Kmean.predict(base_test)
for num in range(len(close_cluster)):
    new_feats_test[num].append(close_cluster[num])
feature_tracker.append('fclusterlabel')
    
# scaling feature eng set
scaler.fit(new_feats_train)
new_feats_train = list(scaler.transform(new_feats_train))
new_feats_test = list(scaler.transform(new_feats_test))

# finding best 4 features by scoring each feature individually
x_feats, y_feats, x_outcomes, y_outcomes = \
  skm.train_test_split(new_feats_train, outcomes_train, \
    test_size=(T_SIZE), random_state = R_STATE)
top_4_feats = [[PH, PH], [PH, PH], [PH, PH], [PH, PH]]
for feature in range(len(new_feats_train[0])):
    temp_set_train = []
    temp_set_test = []
    for record in x_feats:
        temp_set_train.append([float(record[feature])])
    for record in y_feats:
        temp_set_test.append([float(record[feature])])
    knn.fit(temp_set_train, x_outcomes)
    score = knn.score(temp_set_test, y_outcomes)
    for elem in range(len(top_4_feats)):
        if score > top_4_feats[elem][SCORE_]:
            top_4_feats[elem][FEAT_] = feature
            top_4_feats[elem][SCORE_] = score
            top_4_feats = sorted(top_4_feats)
            break

# printing cluster feature acccuracy 
print('Clustering Labels Feature Accuracy :' + \
      "%.3f" % score)
print('')
# printing top 4 features accuracy
print('Top 4 individual performing features:')
print('Feature: Accuracy')
for feature in top_4_feats:
    print(feature_tracker[feature[FEAT_]] + 
          ': ' + "%.3f" % feature[SCORE_])
print('')

# implementing PCA on new_feats_set
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
pca.fit(new_feats_train)
# transforming training and test based of training fit
pca_train = list(pca.transform(new_feats_train))
pca_test = list(pca.transform(new_feats_test))
# scaling 
scaler.fit(pca_train)
pca_train = list(scaler.transform(pca_train))
pca_test = list(scaler.transform(pca_test))
# training and testing pca classifer
knn.fit(pca_train, outcomes_train)
print('Accuracy of PCA on 20 original features + 191 created features :' + \
      "%.3f" % knn.score(pca_test, outcomes_test))
print('')
    
# removing all other features 
sets = [new_feats_train, new_feats_test]
for set_ in range(len(sets)):
    for record in range(len(sets[set_])):
        temp_set_test = []
        for feature in top_4_feats:
            temp_set_test.append(sets[set_][record][feature[FEAT_]])
        sets[set_][record] = temp_set_test

# training and testing top 4 feats classifer 
knn.fit(new_feats_train, outcomes_train)
feat_acc = knn.score(new_feats_test, outcomes_test)

# implementing PCA on original 20 
pca.fit(base_train)
# transforming training and test based of training fit
pca_train = list(pca.transform(base_train))
pca_test = list(pca.transform(base_test))
# scaling 
scaler.fit(pca_train)
pca_train = list(scaler.transform(pca_train))
pca_test = list(scaler.transform(pca_test))
# training and testing pca classifer 
knn.fit(pca_train, outcomes_train)
pca_acc = knn.score(pca_test, outcomes_test)

# creating a train and test set from first 4 feats
first_four_train = []
first_four_test = []
for record in base_train:
    first_four_train.append(record[0:4])
for record in base_test:
    first_four_test.append(record[0:4])
# training and testing classifer 
knn.fit(first_four_train, outcomes_train)
feat4_acc = knn.score(first_four_test, outcomes_test)

print('Accuracy of feature engineering: ' + "%.3f" % feat_acc)
print('Accuracy of PCA: ' + "%.3f" % pca_acc)
print('Accuracy of first four features: ' + "%.3f" % feat4_acc)

# ********************** End of Script **************************