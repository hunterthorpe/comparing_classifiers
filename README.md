# comparing_classifiers
Comparing classification algorithms using life expectancy data. Classifying countries as having Low, Medium or High life expectancy using World Development Indicators and World Health Organization data.

Libraries Used: Pandas, Numpy, sklearn, Statistics 

Background 

Each year, the World Bank publishes the World Development Indicators which provide high quality and international comparable statistics about global development and the fight against poverty [1]. As data scientists, we wish to understand how the information can be used to predict average lifespan in different countries. To this end, we have provided the world.csv file, which contains some of the World Development Indicators for each country and the life.csv file containing information about the average lifespan for each country (based on data from the World Health Organization) [2]. Each data file also contains a country name, country code and year as identifiers for each record. These may be used to link the two datasets but should not be considered features.
Comparing Classification Algorithms
Compare the performance of the following 3 classification algorithms: k-NN (k=5 and k=10) and Decision tree (with a maximum depth of 4) on the provided data. You may use sklearn’s KNeighborsClassifier and DecisionTreeClassifier functions for this task. For the k-NN classifier, all parameters other than k should be kept at their defaults. For the De- cision tree classifier, all parameters other than the maximum depth should be kept at their defaults. Use each classification algorithm to predict the class feature life expectancy at birth(years) of the data (Low, Medium and High life expectancy) using the remaining fea- tures.
For each of the algorithms, fit a model with the following processing steps:
􏰀 Split the dataset into a training set comprising 2/3 of the data and a test set comprising
the remaining 1/3 using the train test split function with a random state of 100.
􏰀 Perform the same imputation and scaling to the training set:
– For each feature, perform median imputation to impute missing values. – Scale each feature by removing the mean and scaling to unit variance.
􏰀 Train the classifiers using the training set
􏰀 Test the classifiers by applying them to the test set.

Code produces a CSV file called task2a.csv describing the median used for imputation for each feature, as well as the mean and variance used for scaling, all rounded to three decimal places. The CSV file must have one row corresponding to each feature. The first three lines of the output should be as follows (where x is a number calculated by your program):
feature, median, mean, variance. Code prints the classification accuracy of each classifier to standard output in the format:

Accuracy of decision tree: %
Accuracy of k-nn (k=5): %
Accuracy of k-nn (k=10): %

Feature Engineering and Selection
Using k-NN with k=5 (from here on referred to as 5-NN). In order to achieve higher prediction accuracy for 5-NN, one can investigate the use of feature engineering and selection to predict the class feature of the data. Feature generation involves the creation of additional features. Two possible methods are:
􏰀 Interaction term pairs. Given a pair of features f1 and f2, create a new feature f12 = f1 × f2. All possible pairs can be considered.
􏰀 Clustering labels: apply k-means clustering to the data in world and then use the resulting cluster labels as the values for a new feature fclusterlabel. You will need to decide how many clusters to use. At test time, a label for a testing instance can be created by assigning it to its nearest cluster.
Given a set of N features (the original features plus generated features), feature selection involves selecting a smaller set of n features (n < N).
An alternative method of performing feature engineering & selection is to use Principal Com- ponent Analysis (PCA). The first n principal components can be used as features.
Your task in this question is to evaluate how the above methods for feature engineering and selection affect the prediction accuracy compared to using 5-NN on a subset of the original features in world. You should:
􏰀 Implement feature engineering using interaction term pairs and clustering labels. This should produce a dataset with 211 features (20 original features, 190 features generated by interaction term pairs and 1 feature generated by clustering). You should (in some principled manner) select 4 features from this dataset and perform 5-NN classification.
􏰀 Implement feature engineering and selection via PCA by taking the first four principal components. You should use only these four features to perform 5-NN classification.
􏰀 Take the first four features (columns D-G, if the dataset is opened in Excel) from the original dataset as a sample of the original 20 features. Perform 5-NN classification.
􏰀 The classification accuracy for the test set for of the three methods in the following format, as the last three lines printed to standard output (where the % symbol is replaced by the accuracy of 5-NN using each feature set, rounded to 3 decimal places):
     Accuracy of feature engineering: %
     Accuracy of PCA: %
     Accuracy of first four features: %
    
        Report 

Which algorithm (decision trees or k-nn) in Task-2A performed better on this dataset? For k-nn, which value of k performed better? Explain your experiments and the results.
The accuracies obtained in Task-2A are as follows (decision tree: 0.689- 0.702, k-nn (k=5): 0.820, and k-nn (k=10): 0.869). Because the random state of the decision tree classifier was not set as per the project specifications, the value took a range differing between each execution of the script. Overall, both of the k-nn classifiers outperformed the decision tree classifier, and between the two implementations of the k-nn algorithm, the one setting k equal to 5 had the highest accuracy. This is most likely due to the larger value for k causing samples of other classes to be included thus leading to misclassifications in the testing step.
The first action in the experiments undertaken in Task-2A was data linkage between the life.csv and world.csv files using exclusively country codes, as some of the records in two files shared the same country codes yet contained differing text listed under countries e.g. ‘Bahamas, The’ and ‘Bahamas’. Countries that lacked an entry in both files were excluded from the dataset used in the experiment. I contained the class feature (‘Life expectancy at birth (years)’) in a separate data structure to the remaining features, and then split both of these into training and test sets. My next step was to iterate through the training set, creating an entry into feat_dict for each feature and appending every non- missing value for this feature, with all non numbers and the number 0 being treated as a missing value. Using this I calculated the mean, median and standard deviation for each feature in the training set, which I later recorded in the file ‘task2a.csv’, except for the standard deviation which I used to calculate and record the variance. I then employed the median to impute the missing values, and the mean and standard deviation to scale via mean removal followed by a division by the standard deviation, thus scaling to unit variance. I performed these operations on both the training set and test set, using only the data from the training set. After this I trained the classifiers using the training set and then scored their accuracy on the test set, printing the results to standard output.
A description of the precise steps I took to perform the analysis in Task-2B. (Including a justification for the method used to select the number of clusters, and a justification for the method used in feature selection) 
Data Linkage, Splitting and Imputation
My analysis in Task-2B began the same way I conducted Task-2A, as I performed data linkage on the two files, split the resulting data sets (into base_train, base_test, outcomes_train and outcomes_test), calculated and imputed the median in the same fashion.  
Feature Creation
However prior to scaling I iterated through each of the sets separately, creating two new seats (new_feats_train and new_feats_test), each with 190 new features for each record through interaction term pairs, multiplying each of the original features with each other original feature. At the same time I created an auxiliary data structure that stores information pertaining to the origin of each feature, labelling each feature in the world.csv file f0 to f19, and the resulting features as a combination of these original 19 e.g. f1 * f2.
Cluster Label Generation
Next I fit StandardScaler() from the sklearn library with base_train, and used the scaler to transform base_train and base_test. The next operation in the script, plotting of within-cluster-sum of squared errors against values of k (where k = number of clusters), was used primarily in the development process to select the number clusters to be used in labelling, and has no runtime impact on the classification accuracies. It relies upon the use a function I
wrote calc_wss(points, range_) which takes in array of multi-dimensional points which are fitted to a Kmeans class for each value between 1 and range_ (which are used as accompanying k values), and calculates the sum of the euclidian distances between the centroids and the predicted clusters for each value of k. As this function is psuedo-random, I analysed many of the resulting plots (one of which is figure 1) when attempting to use the elbow method to find an optimal value for k. Having identified the elbow as closest to k = 5, I set the number of clusters to this value and fitted the Kmeans class from the sklearn library with the base_train set. I then used the .labels method to create values for a new feature in new_feats_train, generated from the resulting cluster labels. Following this, I employed the .predict() method using base_test as input in order to create labels for each record in new_feats_test by assigning them to the nearest cluster.
Feature Selection
I then scaled new_feats_train and new_feats_test, fitting new_feats_train to StandardScaler() and then transforming both sets. Following this, I performed a split on new_feats_train and outcomes_train to create the sets x_feats, y_feats, x_outcomes, y_outcomes. I then used
these sets to select 4 features from new_feats_train to perform 5-NN classification. To do
this I used the x and y sets and two auxiliary data structures to score the accuracy of each of the 211 features when fitted and tested individually via 5-NN classification, and stored the top 4 scores and the relevant position of these features within each record in a third data structure. Originally I had planned to score each feature using the test set, however this would have been a mistake as it would have caused data leakage between the testing
and training sets. Instead I used the training set to create 2 subsets which
would act as temporary ‘training’ and ‘test’ sets to score each of the features
individually. In most executions, the features with the top 4 scores are as
shown in figure 2. However, due to the pseudo-random nature of k-means
clustering, the resulting feature can sometimes achieve an individual
accuracy score of 0.820, therefore entering the top 4 in place of the feature f3
* f10. After identifying the 4 features to be used in 5-NN classification, all remaining features are removed from new_feats_train and new_feats_test. The 5-NN classier is then trained using new_feats_train and outcomes_train and tested using new_feats_test and outcomes_test.
Principal Component Analysis (PCA) Implementation
I applied the PCA class from the sklearn library, with the number of components set to 4 as per the project specifications. As the specifications are quite vague as to which dataset PCA should be applied to, I chose to apply it to base_train (containing the original 20 features imputed and scaled) as the prerogative of Task-2B seems to be focused around comparing different feature engineering and selection methods performed on the same dataset. This is evident in the fact that the other two directives in the task, the feature set based on interaction term pairs and cluster labels, and the feature set based on the first four features, both originate from the original 20 features in world.csv. Furthermore, PCA is described as ‘an alternative method of performing feature engineering’. As such I fit the PCA class with base_train and used it transform pca_train and pca_test (each with a dimensionality of 4), followed by using StandardScaler() fitted with base_train to transform the two sets. I then trained the 5-NN classifer using pca_train and outcomes_train and tested it using pca_test and outcomes_test.
First Four Features
Lastly, I condensed each record in base_train and base_test from 20 features to 4 and trained and classified the 5-NN classifier accordingly.
Which of the three methods investigated in Task-2B produced the best results for classification using 5-NN and why this was the case.
The method with the best accuracy is the one involving feature engineering based on interaction term pairs and clustering labels (referred to as method 1). One of the main contributing factors to this success is the purposeful feature selection. By being able to quantitatively analyse the performance of the 211 features and select an optimal 4 features from them, this method eliminates the impact that less unhelpful features have on the classification, and maximises the impact of the more helpful features. On the other hand, PCA (referred to as method 2) considers all 20 of the original features equally, therefore not benefiting from this effect of feature selection. Whilst the classifier trained on the first four features alone (referred to as method 3) implements some feature selection, it is essentially random as no analysis is done on any of the features selected. Furthermore, method 2 and 3 lack the variety of features available to method 1, therefore reducing the accuracy of their associated classifiers. To test this assumption I employed an additional PCA method, with the same number of components, on the set of 211 features that is used in feature selection. The result of the classifier trained via this method is equal to or sometimes greater than that of method 1 (to 3 decimal places) supporting the idea that method 1’s superior number of features available results in a higher accuracy than methods 2 and 3.
What other techniques you could implement to improve classification accuracy with this data.
One area that could be altered to improve classification accuracy is feature selection. Two of the most commonly selected features involve f3 in their generation. This could ultimately result in feature 3 being overly dominant in the training of the classifier. Similarly, out of the 7 features used in the generation of the 4 features depicted in figure 2, 3 of them (f1, f15 and f12) are contained ‘pollution’ or ‘emission’ in the feature headers in world.csv, resulting in like potential issues as the prevalence of pollution in a country could be overcompensated for by the classifier. One possible solution is to analyse all the combinations of length 4 within the set of 211 features, instead of analysing each one individually. This could result in a more balanced selection of features, giving the classifier more information to work with. An issue with this solution is that the associated time complexity of analysing all the combinations would be O(C(n, 4)), making this operation very computationally expensive.
How reliable I consider my classification model to be.
Overall, I consider the classification model to be fairly reliable, particularly method 1, which illustrates a consistent accuracy of 0.852, which is 3.2% more effective than the basic 5-NN classification implemented in Task2A. Despite this, I would advise against any serious use of this classifier as it currently exists, as I have yet to analyse what changes could be made regarding; optimisation of selected k value used in establishing KNearestNeighbours model, and the number of features selected as part of feature selection. As seen in Task2A, the k value chosen as part of K-NN can have a somewhat serious impact on the resulting accuracy. Additionally, the top 3 individual feature scores illustrated in figure 2 are all higher than the final accuracy of method 1, suggesting there are possible optimisations to the number of features selected for classification that could increase the reliability of the classification model. As such, I would consider this reliability to be sup-optimal in its current state.
