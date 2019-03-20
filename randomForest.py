"""
This file fits a Random Forest Classifier                  
                                                             
Author: Josh Atwal                     
"""

from sklearn.ensemble import RandomForestClassifier
from scipy.io import loadmat
from sklearn.metrics import f1_score
from sklearn.preprocessing import Imputer, scale
from numpy import mean, arange, concatenate, zeros 
from sklearn.cross_validation import StratifiedKFold

dataPath = '../data/'

#Imputer
imp = Imputer(axis=1,strategy='mean') 

# Load data
X = loadmat(dataPath + "trainingFeatures.mat")['F'].transpose()
Y = loadmat(dataPath + 'trainingSet.mat')['trainLabels'][0]
Y = concatenate([Y, loadmat(dataPath + 'testingSet.mat')['testLabels'][0]])

# Remove features with too many missing values
indx = [0,1,2,3] + [x for x in range(11,33)]
X = X[:,indx]

# Impute missing values
X = imp.fit_transform(X)

# Normalise data
X = scale(X)


####################     Training      #######################

# Defining random forest model
RF = RandomForestClassifier(n_jobs=-1, class_weight="balanced", n_estimators=40, criterion='entropy',  min_samples_split=2, max_features=28) 

# Cross validation
nFolds = 5

skf = StratifiedKFold(Y, n_folds=nFolds)
scores = zeros([nFolds, 5])

# Cross validation loop
i = 0
for train_index, test_index in skf:
    X_train, X_test = X[train_index,:], X[test_index,:]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Fit model
    RF.fit(X_train,Y_train)
    scores[i,1:] = f1_score(Y_test, RF.predict(X_test),average=None)
    i += 1

scores[:,0] = mean(scores[:,1:], axis=1)
f1 = mean(scores, axis=0) #average CV score

print("F1: " + str(mean(f1)))
