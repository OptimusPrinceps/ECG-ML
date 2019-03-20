"""
This file fits a Support Vector Machine classifier

Author: Josh Atwal
"""

from sklearn.svm import SVC
from scipy.io import loadmat
from sklearn.metrics import f1_score
from sklearn.preprocessing import Imputer, scale
import numpy as np
from numpy.random import randint
import itertools
from sklearn.cross_validation import StratifiedKFold

#Load data 
dataPath = '../data/'
X_train = loadmat(dataPath + "trainingFeatures.mat")['F'].transpose()
Y_train = loadmat(dataPath + 'trainingSet_w1000.mat')['trainLabels'][0]
X_test = loadmat(dataPath + 'validationFeatures.mat')['F'].transpose()
Y_test = loadmat(dataPath + 'validationSet_w1000.mat')['testLabels'][0]

X = np.concatenate([X_train, X_test])
Y = np.concatenate([Y_train, Y_test])

#Set up imputer to impute missing values with the mean
imp = Imputer(axis=1,strategy='mean')

n = X.shape[1]
best = 0
#Define model parameters
svmClf = SVC(kernel='rbf', class_weight='balanced',cache_size=4000, C=0.9, decision_function_shape='ovr')

#initial guess
solStr = np.array([0., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 0.,
       1., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1., 0., 1., 1., 0., 1.,
       0., 1., 0., 0.])

nFolds = 5
while True:
	# Keep track if any solutions were accepted during this pass over the neighbourhood
	flag = True
	
	while flag:
		flag = False
		for i in range(n): #explore neighbourhood
			solStr[i] = abs(solStr[i]-1) #toggle
			sol = np.where(solStr>0)
			#Scale and impute
			Xx = scale(imp.fit_transform(np.squeeze(X[:,sol])))
			
			skf = StratifiedKFold(Y, n_folds=nFolds)
			scores = np.zeros([nFolds, 5])

			# Cross validation
			i = 0
			for train_index, test_index in skf:
				X_train, X_test = Xx[train_index,:], Xx[test_index,:]
				Y_train, Y_test = Y[train_index], Y[test_index]
			
				# Fit SVM model
				svmClf.fit(X_train,Y_train)
				scores[i,1:] = f1_score(Y_test, svmClf.predict(X_test),average=None)
				i += 1
			
			scores[:,0] = np.mean(scores[:,1:], axis=1)
			f1 = np.mean(scores, axis=0) #average CV score

			if f1[0] > best:
				best = f1[0]
				flag = True #Better solution accepted
			else:
				solStr[i] = abs(solStr[i]-1) #un-toggle
	
	# Random restart
	solStr = np.zeros(n)
	sol = randint(0,n,randint(20,n))
	solStr[sol] = 1
  

#optimal Features:
#[ 'R_avg',
# ' R_std',
# ' HR_avg',
# ' HR_std',
# ' aMean',
# ' aSkew',
# ' aKurtosis',
# ' aRMS',
# ' aMAD',
# ' aMax',
# ' dSkew',
# ' dKurtosis',
# ' dSum',
# ' dMin',
# ' dMax',
# ' PR_std',
# ' TPeak_std']
