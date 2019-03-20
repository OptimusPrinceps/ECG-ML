"""
Train and validate the RNN approach

Author: Josh Atwal
"""
from __future__ import division, print_function, absolute_import
import random
from datetime import datetime
from os import listdir, makedirs

import numpy as np
from scipy.io import loadmat, savemat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import scale, LabelBinarizer
import tflearn
from tflearn.layers.core import dropout
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tensorflow import Graph, reset_default_graph, set_random_seed

def main():
	reset_default_graph()
	nLabels = 4

	tbDir = '/hpc/jatw032/TF_RunLogs/'
	
	#Load data
	train = loadmat('../data/trainingSet_w1000.mat')
	X_train, Y_train = train['train'][0], train['trainLabels'][0]

	test = loadmat('../data/validationSet_w1000.mat')
	X_test, Y_test = test['test'][0], test['testLabels'][0]
	Y_test1D = test['testLabels'][0] - 1

	#One hot encoding
	labBin = LabelBinarizer()
	Y_train = labBin.fit_transform(Y_train)
	Y_test = labBin.fit_transform(Y_test)

	#Prepare data for RNN
	dsf = 2
	dslen = int(np.ceil(18286/dsf))
	X = np.zeros([len(X_train), dslen, 1])
	for i in range(len(X)):
		temp = scale(X_train[i][0][np.arange(0,len(X_train[i][0]),dsf)])
		sigLen = len(temp)
		X[i][:sigLen] = temp.reshape([sigLen,1])

	X_t = np.zeros([len(X_test), dslen, 1])
	for i in range(len(X_test)):
		temp = scale(X_test[i][0][np.arange(0,len(X_test[i][0]),dsf)])
		sigLen = len(temp)
		X_t[i][:len(temp)] = temp.reshape([sigLen,1])

	
	# Build neural network
	rnn = constructRnn(nLabels, dslen)
	model = tflearn.DNN(rnn, tensorboard_verbose=0, clip_gradients=0.)
	

	############### Fit model #############

	nEpochs, bSize = 200, 500
	maxF1, maxF1_i, maxF1_R, maxF1_i_R = 0, 0, 0, 0
	#_, indices = np.unique(sKey, return_index=True)
	#scores = np.zeros([nEpochs, ])

	outfile = "./results/RNN_only_run.csv"
	modelName = '/rnnModel'
	with open(outfile,"w") as f:
		f.write("Epoch, overall, normal, af, other, noisy, best, bestindex\n")
	
	# training and prediction loop for the number of epochs desired
	
	for e in range(nEpochs):
		#Fit model
		model.fit(X, Y_train, n_epoch=1, show_metric=True, validation_set=(X_t, Y_test), shuffle=True, run_id = "abc123", batch_size=10)  # fit model

		#Predictions
		rnnPredictions = model.predict(X_t)

		#Calculate metrics
		rnnff1 = f1_score(Y_test1D, np.squeeze(np.argmax(rnnPredictions,axis=1)), average=None)  
		# calculate F1 score for RNN
		rF1 = np.mean(rnnff1)
		cm_r = confusion_matrix(Y_test1D, np.argmax(rnnPredictions,axis=1))
		if rF1 > maxF1_R:
			maxF1_R, maxF1_i_R, maxff1_R, cmax_r = rF1, e, rnnff1, cm_r
			model.save(tbDir+modelName)
		
		#Output results
		with open(outfile,"a") as f:
			f.write("{},{},{},{},{},{},{},{}\n".format(str(e),str(rF1), str(rnnff1[0]), str(rnnff1[1]), str(rnnff1[2]), str(rnnff1[3]), str(maxF1_R), str(maxF1_i_R))) 
		
		
	with open(outfile,"a") as f:
		f.write("\nConfusionMatrix\n{}".format(str(cmax_r))) 


#Make predictions in batches
def makePredictions(dims, X_test, bSize, model):
	predictions = np.zeros(dims)  # preallocate predictions matrix
	print(dims)
	print(X_test.shap)
	for i in range(X_test.shape[0] // bSize):
		# predict in batches
		predictions[i*bSize:(i+1)*bSize,:] = model.predict(X_test[i*bSize: (i+1)*bSize])
	if X_test.shape[0] % bSize != 0: #if dimensions do not fit into batch size
		i += 1
		predictions[i*bSize:,:] = model.predict(X_test[i*bSize:])

	return predictions

#Define RNN architecture
def constructRnn(nLabels, dslen, dropoutProb=0.80, featureMap=False):
	rnet = tflearn.input_data(shape=[None, dslen,1]) 
	r1 = tflearn.lstm(rnet, 64, return_seq=True,dynamic=True)#64 nodes
	r2 = dropout(r1, dropoutProb)
	r2 = tflearn.lstm(r2, 256,return_seq=True) #256 nodes
	r3 = dropout(r2, dropoutProb)
	r3 = tflearn.lstm(r3, 100) #100 nodes
	out = dropout(r3, dropoutProb)
	out = tflearn.fully_connected(out, nLabels, activation='softmax')
	out = tflearn.regression(out, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.01, shuffle_batches=False)
	if featureMap:
		return r1, r2, r3, out
	else:
		return out


main()
