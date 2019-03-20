"""
This file trains and validates the convolutional recurrent neural network approach

Author: Josh Atwal, TFLearn (where specififed)
"""
from __future__ import division, print_function, absolute_import
import pickle
import random
from datetime import datetime
from os import listdir, makedirs

import numpy as np
from scipy.io import loadmat, savemat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer

import tflearn
from CNetworks import CRNN
from tensorflow import Graph, reset_default_graph, set_random_seed, global_variables_initializer

def main():
	reset_default_graph()
	nLabels = 4
	# load data, combStrategy can be 'af+other', 'other+noisy', 'drop other'
	X, Y, Y_train1D, sKeyTrain, X_test, Y_test, Y_test1D, sKeyTest, window, nSubSamples = loadData()
	
	#One hot encoding
	lb = LabelBinarizer()
	Y_train_R = lb.fit_transform(Y_train1D)
	lb = LabelBinarizer()
	Y_test_R = lb.fit_transform(Y_test1D)
	
	#Log directory
	tbDir = '/hpc/jatw032/TF_RunLogs/'

	#Normalise input data
	X, X_test = normalise(X, scale=False, sKey=sKeyTrain), normalise(X_test, scale=False, sKey=sKeyTest)

	# Build neural network
	g1, g2 = Graph(), Graph()
	with g1.as_default():
		#Convolutional component
		tflearn.init_graph(gpu_memory_fraction=0.5, seed=6969)
		net = CRNN(nLabels=nLabels,window=window, downsampleSecond=False, featureMap=False)
		runID = getRunID(tbDir) #False = downsample on the second layer
		model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir=tbDir+runID, clip_gradients=0.)
	
	with g2.as_default():
		#Recurrent component	
		rnn = constructRnn(nLabels, nSubSamples)
		modelRnn = tflearn.DNN(rnn, tensorboard_verbose=0, tensorboard_dir=tbDir+runID)
	
	
	############### Fit model #############

	nEpochs, bSize = 25, 500
	maxF1, maxF1_R, cmax_r = 0, 0, 0
	
	modelName = 'BestCRNNModel'
	#Initialise file to print results
	outfile = "./results/CRNN.csv"
	with open(outfile,"w") as f:
		f.write("Epoch, overall, normal, af, other, noisy, best, bestindex, RNNEpoch, overallRNN, normalRNN, afRnn, otherRnn, noisyRnn, bestRNN, bestindexRnn, f1train, accuracyTrain, accTest, accTest_R, f1_train_R\n")
	
	# training and prediction loop for the number of epochs desired
	for e in range(nEpochs):
		#Train model for one epoch
		model.fit(X, Y, n_epoch=1, show_metric=True, validation_set=(X_test, Y_test), run_id=runID, shuffle=True, r	)  # fit model
		
		#Make predictions on model
		trainingPredictions, testPredictions = makePredictions(Y.shape, X, bSize, model), makePredictions(Y_test.shape, X_test, bSize, model)

		# redefine the predictions into a per signal basis
		rnnTraining, sTrainPredictions = signalReformat(trainingPredictions, sKeyTrain, nSubSamples, nLabels, returnPredictions=True)
		rnnTesting, sPredictions = signalReformat(testPredictions, sKeyTest, nSubSamples, nLabels, returnPredictions=True)
		
		#Calculate metrics
		ff1_train = f1_score(Y_train1D, sTrainPredictions, average=None)  # calculate F1 score
		acc_train = accuracy_score(Y_train1D, sTrainPredictions)

		ff1 = f1_score(Y_test1D, sPredictions, average=None)  # calculate F1 score
		acc = accuracy_score(Y_test1D, sPredictions)

		cm = confusion_matrix(Y_test1D, sPredictions)
		f1 = np.mean(ff1)
		if f1 > maxF1:
			maxF1, maxF1_i = f1, e
			
		with g2.as_default():
			global_variables_initializer()
			
		# Train RNN based on predictions of CNN on the training set
		for er in range(50):
			#Train
			modelRnn.fit(rnnTraining, Y_train_R, n_epoch=1, show_metric=True, validation_set=(rnnTesting, Y_test_R), run_id=runID+str(e), shuffle=True)

			#Predict
			rnnTrainPredictions = makePredictions(Y_train_R.shape, rnnTraining, bSize, modelRnn)
			rnnPredictions = makePredictions(Y_test_R.shape, rnnTesting, bSize, modelRnn)
			
			#Calculate metrics
			rnnff1_train = f1_score(Y_train1D, np.argmax(rnnTrainPredictions,axis=1), average=None)  # calculate F1 score for RNN
			acc_R_train = accuracy_score(Y_train1D, np.argmax(rnnTrainPredictions,axis=1))

			rnnff1 = f1_score(Y_test1D, np.argmax(rnnPredictions,axis=1), average=None)  # calculate F1 score for RNN
			acc_R = accuracy_score(Y_test1D, np.argmax(rnnPredictions,axis=1))
			cm_r = confusion_matrix(Y_test1D, np.argmax(rnnPredictions,axis=1))
			rF1 = np.mean(rnnff1)
			if rF1 > maxF1_R:
				maxF1_R, maxF1_i_R, cmax_r = rF1, er, cm_r
				savemat('./results/CRNNPreds.mat',{'rnnTraining':rnnTraining, 'Y_train_R':Y_train_R, 'rnnTesting':rnnTesting, 'Y_test_R':Y_test_R, 'Y_train1D':Y_train1D, 'Y_test1D':Y_test1D})
			
			#Output results to file
			with open(outfile,"a") as f:
				f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(str(e),str(f1), str(ff1[0]), str(ff1[1]), str(ff1[2]), str(ff1[3]), str(maxF1), str(maxF1_i),str(er),str(rF1),str(rnnff1[0]),str(rnnff1[1]),str(rnnff1[2]),str(rnnff1[3]),str(maxF1_R),str(maxF1_i_R),str(np.mean(ff1_train)), str(acc_train), str(acc), str(acc_R), str(acc_R_train), str(np.mean(rnnff1_train)))) 

	#Output the best confusion matrix at the end
	with open(outfile,"a") as f:
		f.write("\nConfusionMatrix\n{}".format(str(cmax_r))) 
			
	#Normal signal to be used for gathering intermediate activations
	normalSig = X[300,:]
	normalSig = np.reshape(normalSig, [1]+list(normalSig.shape))

	#Noisy signal to be used for gathering intermediate activations
	noisySig = X[34,:]
	noisySig = np.reshape(noisySig, [1]+list(noisySig.shape))

	#Reset graph and load weights of best performing model
	reset_default_graph()
	res1, res2, res3, res4, net = CRNN(nLabels=nLabels,window=window, downsampleSecond=False, featureMap=True)
	model = tflearn.DNN(res1)
	model.load(tbDir+modelName)	
	normpred1 = model.predict(normalSig)
	
	reset_default_graph()
	res1, res2, res3, res4, net = CRNN(nLabels=nLabels,window=window, downsampleSecond=False, featureMap=True)
	model = tflearn.DNN(res2)
	model.load(tbDir+modelName)	
	normpred2 = model.predict(normalSig)
	
	reset_default_graph()
	res1, res2, res3, res4, net = CRNN(nLabels=nLabels,window=window, downsampleSecond=False, featureMap=True)
	model = tflearn.DNN(res3)
	model.load(tbDir+modelName)	
	normpred3 = model.predict(normalSig)
	
	reset_default_graph()
	res1, res2, res3, res4, net = CRNN(nLabels=nLabels,window=window, downsampleSecond=False, featureMap=True)
	model = tflearn.DNN(res4)
	model.load(tbDir+modelName)	
	normpred4 = model.predict(normalSig)

	# noisy
	reset_default_graph()
	res1, res2, res3, res4, net = CRNN(nLabels=nLabels,window=window, downsampleSecond=False, featureMap=True)
	model = tflearn.DNN(res1)
	model.load(tbDir+modelName)	
	noisypred1 = model.predict(normalSig)
	
	reset_default_graph()
	res1, res2, res3, res4, net = CRNN(nLabels=nLabels,window=window, downsampleSecond=False, featureMap=True)
	model = tflearn.DNN(res2)
	model.load(tbDir+modelName)	
	noisypred2 = model.predict(normalSig)
	
	reset_default_graph()
	res1, res2, res3, res4, net = CRNN(nLabels=nLabels,window=window, downsampleSecond=False, featureMap=True)
	model = tflearn.DNN(res3)
	model.load(tbDir+modelName)	
	noisypred3 = model.predict(normalSig)
	
	reset_default_graph()
	res1, res2, res3, res4, net = CRNN(nLabels=nLabels,window=window, downsampleSecond=False, featureMap=True)
	model = tflearn.DNN(res4)
	model.load(tbDir+modelName)	
	noisypred4 = model.predict(normalSig)
	
	#Save intermediate activations
	savemat('./results/crnnFeatures.mat', dict([('pred1', normpred1), ('pred2', normpred2), ('pred3', normpred3), ('pred4', normpred4), ('input', normalSig), ('npred1', noisypred1),('npred2', noisypred2),('npred3', noisypred3),('npred4', noisypred4),('ninput', noisySig)]))

#Function to normalise data through standard deviation scaling
def normalise(X, scale=True, sKey=None):
	X = X.astype(float)
	if sKey is None:
		for i in range(X.shape[0]):
			X[i] = (X[i] - np.mean(X[i]))/np.std(X[i])
			if scale:
				minX, maxX = np.min(X[i]), np.max(X[i])
				X[i] -= minX
				X[i] = X[i]/(maxX-minX)
	else:
		u = np.unique(sKey)
		for j in range(len(u)):
			i = j+1
			X[sKey == i, :] = (
			X[sKey == i, :] - np.mean(X[sKey == i, :]))/np.std(X[sKey == i, :])
			if scale:
				minX, maxX = np.min(X[sKey == i, :]), np.max(X[sKey == i, :])
				X[sKey == i, :] -= minX
				X[sKey == i, :] = X[sKey == i, :]/(maxX-minX)
	return X

#Function for loading data
def loadData():
	mat, testMat = loadmat('../data/trainingSet_w1000.mat'), loadmat('../data/validationSet_w1000.mat')
	
	X, Y, Y_train1D, window, sKeyTrain, nSubSamples = mat["S"], mat["sLabels"].reshape([-1, 4]), mat['trainLabels'], mat["window"][0][0], mat['sKey'][0], mat['nSubSamples'][0][0]

	X_test, Y_test, sKeyTest, Y_test1D = testMat["S"], testMat["sLabels"], testMat["sKey"][0], testMat['testLabels']
	
	X = X.reshape([-1, window, 1])
	X_test = X_test.reshape([-1, window, 1])
	Y_train1D = np.squeeze(Y_train1D) -1 #0 indexing
	Y_test1D = np.squeeze(Y_test1D) -1 #0 indexing

	return X, Y, Y_train1D, sKeyTrain, X_test, Y_test, Y_test1D, sKeyTest, window, nSubSamples

#Function for defining identification for run
def getRunID(path):
	date = datetime.now()
	dayofweek = date.strftime('%a')
	dayofmonth = date.strftime('%d')
	month = date.strftime('%b')
	hour = date.strftime('%H')
	minute = date.strftime('%M')
	time = dayofweek+dayofmonth+month+"_"+hour+":"+minute
	runNumber = len(listdir(path))+1
	runid = ("Run%.3d_" % runNumber)+time
	makedirs(path+runid)
	return runid

#Function for reformatting signal predictions from a per-segment to a per-signal basis
def signalReformat(predictions, sKey, nSubSamples, nLabels, returnPredictions=False):
	if returnPredictions:
		sPredictions = np.zeros([np.unique(sKey).shape[0], ])
		p = np.zeros(predictions[0, :].shape)
	rnnInput = np.zeros([np.unique(sKey).shape[0], nSubSamples, nLabels])
	l, j, k = sKey[0], 0, 0
	for i in range(predictions.shape[0]):
		if sKey[i] == l:
			if returnPredictions:
				p += predictions[i, :]
			rnnInput[j,k,:] = predictions[i,:]
			k+=1
		else:
			if returnPredictions:
				sPredictions[j] = np.argmax(p)
				p = predictions[i, :]
			l = sKey[i]
			j, k, = j+1, 0
			rnnInput[j,k,:] = predictions[i,:]

	if returnPredictions:
		sPredictions[j] = np.argmax(p)
		return rnnInput, sPredictions
	return rnnInput

#Performs predictions
def makePredictions(dims, X_test, bSize, model):
	predictions = np.zeros(dims)  # preallocate predictions matrix
	for i in range(X_test.shape[0] // bSize):
				# predict in batches
		predictions[i*bSize:(i+1)*bSize,:] = model.predict(X_test[i*bSize: (i+1)*bSize])
	if X_test.shape[0] % bSize != 0: #if dimensions do not fit into batch size
		i += 1
		predictions[i*bSize:,:] = model.predict(X_test[i*bSize:])

	return predictions

#Defines the Recurrent neural network architecture
def constructRnn(nLabels, nSubSamples):
	rnet = tflearn.input_data(shape=[None, nSubSamples, nLabels])
	rnet = tflearn.lstm(rnet, 512, return_seq=True)
	rnet = tflearn.dropout(rnet, 0.8)
	rnet = tflearn.lstm(rnet, 512, return_seq=True)
	rnet = tflearn.dropout(rnet, 0.8)
	rnet = tflearn.lstm(rnet, 512)
	rnet = tflearn.dropout(rnet, 0.8)
	rnet = tflearn.fully_connected(rnet, nLabels, activation='softmax')
	rnet = tflearn.regression(rnet, optimizer='adam',loss='categorical_crossentropy')
	return rnet


main()

# Encoding
# 1 = Normal, 2 = AF, 3 = Other, 4 = Noisy
