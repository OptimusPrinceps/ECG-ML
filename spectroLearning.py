"""
This file trains and validates the Spectrogram learning approach using the Resnet architecture

Author: Josh Atwal
"""
from __future__ import absolute_import, division, print_function

from datetime import datetime
from os import listdir, makedirs

import numpy as np
from scipy.io import loadmat, savemat
from sklearn.metrics import confusion_matrix, f1_score

import tflearn
from tensorflow import reset_default_graph
from CNetworks import resNet


def main():
	reset_default_graph()
	
	#Load data
	nLabels = 4
	X_train, Y_train, Y_train1D, sKeyTrain, X_test, Y_test, Y_test1D, sKeyTest, window, nSubSamples = loadData()

	nFreq, nTime = X_train.shape[1:3]
	tbdir = '/hpc/jatw032/TF_RunLogs'
	
	#Setup graph
	tflearn.init_graph(gpu_memory_fraction=0.5, seed=6969)
	net = resNet(nLabels, nFreq, nTime)
	model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir=tbdir,clip_gradients=0.)

	#Validation data
	Y_test_1D = np.argmax(Y_test, axis=1)
	nEpochs = 100
	maxF1 = 0
	bSize = 500
	

	modelName = '/BestSpecModel'

	#Setup Output file
	outfile = "./results/Spec_run.csv"
	with open(outfile,"w") as f:
		f.write("Epoch, overall, normal, af, other, noisy, best, bestindex\n")

	#Begin training
	for e in range(nEpochs):
		#Fit model
		model.fit(X_train, Y_train, n_epoch=1, show_metric=True, validation_set=(X_test, Y_test), shuffle=False, run_id='abc123') #fit model
		
		#Make predictions
		testPredictions = makePredictions(Y_test.shape, X_test, bSize, model)

		# redefine the predictions into a per signal basis
		rnnTesting, sPredictions = signalReformat(testPredictions, sKeyTest, nSubSamples, nLabels, returnPredictions=True)
		
		#Calculate metrics
		ff1 = f1_score(Y_test1D, sPredictions, average=None)  # calculate F1 score
		cm = confusion_matrix(Y_test1D, sPredictions)
		f1 = np.mean(ff1)
		if f1 > maxF1:
			maxF1, maxF1_i, maxff1, cmax = f1, e, ff1, cm
			model.save(tbdir+modelName)

		#Output results
		with open(outfile,"a") as f:
			f.write("{},{},{},{},{},{},{},{}\n".format(str(e),str(f1), str(ff1[0]), str(ff1[1]), str(ff1[2]), str(ff1[3]), str(maxF1), str(maxF1_i))) 

		
	with open(outfile,"a") as f:
		f.write("\nConfusionMatrix\n{}".format(str(cmax))) 
	
	#Normal signal for intermediate activations
	normalSig = X_train[300,:]
	normalSig = np.reshape(normalSig, [1]+list(normalSig.shape))

	#Noisy signal for intermediate activations
	noisySig = X_train[84,:]
	noisySig = np.reshape(noisySig, [1]+list(noisySig.shape))

	#Reset graph, load weights and gather predictions for intermediate layers
	reset_default_graph()
	res1, res2, res3, res4, res5, out = resNet(nLabels, nFreq, nTime, True)
	model = tflearn.DNN(res1)
	model.load(tbdir+modelName)	
	normalpred1 = model.predict(normalSig)
	noisypred1 = model.predict(noisySig)
	
	reset_default_graph()
	res1, res2, res3, res4, res5, out = resNet(nLabels, nFreq, nTime, True)
	model = tflearn.DNN(res2)
	model.load(tbdir+modelName)	
	normalpred2 = model.predict(normalSig)
	noisypred2 = model.predict(noisySig)

	reset_default_graph()
	res1, res2, res3, res4, res5, out = resNet(nLabels, nFreq, nTime, True)
	model = tflearn.DNN(res3)
	model.load(tbdir+modelName)	
	normalpred3 = model.predict(normalSig)
	noisypred3 = model.predict(noisySig)
	
	reset_default_graph()
	res1, res2, res3, res4, res5, out = resNet(nLabels, nFreq, nTime, True)
	model = tflearn.DNN(res4)
	model.load(tbdir+modelName)	
	normalpred4 = model.predict(normalSig)
	noisypred4 = model.predict(noisySig)
	
	reset_default_graph()
	res1, res2, res3, res4, res5, out = resNet(nLabels, nFreq, nTime, True)
	model = tflearn.DNN(res5)
	model.load(tbdir+modelName)	
	normalpred5 = model.predict(normalSig)
	noisypred5 = model.predict(noisySig)
	
	#Save intermediate results
	savemat('./results/specFeatures.mat', dict([('normalpred1', normalpred1), ('normalpred2', normalpred2), ('normalpred3', normalpred3), ('normalpred4', normalpred4), ('normalpred5', normalpred5), ('normalinput', normalSig), ('noisypred1', noisypred1), ('noisypred2', noisypred2), ('noisypred3', noisypred3), ('noisypred4', noisypred4), ('noisypred5', noisypred5), ('noisyinput', noisySig)]))
	
#Function for reformatting predictions from per-segment to per-signal basis
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

#Function for performing predictions in batches
def makePredictions(dims, X_test, bSize, model):
	predictions = np.zeros(dims)  # preallocate predictions matrix
	for i in range(X_test.shape[0] // bSize):
				# predict in batches
		predictions[i*bSize:(i+1)*bSize,:] = model.predict(X_test[i*bSize: (i+1)*bSize])
	if X_test.shape[0] % bSize != 0: #if dimensions do not fit into batch size
		i += 1
		predictions[i*bSize:,:] = model.predict(X_test[i*bSize:])

	return predictions

#Function for loading data
def loadData():
	
	mat, testMat = loadmat('../data/trainingSpectrogram.mat'), loadmat('../data/validationSpectrogram.mat')

	X, Y, Y_train1D, window, sKeyTrain, nSubSamples = mat["Spec"], mat["sLabels"].reshape([-1, 4]), mat['trainLabels'], mat["window"][0][0], mat['sKey'][0], mat['nSubSamples'][0][0]

	X_test, Y_test, sKeyTest, Y_test1D = testMat["Spec"], testMat["sLabels"], testMat["sKey"][0], testMat['testLabels']

	X = np.reshape(X, list(X.shape)+[1])
	X_test = np.reshape(X_test, list(X_test.shape)+[1])
	Y_train1D = np.squeeze(Y_train1D) -1 #0 indexing
	Y_test1D = np.squeeze(Y_test1D) -1 #0 indexing

	return X, Y, Y_train1D, sKeyTrain, X_test, Y_test, Y_test1D, sKeyTest, window, nSubSamples


main()
