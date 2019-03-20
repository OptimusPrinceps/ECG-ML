"""
This file defines the architecture of the convolutional neural networks used and tested in this project

Author: Josh Atwal, TFLearn (where specififed)
"""
import tensorflow as tf
import tflearn
from tflearn.activations import relu, softmax
from tflearn.layers.conv import (avg_pool_1d, avg_pool_2d, conv_1d, conv_2d,
                                 max_pool_1d, max_pool_2d)
from tflearn.layers.core import (activation, dropout, flatten, fully_connected,
                                 input_data)
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import (batch_normalization,
                                          local_response_normalization)

#Convolutional component of the CRNN architecture, my own implementation of 
#https://arxiv.org/abs/1707.01836
def CRNN(window=1500,nLabels=3,downsampleSecond=True,featureMap=False):
	#Residual block function adapted from TFLearn:
	#https://github.com/tflearn/tflearn/blob/master/tflearn/layers/conv.py
	def residual_block_1D(incoming,out_channels,downsample=False, first=False, filt_len=16, dropout_prob=0.85, downsampleSecond=True):
		resnet = incoming
		in_channels = incoming.shape[-1].value
		strides = (2 if downsample else 1)
		dsLayer = (1 if downsampleSecond else 0)
		identity = resnet

		nConv = 2
		if first:
			resnet = conv_1d(resnet, out_channels, filt_len, strides,weights_init="variance_scaling")
			nConv = 1

		for i in range(nConv):
			resnet = batch_normalization(resnet)
			resnet = relu(resnet)
			resnet = dropout(resnet, dropout_prob)
			if downsample and i==dsLayer: #1 as in, second layer
				resnet = conv_1d(resnet,out_channels,filt_len, strides=1, weights_init="variance_scaling") #puts the downsampling on the first conv layer only
			else:
				resnet = conv_1d(resnet,out_channels,filt_len, strides, weights_init="variance_scaling")

		#Beginning of skip connection
		identity = max_pool_1d(identity,strides, strides)

		if in_channels != out_channels:

			ch = (out_channels - in_channels) // 2
			identity = tf.pad(identity,[[0,0],[0,0],[ch,ch]])
			in_channels = out_channels

		resnet = resnet + identity
		
		return resnet

	#Begin construction of network
	net = input_data(shape=[None, window, 1])

	net = conv_1d(net, 64, 16, weights_init="variance_scaling")
	net = batch_normalization(net)
	net = relu(net)

	dropoutProb = 0.5
	net = residual_block_1D(net, 64, first=True, dropout_prob=dropoutProb)

	for i in range(0,4):
		downsample = (i%2 == 0)
		k = ((i+1)//4)+1
		net = residual_block_1D(net, 64*k, downsample, downsampleSecond=downsampleSecond, dropout_prob=dropoutProb)
	
	res1 = net

	for i in range(4, 8):
		downsample = (i%2 == 0)
		k = ((i+1)//4)+1
		net = residual_block_1D(net, 64*k, downsample, downsampleSecond=downsampleSecond, dropout_prob=dropoutProb)
	res2 = net

	for i in range(8,12):
		downsample = (i%2 == 0)
		k = ((i+1)//4)+1
		net = residual_block_1D(net, 64*k, downsample, downsampleSecond=downsampleSecond, dropout_prob=dropoutProb)
	res3 = net

	for i in range(12, 15):
		downsample = (i%2 == 0)
		k = ((i+1)//4)+1
		net = residual_block_1D(net, 64*k, downsample, downsampleSecond=downsampleSecond, dropout_prob=dropoutProb)
	res4 = net

	net = batch_normalization(net)
	net = relu(net)
	
	net = fully_connected(net, nLabels)
	net = softmax(net)
	
	net = regression(net, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001, shuffle_batches=False)

	#Return intermediary activations
	if featureMap:
		return res1, res2, res3, res4, net
	else:
		return net


#Residual network architecture, from:
#https://github.com/tflearn/tflearn/blob/master/examples/images/resnext_cifar10.py
def resNet(nLabels, nFreq, nTime, featureMap=False):
	n=5
	tflearn.init_graph(gpu_memory_fraction=0.5, seed=6969)
	net = tflearn.input_data(shape=[None, nFreq, nTime, 1])
	net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
	res1 = tflearn.residual_block(net, n, 16)
	res2 = tflearn.residual_block(res1, 1, 32, downsample=True)
	res3 = tflearn.residual_block(res2, n-1, 32)
	res4 = tflearn.residual_block(res3, 1, 64, downsample=True)
	res5 = tflearn.residual_block(res4, n-1, 64)
	out = tflearn.batch_normalization(res5)
	out = tflearn.activation(out, 'relu')
	out = tflearn.global_avg_pool(out)
	out = tflearn.fully_connected(out, nLabels, activation='softmax')
	out = tflearn.regression(out, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)
	if featureMap:
		return res1, res2, res3, res4, res5, out
	else:
		return out

#Googlenet architecture, from:
#https://github.com/tflearn/tflearn/blob/master/examples/images/googlenet.py
def defGoogleNet(nLabels, nFreq, nTime):
	tflearn.init_graph(gpu_memory_fraction=0.5, seed=6969)
	network = tflearn.input_data(shape=[None, nTime, nFreq, 1])

	conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
	pool1_3_3 = max_pool_2d(conv1_7_7, 3, strides=2)
	pool1_3_3 = local_response_normalization(pool1_3_3)
	conv2_3_3_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
	conv2_3_3 = conv_2d(conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
	conv2_3_3 = local_response_normalization(conv2_3_3)
	pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

	# 3a
	inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
	inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
	inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128, filter_size=3,  activation='relu', name='inception_3a_3_3')
	inception_3a_5_5_reduce = conv_2d(pool2_3_3, 16, filter_size=1, activation='relu', name='inception_3a_5_5_reduce')
	inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name='inception_3a_5_5')
	inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, name='inception_3a_pool')
	inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')
	inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

	# 3b
	inception_3b_1_1 = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
	inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
	inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu', name='inception_3b_3_3')
	inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name='inception_3b_5_5_reduce')
	inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name='inception_3b_5_5')
	inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
	inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1, activation='relu', name='inception_3b_pool_1_1')
	inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat', axis=3, name='inception_3b_output')
	pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')

	# 4a
	inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
	inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
	inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
	inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
	inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
	inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')
	inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

	# 4b
	inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
	inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
	inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
	inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')
	inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
	inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')
	inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')

	# 4c
	inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
	inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
	inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
	inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
	inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')
	inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
	inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')
	inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3, name='inception_4c_output')

	# 4d
	inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
	inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
	inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
	inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
	inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
	inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
	inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')
	inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

	# 4e
	inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
	inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
	inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
	inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
	inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
	inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
	inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')
	inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=3, mode='concat')
	pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

	# 5a
	inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
	inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
	inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
	inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
	inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
	inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
	inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1, activation='relu', name='inception_5a_pool_1_1')
	inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3, mode='concat')

	# 5b
	inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
	inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
	inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3, activation='relu', name='inception_5b_3_3')
	inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
	inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_5b_5_5')
	inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
	inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
	inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')
	pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
	pool5_7_7 = dropout(pool5_7_7, 0.2)

	net = tflearn.fully_connected(pool5_7_7, nLabels, activation='softmax')
	net = tflearn.regression(net, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001, shuffle_batches=False)
	return net
