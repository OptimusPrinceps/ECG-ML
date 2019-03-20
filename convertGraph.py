'''
This file saves a tflearn model as a protobuf file, then converts that to a .mlmodel format

Steps:
- Define the computation graph
- Load model weights from checkpoint
- Fix bug #1
- Transfer graph and and values to a tensorflow session
- Fix bug #2
- Freeze weights
- Write graph to output .pb file
- Convert .pb file to .mlmodel file

Author: Josh Atwal
'''

import tflearn
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tflearn.layers.core import input_data, dropout, fully_connected, activation, flatten
from tflearn.layers.conv import conv_1d, avg_pool_1d, max_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression
from tflearn.activations import relu, softmax
from CNetworks import CRNN
import tfcoreml

tflearn.config.init_training_mode()

## --- Defining Computational Graph 
model = tflearn.DNN(CRNN())

## --- Loading trained model from checkpoint
model.load("model_1.tflearn")

## --- fix "Adam not defined" bug
with model.session.as_default():
	del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]

## --- Transferring tflearn graph to tensorflow session
sess = model.session
input_graph_def = sess.graph_def

## --- fix batch norm bug
for node in input_graph_def.node:
  if node.op == 'RefSwitch':
    node.op = 'Switch'
    for index in range(len(node.input)):
      if 'moving_' in node.input[index]:
        node.input[index] = node.input[index] + '/read'
  elif node.op == 'AssignSub':
    node.op = 'Sub'
    if 'use_locking' in node.attr: del node.attr['use_locking']

## --- Freeze graph and write to .pb file
output_graph = 'frozen_graph.pb'
output_graph_def = graph_util.convert_variables_to_constants(sess,input_graph_def,['FullyConnected/Softmax'])
tf.gfile.GFile(output_graph, "wb").write(output_graph_def.SerializeToString())

## --- Convert to .mlmodel format
tf.coreml.convert(
	tf_model_path = "frozen_graph.pb",
	mlmodel_path = "CRNN.mlmodel",
	input_name_shape_dict={"Input:0":[1,1000]}, #Input is a vector of length 1000
	output_feature_names= ["Softmax:0"]
)