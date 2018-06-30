# simplex_operations2.py

import math
import numpy as np 
from scipy import linalg
import random as rand

def calc_error(network, target_values):
	output_layer = network.layers[2]
	total_error = 0
	for i in range(len(target_values)):
		total_error += 0.5 * (target_values[i] - output_layer.nodes[i].value) ** 2
	total_error /= len(target_values)
	return total_error[0]


def activate(x, function, prime = False):
	if function == 'sigmoid':
		if prime == False:
			return 1.0 / (1 + 2.781 ** -x)
		else:
			return x * (1 - x)
	elif function == 'tanh':
		if prime == False:
			return math.tanh(x)
		else:
			return 1 - x ** 2

# takes a layer and returns a column matrix of node values
def getNodeMatrix(layer):
	network = layer.network
	node_matrix = []
	if layer.layer_type != 'output' and layer.biased == True:
		for i in range(layer.dim + 1):
			node_matrix.append(layer.nodes[i].value)
		node_matrix = np.array([node_matrix])
		node_matrix = node_matrix.T
	else: 
		for i in range(layer.dim):
			node_matrix.append(layer.nodes[i].value)
		node_matrix = np.array([node_matrix])
		node_matrix = node_matrix.T
	return node_matrix

def activateMatrix(X, function, prime = False):
	if function == 'sigmoid':
		if prime == False:
			for i in range(X.shape[0]):
				for j in range(X.shape[1]):
					X[i][j] = activate(X[i][j], 'sigmoid', prime = False)
		else:
			X = X * (1 - X)
	return X

# takes a layer and returns a matrix of parent weight values
def getWeightMatrix(layer):
	weight_matrix = []
	for i in range(layer.dim):
		weight_matrix.append(layer.nodes[i].getWeightValues())
	weight_matrix = np.array(weight_matrix)
	return weight_matrix

def forward(network, input_matrix):
	if network.layers[0].biased == True:
		bias = network.layers[0].nodes[network.layers[0].dim].value
		input_matrix = appendBias(input_matrix, bias)
	network.layers[0].updateNodeValues(input_matrix) # updates input layer with input_matrix
	next_node_matrix = input_matrix
	for i in range(1, 3):
		weight_matrix = getWeightMatrix(network.layers[i])
		next_node_matrix = weight_matrix.dot(next_node_matrix)
		next_node_matrix = activateMatrix(next_node_matrix, network.layers[i].activation)
		network.layers[i].updateNodeValues(next_node_matrix)
		if network.layers[i].layer_type != 'output' and network.layers[i].biased == True:
			bias = network.layers[i].nodes[network.layers[i].dim].value
			next_node_matrix = appendBias(next_node_matrix, bias)
	return next_node_matrix


#  appends bias value to bottom of a column node matrix
def appendBias(node_matrix, bias):
	node_matrix = list(node_matrix)
	node_matrix.append(np.array([bias]))
	node_matrix = np.array(node_matrix)
	return node_matrix


# takes a weight_matrix and returns the change in cost w.r.t change in weight 
def get_delError(network, weight_matrix, weight_type, target_values):
	# output
	output_layer = network.layers[2]
	hidden_layer = network.layers[1]
	if weight_type == 'output':
		output_node_matrix = getNodeMatrix(output_layer)
		hidden_node_matrix = getNodeMatrix(hidden_layer)
		sigma = target_values - output_node_matrix
		sigma *= activateMatrix(output_node_matrix, 'sigmoid', prime=True)
		sigma = sigma.dot(hidden_node_matrix.T)
		return sigma
	#input
	elif weight_type == 'hidden':
		weight_matrix = getWeightMatrix(output_layer)
		sigma = target_values - getNodeMatrix(output_layer)
		output_node_matrix = getNodeMatrix(output_layer)
		sigma *= activateMatrix(output_node_matrix, 'sigmoid', prime=True)
		sigma = (weight_matrix.T).dot(sigma)
		hidden_node_matrix = getNodeMatrix(hidden_layer)
		sigma *= activateMatrix(hidden_node_matrix, 'sigmoid', prime=True)
		input_node_matrix = getNodeMatrix(network.layers[0])
		sigma = sigma.dot(input_node_matrix.T)
		return sigma

def backpass(network, target_values, learning_rate):
	output_layer = network.layers[2]
	output_weight_matrix = getWeightMatrix(output_layer)
	hidden_layer = network.layers[1]
	hidden_weight_matrix = getWeightMatrix(hidden_layer)
	input_layer = network.layers[0]

	delta_weight = get_delError(network, output_weight_matrix, 'output', target_values)
	for i in range(output_layer.dim):
		for j in range(hidden_layer.dim):
			output_layer.nodes[i].parent_weights[j].value += (delta_weight[i][j] * learning_rate)
	
	delta_weight = get_delError(network, hidden_weight_matrix, 'hidden', target_values)
	for i in range(hidden_layer.dim):
		for j in range(input_layer.dim):
			hidden_layer.nodes[i].parent_weights[j].value += (delta_weight[i][j] * learning_rate)


def train(network, X, Y, epoch, learning_rate = 0.1, print_log = True):
	for i in range(epoch):
		cost = 0
		for j in range(X.shape[0]):
			forward(network, X[[j]].T)
			backpass(network, Y[[j]].T, learning_rate)
			cost += calc_error(network, Y[[j]].T)
		cost /= X.shape[0]
		print("Epoch:", i, "- Error:", cost)
	for i in range(X.shape[0]):
		print(X[i],":", forward(network, X[[i]].T)[0][0])

			








