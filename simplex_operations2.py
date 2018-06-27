# simplex_operations2.py

import math
import numpy as np 
from scipy import linalg

def calc_error(network, target_values):
	output_layer = network.layers[2]
	total_error = 0
	for i in range(len(target_values)):
		total_error += 0.5 * (target_values[i] - output_layer.nodes[i].value) ** 2
	total_error /= len(target_values)
	return total_error


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
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			X[i][j] = activate(X[i][j], function, prime)
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



# the following functions are designed to make backpropagation method more readable

# returns the change in output w.r.t. change in net input of a node
def get_delOut_delNet(node):
	function = node.activation
	value = node.value
	return activate(value, function, prime = True)

# returns the change in net input of child node w.r.t. change in output of parent node
def get_delNet_delOut(node, parent_node):
	return node.parent_weights[parent_node.node_index].value

# returns the change in net input of a node w.r.t. change in weight value
def get_delNet_delWeight(weight):
	return weight.parent_node.value


# returns the change in the error of the output node w.r.t. change in weight value
# NOTE: returns delError of a SINGLE output node, not all output nodes
def get_delError_delWeight(weight, output_node, target_value):
	network = output_node.layer.network
	del_error_del_weight = target_value - output_node.value
	del_error_del_weight *= get_delOut_delNet(output_node)
	del_error_del_weight *= get_delNet_delWeight(weight)
	if weight.child_node.node_type == 'hidden':
		del_error_del_weight *= get_delNet_delOut(output_node, weight.child_node)
		del_error_del_weight *= get_delOut_delNet(weight.child_node)
	return del_error_del_weight

# does one iteration of 
def backpass(network, target_values, learning_rate):
	for i in range(network.layers[2].dim):
		for j in range(network.layers[1].dim):
			weight = network.layers[2].nodes[i].parent_weights[j] 
			weight.value += learning_rate * get_delError_delWeight(weight, network.layers[2].nodes[i], target_values[i])
	for i in range(network.layers[1].dim):
		for j in range(network.layers[0].dim):
			weight = network.layers[1].nodes[i].parent_weights[j]
			delta_weight = 0
			for k in range(network.layers[2].dim):
				delta_weight += get_delError_delWeight(weight, network.layers[2].nodes[k], target_values[k])
			weight.value += learning_rate * delta_weight 


def train(network, X, Y, learning_rate = 0.1, epoch = 1000, print_log = True):
	for i in range(epoch):
		cost = 0
		for j in range(X.shape[0]):
			forward(network,X[[j]].T)
			backpass(network, Y[j], learning_rate)
			cost += calc_error(network, Y[j])
		cost /= 4
		if print_log == True:
			print("Epoch:", i, "Cost:", cost)
	if print_log == True:
		for i in range(X.shape[0]):
			print(X[i], ":", forward(network,X[[i]].T)[0][0])

			








