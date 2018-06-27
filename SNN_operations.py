import three_layer_SNN_objects as SNNObjects
import numpy as np 
from scipy import linalg
import math

# contains sigmoid and tanh and their derivatives
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

# accepts a matrix and runs every element through the activation function
def activateMatrix(X, function, prime = False):
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			X[i][j] = activate(X[i][j], function, prime)
	return X

#  appends bias value to bottom of a column node matrix
def appendBias(node_matrix, bias):
	node_matrix = list(node_matrix)
	node_matrix.append(np.array([bias]))
	node_matrix = np.array(node_matrix)
	return node_matrix

# takes a layer and returns a column matrix of the node values, including bias node
def getNodeMatrix(layer):
	network = layer.network
	node_matrix = []
	if (layer.layer_type == 'hidden') or (layer.layer_type == 'input'):
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

# takes a layer and returns a matrix of parent weights
def getWeightMatrix(layer):
	weight_matrix = []
	for i in range(layer.dim):
		weight_matrix.append(layer.nodes[i].getWeightValues())
	weight_matrix = np.array(weight_matrix)
	return weight_matrix

# regular propagation: takes a matrix of input values and returns
# the neural network output in the form of a matrix
# updates node values
def forward(network, input_matrix):
	network.layers[0].updateNodeValues(input_matrix)
	bias = network.layers[0].nodes[network.layers[0].dim].value
	input_matrix = appendBias(input_matrix, bias)
	next_node_matrix = input_matrix
	for i in range(1, 3):
		weight_matrix = getWeightMatrix(network.layers[i])
		next_node_matrix = weight_matrix.dot(next_node_matrix)
		next_node_matrix = activateMatrix(next_node_matrix, network.layers[i].activation)
		network.layers[i].updateNodeValues(next_node_matrix)
		if network.layers[i].layer_type != 'output':
			bias = network.layers[i].nodes[network.layers[i].dim].value
			next_node_matrix = appendBias(next_node_matrix, bias)
	return next_node_matrix


### back propagation stuff
# these functions seem redundant but they make the get_delError_delWeight method look cleaner

# takes a node and returns the dericatives of the output w.r.t. the net input
def get_delOut_delNet(node):
	function = node.activation
	value = node.value
	return activate(value, function, prime = True)

# returns the change in net input of child node w.r.t. change in output of parent node
def get_delNet_delOut(node, parent_node):
	return node.parent_weights[parent_node.node_index].value

# returns the change in net input of a node w.r.t. change in parent weight value
def get_delNet_delWeight(weight):
	return weight.parent_node.value


# calc dE/dW for a single output_node and weight
def get_delError_delWeight(weight, output_node, target_value):
	del_error = 1
	if weight.child_node.node_type == 'hidden':
		del_error = target_value - output_node.value
		del_error *= get_delOut_delNet(output_node)
		del_error *= get_delNet_delOut(output_node, weight.child_node)
		del_error *= get_delOut_delNet(weight.child_node)
		del_error *= get_delNet_delWeight(weight)
	else:
		del_error *= target_value - output_node.value 
		del_error *= get_delOut_delNet(output_node)
		del_error *= get_delNet_delWeight(weight)
	return del_error

# back propagation method (this is made for 3 layer networks)
def backpass(network, target_values, learning_rate):
	# output layer weights
	for i in range(network.layers[2].dim):
		output_node = network.layers[2].nodes[i]
		for j in range(len(output_node.parent_weights)):
			weight = output_node.parent_weights[j]
			del_weight = get_delError_delWeight(weight, output_node, target_values[i][0])

			weight.value += del_weight * learning_rate
			
	for i in range(network.layers[1].dim):
		node = network.layers[1].nodes[i]
		for j in range(len(node.parent_weights)):
			weight = node.parent_weights[j]
			
			# dE/dNode = dE0/dNode + dE1/dNode + ... for every output node
			del_weight = 0
			for k in range(network.layers[2].dim):
				del_weight += get_delError_delWeight(weight, network.layers[2].nodes[k], target_values[k][0])
			
			weight.value = del_weight * learning_rate




