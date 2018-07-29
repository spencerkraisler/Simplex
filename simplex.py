# simplex.py
#
#
# Spencer Kraisler 2018
# 
#
# This file contains method and objects that create a working 3-layered sequential neural network. 
# This file does not use node or weight objects, only layer and network objects.
# Node and weight values are stored in numpy matrices and handled with various methods used in this file.
# In mathematics, a simplex is the generalization of a triangle for higher dimensions.
# (2d: triangle, 3d: tetrahedron, 4d: 4-simplex, etc.)
# Since the number 3 is entangled in simplicies, and this is for a 3 layered network, I accepted that term as the name :)
#
#

from scipy import linalg
import math
import numpy as np
import random as rand


class Network:
	def __init__(self, input_dim, hidden_dim, output_dim):
		# create array of layers
		self.layers = []
		self.layers.append(Layer(self, input_dim, 0, 'sigmoid', True))
		self.layers.append(Layer(self, hidden_dim, 1, 'sigmoid', True))
		self.layers.append(Layer(self, output_dim, 2, 'sigmoid', False))

	# takes an input vector and passes it through a network
	# changes node values of network after they are passed through the act. function
	def forward(self, input_matrix):
		input_matrix = appendBias(input_matrix)
		if input_matrix.shape != self.layers[0].node_matrix.shape:
			print(input_matrix.shape, self.layers[0].node_matrix.shape)
			print("Error: parameter input vector does not align with network input vector")
		else:
			size = len(self.layers)
			node_matrix = input_matrix
			self.layers[0].node_matrix = node_matrix
			# using len(network.layers) in case you wish to expand this library
			# to include 3+ SNNs
			for i in range(1, size):
				current_layer = self.layers[i]
				node_matrix = current_layer.weight_matrix.dot(node_matrix)
				node_matrix = activate(node_matrix, current_layer.activation)
				
				# output layer has no bias
				if i < size - 1:
					node_matrix = appendBias(node_matrix) 
				current_layer.node_matrix = node_matrix
		return node_matrix

class Layer:
	def __init__(self, network, dim, layer_index, activation, biased):
		self.network = network
		self.dim = dim
		self.layer_index = layer_index
		self.activation = activation

		# output layer has no bias
		self.node_matrix = np.array([])
		if self.layer_index == 2: self.node_matrix = np.zeros((self.dim, 1))
		else: 
			self.node_matrix = np.zeros((self.dim + 1, 1))
			# if layer has no bias, then bias = 0 (for conveinece)
			if biased == True:
				self.node_matrix[self.dim][0] = 1

		# input layer does not have a weight matrix
		self.weight_matrix = np.array([])
		self.gradient = np.array([])
		if self.layer_index != 0:
			parent_layer = self.network.layers[layer_index - 1]
			self.weight_matrix = getRandMatrix(np.zeros((self.dim, parent_layer.dim + 1)))
			# stores dCost/dWeights for backpropagation
			self.gradient = np.zeros((self.weight_matrix.shape))

# this method passes a node matrix through an activation function (e.g. sigmoid)
# prime is derivatve, so prime = True means the derivative of the function
def activate(X, function, prime = False):
	if function == 'sigmoid':
		if prime == False: return 1.0 / (1 + 2.781 ** -X)
		else: return X * (1 - X)
	if function == 'tanh':
		if prime == False: 
			# cannot pass numpy arrays through math.tanh
			tanh_X = np.zeros((X.shape))
			for i in range(X.shape[0]):
				for j in range(X.shape[1]):
					tanh_X[i][j] = math.tanh(X[i][j])
			return tanh_X
		else: return 1 - X ** 2
	if function == 'identity':
		if prime == False: return X
		else: return np.ones((X.shape))
	else: print("Error: function is not registered in activate() method")

# this method creates a matrix of random values that is the same shape as X
def getRandMatrix(X):
	R = np.zeros((X.shape[0], X.shape[1]))
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			R[i][j] = rand.uniform(-1,1)
	return R

# appends bias value to bottom of a column node matrix
def appendBias(node_matrix):
	biased_node_matrix = list(node_matrix)
	biased_node_matrix.append(np.array([1]))
	biased_node_matrix = np.array(biased_node_matrix)
	return biased_node_matrix

# updates the delCost/delWeight of every weight in the network
	# this is a network object method because if I do ever add
	# 3+ layer compatibility to this library, updating the gradients 
	# will involve gradient values from other layers
def updateGradient(network, Y):
	size = len(network.layers)
	output_layer = network.layers[2]
	hidden_layer = network.layers[1]
	input_layer = network.layers[0]

	# output layer
	gradient = Y - output_layer.node_matrix
	gradient *= activate(output_layer.node_matrix, 'sigmoid', prime=True)
	gradient = gradient.dot(hidden_layer.node_matrix.T)
	output_layer.gradient = gradient

	# hidden layer
	for i in range(size - 2, 0, -1):
		gradient = Y - output_layer.node_matrix
		gradient *= activate(output_layer.node_matrix, 'sigmoid', prime=True)
		gradient = output_layer.weight_matrix.T.dot(gradient)
		gradient *= activate(hidden_layer.node_matrix, 'sigmoid', prime=True)
		gradient = gradient[:gradient.shape[0] - 1] # remove bias node
		gradient = gradient.dot(input_layer.node_matrix.T)
		hidden_layer.gradient = gradient


def backpass(network, Y, learning_rate):
	updateGradient(network, Y)
	for i in range(1, len(network.layers)):
		layer = network.layers[i]
		layer.weight_matrix += layer.gradient * learning_rate

def cost_supervised(network, X, Y):
	output_layer = network.layers[len(network.layers) - 1]
	cost = 0
	error_matrix = np.zeros((Y.shape))
	for i in range(X.shape[0]):
		network.forward(X[[i]].T)
		error_matrix = 0.5 * (Y - output_layer.node_matrix) ** 2
		total_error = 0
		for j in range(Y.shape[0]):
			total_error += error_matrix[i][0]
		total_error /= error_matrix.shape[0]
		cost += total_error
	cost /= X.shape[0]
	return cost

def train_with_backprop(network, X, Y, learning_rate, epoch, print_info=True):
	for i in range(epoch):
		for j in range(X.shape[0]):
			network.forward(X[[j]].T)
			backpass(network, Y[j], learning_rate)
	if print_info == True:
		for i in range(Y.shape[0]):
			print(X[[i]][0][0], network.forward(X[[i]].T)[0][0])

	
