import random as rand

# this file contains all the objects for the neural network

class Network:
	def __init__(self, input_dim, hidden_dim, output_dim):
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.layers = []
		self.layers.append(Layer(self, self.input_dim, 'input'))
		self.layers.append(Layer(self, self.hidden_dim, 'hidden'))
		self.layers.append(Layer(self, self.output_dim, 'output'))

	# activation function is automatically sigmoid, but it can be changed here quickly
	def initActivationFunctions(self, input_activation, hidden_activation, output_activation):
		self.layers.append(Layer(self, self.input_dim, 0, input_activation))
		self.layers.append(layer(self, self.hidden_dim, 1, hidden_activation))
		self.layers.append(Layer(self, self.output_dim, 2, output_activation))

class Layer:
	def __init__(self, network, dim, layer_type, activation = 'sigmoid', bias = 1): # the bias parameter is the value of bias
		self.network = network
		self.dim = dim
		self.layer_type = layer_type
		self.activation = activation
		self.bias = bias
		self.nodes = []
		for i in range(dim):
			self.nodes.append(Node(self, rand.random(), self.layer_type, i, self.activation)) # nodes
		self.nodes.append(Node(self, self.bias, 'bias', dim, 'identity')) # bias node

	# this value returns all the node values in a layer in a regular array
	def getNodeValues(self):
		node_array = []
		if self.layer_type != 'output':
			for i in range(self.dim + 1):
				node_array.append(self.nodes[i].value)
		else:
			for i in range(self. dim):
				node_array.append(self.nodes[i].value)
		return node_array

	# accepts an np.array and updates all values in layer accordingly
	def updateNodeValues(self, new_node_values_matrix):
		for i in range(new_node_values_matrix.shape[0]):
			if self.nodes[i].node_type != 'bias':
				self.nodes[i].value = new_node_values_matrix[i][0]

class Node:
	def __init__(self, layer, value, node_type, node_index, activation):
		self.layer = layer
		self.value = value
		self.node_type = node_type # either input, output, hidden, or bias
		self.node_index = node_index
		self.activation = activation

		self.parent_weights = []
		if (self.node_type == 'hidden'):
			parent_layer = self.layer.network.layers[0]
			for i in range(parent_layer.dim + 1): # the + 1 is for the bias node in the input layer
				self.parent_weights.append(Weight(parent_layer.nodes[i], self, rand.uniform(-1,1)))
		elif (self.node_type == 'output'):
			parent_layer = self.layer.network.layers[1]
			for i in range(parent_layer.dim + 1):
				self.parent_weights.append(Weight(parent_layer.nodes[i], self, rand.uniform(-1,1)))

	def getWeightValues(self):
		parent_weight_values = []
		for i in range(len(self.parent_weights)):
			parent_weight_values.append(self.parent_weights[i].value)
		return parent_weight_values

class Weight:
	def __init__(self, parent_node, child_node, value):
		self.parent_node = parent_node
		self.child_node = child_node
		self.value = value


