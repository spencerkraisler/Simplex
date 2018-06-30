# simplex_objects2.py

import random as rand



# network object: contains an array of layers
class Network:
	def __init__(self, input_dim, hidden_dim, output_dim):
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.layers = []
		self.layers.append(Layer(self, self.input_dim, 'input', 'sigmoid', True))
		self.layers.append(Layer(self, self.hidden_dim, 'hidden', 'sigmoid', True))
		self.layers.append(Layer(self, self.output_dim, 'output', 'sigmoid', False))

class Layer:
	def __init__(self, network, dim, layer_type, activation, biased):
		self.network = network
		self.dim = dim
		self.layer_type = layer_type
		self.activation = activation
		self.biased = biased
		self.nodes = []

		# initializes nodes
		for i in range(self.dim):
			self.nodes.append(Node(0.0, self, self.layer_type, i, self.activation))
		if self.biased == True:
			self.nodes.append(Node(1.0, self, 'bias', self.dim + 1, activation = None))


	def updateNodeValues(self, new_node_values_matrix):
		for i in range(new_node_values_matrix.shape[0]):
			if self.nodes[i].node_type != 'bias':
				self.nodes[i].value = new_node_values_matrix[i][0]


class Node:
	def __init__(self, value, layer, node_type, node_index, activation):
		self.value = value
		self.layer = layer
		self.node_type = node_type
		self.node_index = node_index
		self.activation = activation
		self.parent_weights = []

		if node_type != 'bias':
			if node_type == 'output':
				parent_layer = self.layer.network.layers[1]
				if parent_layer.biased == True:
					for i in range(parent_layer.dim + 1):
						self.parent_weights.append(Weight(rand.uniform(-1,1), parent_layer.nodes[i], self))
				else:
					for i in range(parent_layer.dim ):
						self.parent_weights.append(Weight(rand.uniform(-1,1), parent_layer.nodes[i], self))
			elif node_type == 'hidden':
				parent_layer = self.layer.network.layers[0]
				if parent_layer.biased == True:
					for i in range(parent_layer.dim + 1):
						self.parent_weights.append(Weight(rand.uniform(-1,1), parent_layer.nodes[i], self))
				else:
					for i in range(parent_layer.dim ):
						self.parent_weights.append(Weight(rand.uniform(-1,1), parent_layer.nodes[i], self))


	def getWeightValues(self):
		parent_weight_values = []
		for i in range(len(self.parent_weights)):
			parent_weight_values.append(self.parent_weights[i].value)
		return parent_weight_values

class Weight:
	def __init__(self, value, parent_node, child_node):
		self.value = value
		self.parent_node = parent_node
		self.child_node = child_node

