# simplex_main2.py
import random
from simplex_objects2 import *
from simplex_operations2 import *
import numpy as np

model = Network(2,10,1)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

test_weight = model.layers[1].nodes[0].parent_weights[0]
#print(test_weight.value)


train(model, X, Y,  1, 10000, print_log = True)
print(forward(model, X[[0]].T))
print(forward(model, X[[1]].T))
print(forward(model, X[[2]].T))
print(forward(model, X[[3]].T))


