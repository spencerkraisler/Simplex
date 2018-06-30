# simplex_main2.py
import random
from snn_objects import *
from snn_operations import *
import numpy as np

# create network
model = Network(2,3,1)

# training data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# train
train(model, X, y, epoch=3000, learning_rate=1)


