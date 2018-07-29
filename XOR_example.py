# XOR_examply.py
#
# 
# Spencer Kraisler 2018
#
#
# This file contains a neural network created from my simplex library
# that is optimized to emulator an XOR neural network. 
#
#

from simplex import Network
from simplex import train_with_backprop
import numpy as np

model = Network(2,3,1)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

train_with_backprop(model, X, y, learning_rate=1, epoch=1000)