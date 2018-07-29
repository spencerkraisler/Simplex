from simplex import *

model = Network(2,3,1)

# a simple XOR emulator

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

train_with_backprop(model, X, y, learning_rate=1, epoch=1000)

model.forward(np.array([[11231/1000.0],[10]]))