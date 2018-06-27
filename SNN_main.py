from SNN_objects import * 
from SNN_operations import *

model = Network(2,3,1)
train_X = np.array([[0,0],[0,1],[1,0],[1,1]])
train_y = np.array([[0],[1],[1],[0]])



# train
for i in range(100):
	print(i)
	for j in range(4):
		print(forward(model, train_X[[j]].T))
		backpass(model, train_y, .1)


print()
# print results
for j in range(4):
	print(train_X[j], forward(model, train_X[[j]].T)[0][0])


