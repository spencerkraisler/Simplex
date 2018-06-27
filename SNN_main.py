from SNN_objects import * 
from SNN_operations import *

model = Network(2,3,1)

# emulating XOR logic gate
train_X = np.array([[0,0],[0,1],[1,0],[1,1]])
train_y = np.array([[0],[1],[1],[0]])



# train
for i in range(100):
	print(i)
	
	# 4 possible XOR states
	for j in range(4):
		# train_X is in a weird way since 
		# having every input be a column matrix would look weird imo
		print(forward(model, train_X[[j]].T)) 
		backpass(model, train_y, .1)


print()
# print results
for j in range(4):
	print(train_X[j], forward(model, train_X[[j]].T)[0][0])


