import numpy as np


# generate matrix with random values in the range (-10,10)
# 10 classes (rows) by 100 instances (columns)
array = 10*np.random.uniform(-1,1, (10,100))

print(array)
print(type(array[0,0]))

# choose the right value for each column randomly randint(0,9)

# compute SVM Loss for each and average

# compute Softmax cross-entropy for each and average

# repeat the process 50 times